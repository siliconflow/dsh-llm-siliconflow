/**
 * Serialize harness messages into SiliconFlow chat completions. User text is
 * joined; assistant text becomes `content`, tool calls become `tool_calls`,
 * and tool results become separate tool messages. Assistant reasoning is
 * replayed as `reasoning_content` only on tool-call turns, as hosted reasoning
 * models (DeepSeek-R1 and siblings) require.
 *
 * **Multimodal**: image blocks in user messages are serialized as
 * OpenAI-compatible `image_url` content parts with base64 data URLs. The
 * serializer resolves image attachments through the optional attachment-store
 * resolver; when no resolver is available, images are replaced with the
 * `OFFLOADED_IMAGE_TEXT` sentinel. Tool-result content is flattened to text
 * (images within tool results are also replaced with the sentinel). Unknown
 * declaration-merged block types retain the adapter's documented extension
 * fallback.
 *
 * @module dsh-llm-siliconflow/serialize
 */

import { OFFLOADED_IMAGE_TEXT } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, Message } from '@deepseek-ai/dsh-llm'
import type { ImageAttachmentRef } from '@deepseek-ai/dsh-attachment'
import type { WireImagePart, WireMessage, WireTextPart, WireRequest, WireTool } from './types.ts'

/** Join the text blocks of a message (used for user/tool-result content). */
function flattenText(blocks: ContentBlock[]): string {
  return blocks
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

/**
 * Replace image blocks (including nested ones in tool results) with the
 * `OFFLOADED_IMAGE_TEXT` sentinel so the provider sees a coherent text
 * placeholder instead of silently dropped bytes. This is the no-store fallback;
 * the multimodal path resolves real base64 data URLs instead.
 */
function replaceImagesWithSentinel(blocks: readonly ContentBlock[]): ContentBlock[] {
  return blocks.map((block) => {
    if (block.type === 'image') return { type: 'text', text: OFFLOADED_IMAGE_TEXT }
    if (block.type === 'tool-result') {
      return { ...block, content: replaceImagesWithSentinel(block.content) }
    }
    return block
  })
}

/** Whether this message's content has any image blocks (user side). */
function hasImages(blocks: readonly ContentBlock[]): boolean {
  return blocks.some(block => block.type === 'image')
}

/**
 * Build the wire content parts for a user message: text blocks become `text`
 * parts, image blocks become `image_url` parts. Non-text/non-image blocks are
 * skipped (merge-extensible fallback). An image block whose attachment cannot
 * be resolved becomes the `OFFLOADED_IMAGE_TEXT` sentinel text part.
 */
async function serializeUserContent(
  blocks: ContentBlock[],
  resolveImage: (ref: ImageAttachmentRef) => Promise<string | undefined>,
): Promise<string | (WireTextPart | WireImagePart)[]> {
  const hasImage = hasImages(blocks)
  if (!hasImage) return flattenText(blocks)

  const parts: (WireTextPart | WireImagePart)[] = []
  for (const block of blocks) {
    if (block.type === 'text') {
      parts.push({ type: 'text', text: block.text })
    } else if (block.type === 'image') {
      const url = await resolveImage(block.attachment)
      if (url !== undefined) {
        parts.push({ type: 'image_url', image_url: { url } })
      } else {
        parts.push({ type: 'text', text: OFFLOADED_IMAGE_TEXT })
      }
    }
    // Unknown block types are skipped (merge-extensible ContentBlockMap).
  }
  return parts
}

/**
 * Serialize one assistant message (text + reasoning + tool calls). Image
 * blocks in assistant content (forward compatibility) are replaced with the
 * sentinel text — the chat-completions wire route carries images only in user
 * content.
 */
function serializeAssistant(message: Message): WireMessage {
  const text = flattenText(message.content)
  const reasoning = message.content
    .filter(block => block.type === 'reasoning')
    .map(block => block.text)
    .join('')
  const toolCalls = message.content
    .filter(block => block.type === 'tool-call')
    .map(block => ({
      id: block.id,
      type: 'function' as const,
      function: { name: block.name, arguments: block.arguments },
    }))

  return {
    role: 'assistant',
    // Text-less turns send "" — NEVER null. Pure tool-call turns: the
    // upstream samples replay message.content verbatim (which is "") and some
    // gateways reject null outright. Reasoning-ONLY turns (a reasoning model
    // can answer entirely in the reasoning channel): the API rejects
    // null-content/no-tool_calls assistant messages, and since the message
    // sits durably in the session log, a null here bricks every later turn.
    content: text,
    // Reasoning passback rule: reasoning_content must return on tool-call
    // turns; it is ignored on plain turns, so we drop it there to save tokens.
    ...toolCalls.length > 0 && reasoning.length > 0 ? { reasoning_content: reasoning } : {},
    ...toolCalls.length > 0 ? { tool_calls: toolCalls } : {},
  }
}

/**
 * Serialize the conversation. `tool-result` blocks become standalone
 * `{role: 'tool'}` messages; the harness puts each tool result in its own
 * user-role message, so a mixed user message contributes its text first and
 * its tool results as separate wire messages after.
 *
 * Image blocks within user messages are resolved through the `resolveImage`
 * callback; images within tool results are flattened to the sentinel text
 * (tool results carry structured data, not multimodal content).
 * @param messages - the harness conversation, in order.
 * @param resolveImage - resolves an image attachment ref to a base64 data URL; returns `undefined` when unavailable.
 * @returns the wire messages; order preserved, each tool result expanded into its own entry.
 */
export async function serializeMessages(
  messages: Message[],
  resolveImage: (ref: ImageAttachmentRef) => Promise<string | undefined> = () => Promise.resolve(undefined),
): Promise<WireMessage[]> {
  const wire: WireMessage[] = []
  for (const message of messages) {
    if (message.role === 'system') {
      wire.push({ role: 'system', content: flattenText(message.content) })
      continue
    }
    if (message.role === 'assistant') {
      wire.push(serializeAssistant(message))
      continue
    }
    // user role: tool results ride in user messages in the harness
    // vocabulary, but SiliconFlow wants them as role:'tool' messages.
    const toolResults = message.content.filter(block => block.type === 'tool-result')
    const userBlocks = message.content.filter(block => block.type !== 'tool-result')

    // Emit a user message for the text/image portion when it has any content.
    // A block-less user message still emits as an empty user message.
    if (userBlocks.length > 0 || toolResults.length === 0) {
      const content = await serializeUserContent(userBlocks, resolveImage)
      wire.push({ role: 'user', content })
    }

    for (const result of toolResults) {
      // Tool result content is text-only on the wire; images in tool results
      // are replaced with the sentinel.
      const flatContent = flattenText(replaceImagesWithSentinel(result.content)) || '(no output)'
      wire.push({
        role: 'tool',
        tool_call_id: result.toolCallId,
        content: flatContent,
      })
    }
  }
  return wire
}

/**
 * Build the full wire request. Always streaming (`stream: true`, usage
 * reporting on); optional fields are omitted rather than sent as null, so
 * provider defaults apply.
 * @param options - the harness request (model, history, system, tools, sampling).
 * @param resolveImage - resolves an image attachment ref to a base64 data URL; returns `undefined` when unavailable.
 * @returns the chat-completions request body.
 */
export async function serializeRequest(
  options: GenerateOptions,
  resolveImage: (ref: ImageAttachmentRef) => Promise<string | undefined> = () => Promise.resolve(undefined),
): Promise<WireRequest> {
  const messages: WireMessage[] = []
  if (options.system !== undefined) {
    messages.push({ role: 'system', content: options.system })
  }
  messages.push(...await serializeMessages(options.messages, resolveImage))

  const tools: WireTool[] | undefined = options.tools?.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }))

  return {
    model: options.model,
    messages,
    stream: true,
    stream_options: { include_usage: true },
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
    ...options.temperature !== undefined ? { temperature: options.temperature } : {},
    ...options.maxTokens === undefined ? {} : { max_tokens: options.maxTokens },
    ...options.stop !== undefined ? { stop: options.stop } : {},
  }
}
