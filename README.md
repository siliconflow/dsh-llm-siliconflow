# @siliconflow/dsh-llm-siliconflow

面向 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) LLM 接缝的 SiliconFlow chat-completions 适配器插件：用直接 `fetch` + SSE（由 `eventsource-parser` 分帧）把 SiliconFlow 的 OpenAI 兼容线上格式翻译成 `StreamChunk` 协议。SiliconFlow 托管着广泛的开源模型目录，其中包含 delta 携带 `reasoning_content` 的推理模型（DeepSeek-R1、QwQ、Kimi-K2-Thinking）——适配器把该通道翻译成 harness reasoning 块，并在工具调用轮次按这些模型的要求将其回传。

本包拥有 `siliconflow` 提供方路由，因此部署只需提供一个 SiliconFlow API key 即可使用。其模型选择器从实时 `GET /models?sub_type=chat` 列表按端点顺序填充；配置的 `models` 列表是在没有 key 或发现失败时展示的回退目录。这是一个纯 OpenAI 兼容端点，没有 `thinking`/`reasoning_effort` 开关，因此适配器不暴露任何推理档位元数据、也不序列化任何推理档位字段：推理模型通过其目录 id 选择，请求上显式指定 `reasoningEffort` 会在网络 I/O 前以 `UNSUPPORTED_REASONING_EFFORT` 被拒绝。为 `siliconflow` 注册另一个适配器会抛出 `LlmError('DUPLICATE_ADAPTER')`。

包根导出 Cordis 插件契约与 `SiliconFlowAdapter`；线上序列化、SSE 解析、chunk 翻译与发现辅助函数不属于该根契约。

本项目由 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 的 dsh agent 开发完成——实现、单元测试与覆盖率、工程化质量门禁、GitHub 仓库与 CI、文档，均在一个 dsh 会话内完成。

## 安装

一行命令装进任意 profile：

```sh
dsh plugin --profile <name> add @siliconflow/dsh-llm-siliconflow
```

该命令由 `dsh plugin` 转发给 pnpm，把本包装进 profile 并把它声明的 bundle patch（`cordis.patch.yml`，自动挂载 `siliconflow` 路由）合并进 `dsh.profile.bundles`。`@deepseek-ai/dsh-*` 以 peerDependencies 声明，由 harness 安装闭包在运行时提供，无需重复打包。

装完后运行随包发布的配置向导，交互式地填 key、拉取实时模型列表并把它设为默认渠道：

```sh
dsh-siliconflow-setup
```

向导依次：询问是否把 SiliconFlow 设为默认渠道 → 未找到 `SILICONFLOW_API_KEY` 时引导填写并写入 `$DSH_HOME/.credentials.yaml` → 用 key 做 live discovery 拉取 `/models?sub_type=chat` 列表（失败则回退到内置目录）→ 选择默认模型 → 写入 `$DSH_HOME/settings.yaml` 的 `agent-default-model`。它只读 `$DSH_HOME`（缺省 `~/.dsh`），不改动 harness 本体。

不想用向导时，装完填一个 key 即可使用：

```sh
export SILICONFLOW_API_KEY=sk-...   # 或写入 $DSH_HOME/.credentials.yaml
```

在包发布到 npm 之前，先用 `pnpm install && pnpm build` 构建出 `lib/`，再从本地路径安装：

```sh
dsh plugin --profile <name> add /path/to/dsh-llm-siliconflow
```

本代码派生自 MIT 许可的 DeepSeek Harness `llm-deepseek` 适配器；见 [LICENSE](LICENSE)。

## 启动

插件随 profile 挂载，启动方式不影响其可用性——`npx`、本地或全局安装的 `dsh` 都读同一个 `$DSH_HOME`（缺省 `~/.dsh`）下的 profile。

### 用 npx 启动（无需本地安装 dsh）

```sh
npx @deepseek-ai/dsh plugin --profile web add @siliconflow/dsh-llm-siliconflow  # 装到 web profile
npx @siliconflow/dsh-llm-siliconflow                                            # 交互式 setup
npx @deepseek-ai/dsh web                                                        # 打开 http://127.0.0.1:3080
```

`npx @siliconflow/dsh-llm-siliconflow` 运行本包唯一 bin（`dsh-siliconflow-setup`），与被装进哪个 profile 无关——setup 只读写 `$DSH_HOME` 下的 credentials 与 settings。

### 后台启动

后台只是进程脱离终端，不影响插件加载；唯一要求是 `dsh-siliconflow-setup` 是交互式向导，须先前台、在有 TTY 的环境跑一次。

```sh
# web UI 常驻后台（默认 http://127.0.0.1:3080；远程访问需把 host 配成 0.0.0.0）
nohup npx @deepseek-ai/dsh web > ~/.dsh/web.log 2>&1 &

# headless 后台跑一次性任务
nohup npx @deepseek-ai/dsh --profile headless "任务" > ~/.dsh/task.log 2>&1 &
```

## 配置

```yaml
- id: llm-siliconflow
  name: '@siliconflow/dsh-llm-siliconflow'
  config:
    apiKeyEnv: SILICONFLOW_API_KEY # default; resolved per request via ctx.credentials, then the environment
    baseURL: https://api.siliconflow.cn/v1 # optional; $SILICONFLOW_BASE_URL then the public API when omitted
    maxTokens: 8192         # optional positive per-request output cap; this is the default
    streamIdleTimeoutMs: 300000 # optional; positive finite Node timer delay; five-minute default
    retryPolicy:            # optional; omission uses bounded normal defaults
      mode: normal          # normal | always
      backoff:
        initialDelayMs: 500
        maxDelayMs: 10000
        jitterRatio: 0.1
    defaultContextWindow: 32768 # optional positive-integer fallback; this is the default
    models:                 # optional; the fallback catalog shown when discovery cannot run
      - id: zai-org/GLM-5.2
      - id: deepseek-ai/DeepSeek-V4-Flash
```

插件把单个提供方路由 `siliconflow` 连同其已解析的 `retryPolicy` 一并注册。请求用 `provider: siliconflow` 选中它；其 `model` 原样作为线上 `model` 字符串透传，因此更换 SiliconFlow 模型不需要生命周期级重新注册。线上模型 id 是 SiliconFlow 的 `org/model` 写法（如 `deepseek-ai/DeepSeek-V4-Flash`），绝不是短别名。省略 `models` 时保留一份由六个当前托管对话模型组成的小回退目录；显式列表会替换这些默认值，而 `models: []` 则一个都不通告。目录条目通过 `ctx.llm.listModels('siliconflow')` 暴露给 ACP 编辑器与 Web 选择器这类客户端，但始终是建议性的：未列出的模型 id 依然原样透传。省略的条目名默认等于其 id。

`contextWindow` 按模型可选。`ctx.llm.resolveModelInfo('siliconflow', model).context` 先返回精确值——来自配置条目或温热的发现缓存——再对未被任何来源定容的模型回退到 `defaultContextWindow`。适配器默认值是 32,768；SiliconFlow 目录大致横跨 8k 到 1M 上下文（GLM-5.2、DeepSeek-V4-Pro/Flash 均支持 1M），因此披露了上下文的发现列表是权威值，回退值仅在没有任何来源披露时使用。对压力敏感的插件由此获得部署自有的容量，而不把模型选择器当作权威。

`maxTokens` 是对话请求的适配器级输出上限，默认 8,192。目录条目可携带自己的 `maxTokens`，对该模型优先生效；没有该字段的条目以及任何未列出的透传 id 解析为配置值。精确模型解析把胜出者暴露为 `defaultMaxTokens`；`LlmRuntime` 在 agent 循环写 `request/header` 之前把该值物化进 `GenerateOptions.maxTokens`，因此线上请求可重建。显式请求或 `AgentOptions.maxTokens` 值优先，并被序列化为 `max_tokens`。适配器不会把该请求预算对照 `contextWindow` 裁剪；上下文更小或有提供方输出限制的部署必须配置兼容的 `maxTokens`。

`streamIdleTimeoutMs` 约束每次未完成的提供方读取（包括首次 `fetch`），不计消费者在 chunk 之间花费的时间。SSE 注释会为未完成的读取重新计时作为传输活动，但永远不会成为 `StreamChunk` 值或会话日志事件。一次调用全程只有一个稳定的中止信号同时到达请求与响应体读取器；超时会停止传输并抛出 `LlmError('TIMEOUT')`，而更早的调用方中止抛出 `LlmError('ABORTED')`。适配器每次 `stream()` 调用只发一次提供方请求；它把配置的策略注册为提供方元数据，`dsh-llm-retry` 再在持久化的 agent 步骤边界单独执行它。

## 动态模型发现

`listModels` 通告实时对话列表而非手工维护的快照：它用已解析的 key 询问 `GET {baseURL}/models?sub_type=chat`，按**端点顺序**保留回复，并缓存五分钟。发现是建议性的、尽力而为的——缺少 key、端点不可达、凭据被拒、或回复不可读，都会回退到配置的 `models` 列表而非破坏选择器，因为空目录会整个隐藏该提供方。成功的列表同样服务于精确模型解析：静态目录未命名的模型仍能得到其列表披露的上下文窗口与输出上限。

配置面的「获取可用模型」动作通过 `ctx.llm.discoverModels('llm-siliconflow', …)` 使用同一询问：表单里输入的 key 优先，否则探测已存凭据或环境变量；没有 key 的路由以未认证方式探测，因此配置面仍能回答「这个端点服务哪些模型」。

## 动态配置（settings + credentials）

连接事实不在加载时冻结。`resolveAdapterOptions` 是从原始配置到已校验事实的唯一显式解析步骤，适配器通过 thunk **每个操作一次** 重读它们：base URL、目录、请求默认值与空闲预算都在下一个请求生效，而进行中的流保留其启动时的事实。两个可选接缝为该 thunk 供料：

- **`ctx.settings`** —— 插件用同样的 `Config` schema 注册 `llm-siliconflow` 命名空间，并把其 `cordis.yml` 条目作为组合 `base`，因此用户设置文档中的 `llm-siliconflow:` 段可以在不重启的情况下覆盖任意字段。未挂载 settings 服务时仅由条目配置驱动适配器，行为不变。一个通过 schema 但越过 schema 之外界限（如重复目录 id）的实时设置快照会保留最后的好事实并记录失败；条目配置本身仍会在插件加载时失败。
- **`ctx.credentials`** —— API key 每个流调用解析一次，且来自提供端点的 *同一个* 已解析快照。配置只携带 `apiKeyEnv`，绝不携带明文 key：该引用通过凭证接缝解析，未挂载该接缝时通过受信任的环境层解析。由于凭证事实随连接事实一起传递，解析器拒绝的设置快照既不贡献其端点也不贡献其 key。每个解析出的 key 在使用前都做格式检查，因此 HTTP 头无法承载的值会以 `LlmError('INVALID_CREDENTIAL')` 被拒绝，并指出失败的入口点——绝不包含 key 的任何部分。任何地方都没有 key 的请求以 `MISSING_CREDENTIAL` 失败，并指出每个配置入口点，而路由保持注册、目录保持可浏览。

唯一在注册时捕获的事实是重试策略：当其解析值变化时，插件原地重注册路由（同一适配器实例、一个同步区段），因此 `ctx.llm.providerRetryPolicy('siliconflow')` 始终报告当前策略。插件还会在可配置提供方目录（`ctx.llm.listConfigurableProviders()`）中声明其路由：提供方 `siliconflow`，settings 命名空间 `llm-siliconflow`，settings 路径为空。

## 应用归属

每个请求都携带 dsh-llm 的 `attributionHeaders()` 提供的共享归属头——标识 harness 的强制 `User-Agent` 基线。在凭证解析之后，每个提供方请求都携带 `x-siliconflow-harness-user-id`（来自 `@deepseek-ai/dsh-anonymous-user-id` 的稳定匿名 id）；携带 `GenerateOptions.sessionId` 的请求还会把该精确值作为 `x-siliconflow-harness-session-id` 发送，而没有会话的直接调用省略该会话头。`GenerateOptions.purpose` 为 `compaction` 的请求额外携带 `x-siliconflow-harness-compact: 1`。这三个头都发往已解析的 `baseURL`，并保持在请求体与模型可见内容之外。

## 线上格式说明

- 仅流式（`stream_options.include_usage` 始终开启）。`usage` 可能附在结束 chunk 上，也可能作为尾随的仅 usage chunk 出现——翻译器把两者都推迟到 `[DONE]`，因此 `usage` 始终先于 `finish`，且 `finish` 之后无内容。
- 适配器从不发送 `thinking` 或 `reasoning_effort`；SiliconFlow 端点没有这些开关，推理模型通过 id 选择。
- 推理模型首个 chunk 携带 `reasoning_content: ""` —— 已处理（不会产生多余 reasoning 块）。
- **推理回传规则**：在携带工具调用的 assistant 轮次，`reasoning_content` 会被序列化回历史（托管的 DeepSeek-R1 类模型所要求）；无工具调用的轮次则丢弃（反正被忽略——省 token）。
- 缓存记账：`cacheReadTokens` ← `prompt_cache_hit_tokens` / `prompt_tokens_details.cached_tokens`；SiliconFlow 不报告缓存写入指标。

## 错误

非 2xx 响应抛出带稳定码的 `LlmError`：`AUTH`（401/403）、`QUOTA`（提供方详情指明配额、余额或额度耗尽的响应）、`RATE_LIMIT`（其他 429）、`CONTEXT_WINDOW_EXCEEDED`（400 且其提供方 code、type 或 message 指明上下文溢出）、`INVALID_REQUEST`（其他 400）、`SERVER`（5xx）、其余 `HTTP_<status>`。其可序列化的 `failure` 保留 HTTP 状态，以及存在时的有效正 `Retry-After` 秒数/日期延迟和 `x-request-id`。响应前的传输失败（DNS、连接拒绝、TLS、代理）抛出 `TRANSPORT`，指明已配置端点并把原始拒绝链为 `cause`；调用方中止抛出 `ABORTED`，循环的取消信号保持权威。协议违规抛出 `STREAM_CLOSED`（无 `[DONE]`）或 `MALFORMED_RESPONSE`（坏 JSON payload）。未知的线上 `finish_reason`（如 `content_filter`、`insufficient_system_resource`）变为 `finish {kind: 'error', failure}` chunk；一个 `stop`（或缺省）结束但未打开任何内容块的已完成流变为 `finish {kind: 'error'}`，码为 `EMPTY_RESPONSE`（默认策略会重试）。

## Model Experience

### SiliconFlow 请求

#### 模型看到什么

选中的 SiliconFlow 模型收到 harness 系统提示、消息历史、工具 schema、停止序列与调用配置，不含适配器编写的提示文本。在先前携带工具调用的 assistant 轮次，其推理内容按要求回传；无工具调用轮次的推理被省略。

#### Token 影响

提供方分词决定精确输入。条件性推理回传增加工具往返上下文，而丢弃其他推理避免重复付费；缓存读取用量在可用时上报。

#### KV Cache 影响

未变化的组装前缀有资格被提供方缓存复用，本适配器在用量中上报。模型路由变更或任何上游提示、schema、前缀、历史变更都可能从第一个变化 token 起阻止复用；推理回传在工具往返期间追加。

### SiliconFlow 响应

#### 模型看到什么

推理、文本与原始字符串工具参数被翻译成 harness chunk，供循环记录与组装。

#### Token 影响

生成 token 遵循请求记录的 `maxTokens`；只有循环保留的块影响后续输入。

#### KV Cache 影响

循环保留的响应块追加到下一个请求并保留其更早的可复用前缀；被丢弃的块没有后续缓存影响。更换提供方或模型会选择不同的缓存域。

## Known Limitations and Deferred Work

- **回退 `models` 列表是手工维护的** —— 六个默认值是一份小快照，仅在发现无法运行时展示；实时列表才是权威目录。
- **发现不跨 baseURL 变化缓存** —— 缓存按端点键控，因此改指路由会在下一次 `listModels` 重新询问。
- **settings 的 `models` 列表整体替换组合列表** —— settings 层合并在字段粒度进行，数组是一个字段；按条目合并目录需要键控结构。
- **未映射 `tool_choice`** —— 不属于核心词汇表（MVP 裁剪，与 pi-ai 和 DeepSeek 双胞胎相同）。
- **请求使用原始 `fetch`，而非 `@cordisjs/plugin-http`** —— 没有共享代理/拦截配置；待有第二个直接 fetch 适配器需要时再采用（`TODO(http)`）。
- **序列化把 user 与 tool-result 内容扁平化为文本块** —— 插件添加的块类型被跳过，空工具输出以字面 `(no output)` 上线。
- **图片内容被拒绝** —— 这里的 chat-completions 线上路由仅文本；多模态 SiliconFlow 路由需要自己的内容序列化器。
