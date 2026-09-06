# GPT-6 Astra harness support

Select Astra explicitly; existing provider profiles and model defaults keep
their configured workloads:

```sh
HI_API_KEY=sk-... hi --provider openai \
  --base-url https://api.openai.com/v1 --model gpt-6-astra \
  --reasoning-effort high "review and fix the failing test"
```

The OpenAI adapter sends `gpt-6-astra` and `openai/gpt-6-astra` requests to
`/responses`. Gateways must support that endpoint for Astra; failures do not
trigger a Chat Completions or text-tool fallback inside the adapter. Other
models keep their existing transport.

## Compatibility

- Requests use `max_output_tokens`, `store: false`, and flat function tools.
  Tool results retain the original `call_id`. Tool schemas explicitly use
  `strict: false` to preserve the harness's optional arguments and local
  validation semantics.
- Astra requests omit sampling and Chat Completions parameters, including
  `temperature`, `top_p`, and `frequency_penalty`, even during recovery.
  Explicit `minimal` reasoning becomes `low`; other supported harness levels
  are preserved. Unset reasoning leaves the model's default in effect.
- Caching uses `prompt_cache_options.ttl: "30m"`. Reported cache reads and
  writes populate the existing usage counters. The adapter does not request
  a fast or priority service tier, including for EU endpoints.
- Every completed output item is saved for stateless continuation, including
  encrypted reasoning, message phases, and tool calls in their original order.
  Replay metadata survives session serialization, is isolated from other
  provider adapters, and is invalidated when the visible message is changed
  or pruned. Opaque replay is excluded from visible text and heuristic token
  estimates; provider usage remains authoritative.
- A terminal Responses event is required before returning executable tool
  calls. Truncated streams and malformed tools produce typed errors.
- The system and skill prompts clarify action requests, existing authorization,
  consequential questions, skill precedence, bounded delegation, and
  proportionate verification while retaining enforced permission policies.

## Transport limits

This integration uses HTTP streaming and the harness's existing synchronous
tool rounds. It does not advertise native asynchronous tools or WebSocket
mid-turn steering. Existing client-side steering and delegation remain available.

Reasoning changes currently update the next request's `reasoning.effort`.
The optional `configuration_update` protocol is not yet persisted in the
harness's conversation history, so changing effort can reduce cache reuse.
Automatic Responses compaction and pro mode are not enabled by this adapter.

No model pricing, context limit, account availability, or performance result is
inferred from a model name. The existing discovery and configuration mechanisms
remain responsible for those values. Protocol tests use local canned Responses
streams; a live account smoke test is still needed to verify account access.

## Sources

- [Official GPT-6 Astra migration and prompting guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)
- [Stateless reasoning and complete output replay](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-across-calls)
- [Function tool strict mode](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)
- [Prompt caching changes](https://developers.openai.com/api/docs/guides/prompt-caching#summary-of-model-differences)
- [Reasoning updates and compatibility limits](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation)
