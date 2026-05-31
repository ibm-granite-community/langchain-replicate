# Project Advanced Coding Rules (Non-Obvious Only)

- Always pass prediction payloads through [`_adjust_prediction_input()`](../../src/langchain_replicate/_base.py:52); Replicate rejects LangChain-style `tool_choice="any"` and expects named choices serialized to JSON strings.
- Use [`_json_dumps`](../../src/langchain_replicate/_base.py:43) instead of [`json.dumps`](json/__init__.py) for tool/function payloads so output matches the backend JSON formatting assumptions.
- Normalize assistant tool-call arguments with [`_normalize_tool_arguments()`](../../src/langchain_replicate/chat_models.py:79) before parsing; malformed Python-dict strings and nested JSON are expected in real responses.
- Do not hardcode prompt or embedding input keys; [`Replicate._create_prediction_input()`](../../src/langchain_replicate/llms.py:101) and [`ReplicateEmbeddings._create_prediction_input()`](../../src/langchain_replicate/embeddings.py:58) derive them from the model OpenAPI schema.
- Deployment model IDs mutate [`ReplicateBase.model`](../../src/langchain_replicate/_base.py:78) to the released model name inside [`ReplicateBase._deployment`](../../src/langchain_replicate/_base.py:111); avoid logic that depends on the original `owner/name:deployment` string persisting.
- If a model schema lacks native stop support, [`Replicate._stream()`](../../src/langchain_replicate/llms.py:145) handles truncation locally and cancels the prediction; preserve that buffering logic when changing streaming behavior.
- Some embedding models require serialized or joined text input; use [`ReplicateEmbeddings.texts_value_mapping`](../../src/langchain_replicate/embeddings.py:37) instead of reshaping payloads ad hoc.
- Type checking is pyrefly strict mode; existing ignores like `missing-attribute`, `bad-argument-type`, and `unexpected-keyword` are there for Replicate SDK/pydantic interop, not dead code.
