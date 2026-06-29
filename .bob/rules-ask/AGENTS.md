# Project Documentation Rules (Non-Obvious Only)

- The tests are not pure unit tests: most cases hit live Replicate models.
- Deployment model strings like `owner/name:deployment` are resolved through [`ReplicateBase._deployment`](../../src/langchain_replicate/_base.py:111), which mutates [`ReplicateBase.model`](../../src/langchain_replicate/_base.py:78) to the released model name; docs should not imply the original identifier is preserved.
- Prompt and embedding input field names are discovered from the model OpenAPI schema in [`Replicate._create_prediction_input()`](../../src/langchain_replicate/llms.py:101) and [`ReplicateEmbeddings._create_prediction_input()`](../../src/langchain_replicate/embeddings.py:58), so examples should avoid hardcoded keys like `prompt` or `text`.
- Tool-calling compatibility depends on two repair layers: [`_adjust_prediction_input()`](../../src/langchain_replicate/_base.py:52) rewrites outbound `tool_choice`, and [`_normalize_tool_arguments()`](../../src/langchain_replicate/chat_models.py:79) repairs malformed inbound arguments before LangChain parsing.
- Stop-sequence behavior is backend-schema dependent; when the model lacks native stop fields, [`Replicate._stream()`](../../src/langchain_replicate/llms.py:145) truncates locally and cancels the prediction, so streaming docs should not promise backend-enforced stops.
- Embedding examples need model-specific input shaping: some models require [`json.dumps`](json/__init__.py) passed into [`ReplicateEmbeddings.texts_value_mapping`](../../src/langchain_replicate/embeddings.py:37), while others accept raw `list[str]`.
- Lint commands: `uv run ruff check`, `uv run ruff format`; type check: `uv run pyrefly check`.
