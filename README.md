# LangChain Integration for Replicate

[![CI Build](https://github.com/ibm-granite-community/langchain-replicate/actions/workflows/build.yaml/badge.svg)](https://github.com/ibm-granite-community/langchain-replicate/actions/workflows/build.yaml)
<!--
[![PyPI version](https://badge.fury.io/py/langchain-replicate.svg)](https://badge.fury.io/py/langchain-replicate)
-->
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This package provides improved LangChain integrations for [Replicate](https://replicate.com/), enabling seamless use of Replicate's AI models within LangChain applications.

## Features

- 🤖 **Chat Models**: Full-featured chat model integration with streaming support
- 📝 **LLMs**: Completion model support for text generation
- 🔢 **Embeddings**: Vector embeddings for semantic search and RAG applications
- ⚡ **Async Support**: Native async/await support for all components
- 🔄 **Streaming**: Real-time streaming responses
- 🛠️ **Tool Calling**: Support for function/tool calling with compatible models

## Installation

```bash
uv pip install langchain-replicate
```

Or with pip:

```bash
pip install langchain-replicate
```

## Quick Start

### Prerequisites

1. Get your Replicate API token from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
2. Set the environment variable:

   ```bash
   export REPLICATE_API_TOKEN="your-token-here"
   ```

### Chat Completion

```python
from langchain_replicate import ChatReplicate
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize the chat model
chat = ChatReplicate(
    model="ibm-granite/granite-4.1-8b",
    model_kwargs={"temperature": 0.7, "max_tokens": 500},
)

# Single message
messages = [HumanMessage(content="What is the capital of France?")]
response = chat.invoke(messages)
print(response.text)

# With system message
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain quantum computing in simple terms."),
]
response = chat.invoke(messages)
print(response.text)
```

### Streaming Responses

```python
from langchain_replicate import ChatReplicate

chat = ChatReplicate(
    model="ibm-granite/granite-4.1-8b",
)

for chunk in chat.stream("Write a short poem about Python programming"):
    print(chunk.text, end="", flush=True)
```

### Completion

```python
from langchain_replicate import Replicate

llm = Replicate(
    model="ibm-granite/granite-4.1-8b",
    model_kwargs={"temperature": 0.5, "max_tokens": 200},
)

response = llm.invoke("Explain the theory of relativity.")
print(response)
```

### Embeddings

```python
from langchain_replicate import ReplicateEmbeddings

embeddings = ReplicateEmbeddings(
    model="ibm-granite/granite-embedding-small-english-r2"
)

# Embed a single document
doc_embedding = embeddings.embed_query("This is a test document")
print(f"Embedding dimension: {len(doc_embedding)}")

# Embed multiple documents
docs = [
    "First document",
    "Second document",
    "Third document",
]
doc_embeddings = embeddings.embed_documents(docs)
print(f"Number of embeddings: {len(doc_embeddings)}")
```

### Async Usage

```python
import asyncio
from langchain_replicate import ChatReplicate
from langchain_core.messages import HumanMessage

async def main():
    chat = ChatReplicate(model="ibm-granite/granite-4.1-8b")

    # Async invoke
    response = await chat.ainvoke([HumanMessage(content="Hello!")])
    print(response.text)

    # Async streaming
    async for chunk in chat.astream("Tell me a joke"):
        print(chunk.text, end="", flush=True)

asyncio.run(main())
```

## Advanced Usage

### Tool Calling

```python
from langchain_replicate import ChatReplicate
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny."

chat = ChatReplicate(
    model="ibm-granite/granite-4.1-8b",
)

# Bind tools to the model
chat_with_tools = chat.bind_tools([get_weather])

response = chat_with_tools.invoke("What's the weather in Paris?")
print(response.tool_calls)
```

### Custom Model Parameters

```python
from langchain_replicate import Replicate

llm = Replicate(
    model="stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    model_kwargs={
        "width": 512,
        "height": 512,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
    }
)
```

### RAG (Retrieval Augmented Generation) Example

```python
from langchain_replicate import ChatReplicate, ReplicateEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Create embeddings
embeddings = ReplicateEmbeddings(
    model="ibm-granite/granite-embedding-small-english-r2"
)

# Create vector store
docs = [
    Document(page_content="Paris is the capital of France."),
    Document(page_content="London is the capital of the UK."),
    Document(page_content="Berlin is the capital of Germany."),
]
vectorstore = FAISS.from_documents(docs, embeddings)

# Query
query = "What is the capital of France?"
relevant_docs = vectorstore.similarity_search(query, k=1)

# Generate answer
chat = ChatReplicate(model="ibm-granite/granite-4.1-8b")
context = relevant_docs[0].page_content
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
response = chat.invoke(prompt)
print(response.text)
```

## Supported Models

This integration works with many Replicate models. Popular choices include:

### Chat Models

- `ibm-granite/granite-4.1-8b`
- `ibm-granite/granite-vision-4.1-4b`
- `ibm-granite/granite-speech-4.1-2b`

### Embedding Models

- `ibm-granite/granite-embedding-278m-multilingual`
- `ibm-granite/granite-embedding-small-english-r2`

Find more models at [replicate.com/explore](https://replicate.com/explore)

## Configuration

### Environment Variables

- `REPLICATE_API_TOKEN`: Your Replicate API token (required)

### Model Parameters

Common parameters supported by most models:

- `temperature`: Controls randomness (0.0 to 1.0)
- `max_tokens`: Maximum tokens to generate
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `repetition_penalty`: Penalty for repetition

Refer to specific model documentation on Replicate for available parameters.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/ibm-granite-community/langchain-replicate.git
cd langchain-replicate

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest tests

# Run specific test file
uv run pytest tests/test_chat_models.py
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check --fix

# Type checking
uv run pyrefly check
```

## Troubleshooting

### Common Issues

#### API Token Not Found

```text
Error: REPLICATE_API_TOKEN environment variable not set
```

Solution: Set your API token as an environment variable or pass it directly:

```python
chat = ChatReplicate(
    model="ibm-granite/granite-4.1-8b",
    replicate_api_token="your-token-here"
)
```

#### Model Not Found

```text
Error: Model not found
```

Solution: Verify the model name and version on [replicate.com](https://replicate.com/)

#### Rate Limiting

If you encounter rate limits, consider:

- Upgrading your Replicate plan
- Implementing retry logic with exponential backoff
- Reducing request frequency

## Contributing

Please see our [Contributing Guide][CG] and [Code of Conduct][CoC] for details.

### Requirements

- All commits must be [DCO signed-off][CG-legal]
- All commits must be [GPG or SSH signed][CG-signing]
- Follow the code style (enforced by Ruff)
- Add tests for new features
- Update documentation as needed

## License

This project is licensed under the [MIT License](LICENSE), in alignment with [langchain-community](https://github.com/langchain-ai/langchain-community).

## Support

- 📖 [Documentation](https://github.com/ibm-granite-community/langchain-replicate#readme)
- 🐛 [Issue Tracker](https://github.com/ibm-granite-community/langchain-replicate/issues)
- 💬 [Discussions](https://github.com/ibm-granite-community/langchain-replicate/discussions)

## IBM Public Repository Disclosure

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

## Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for developing LLM applications
- [Replicate](https://replicate.com/) - Run and fine-tune open-source models
- [IBM Granite](https://huggingface.co/ibm-granite) - IBM's family of open-source AI models

[CoC]: https://github.com/ibm-granite-community/.github/blob/main/CODE_OF_CONDUCT.md
[CG]: https://github.com/ibm-granite-community/.github/blob/main/CONTRIBUTING.md
[CG-legal]: https://github.com/ibm-granite-community/.github/blob/main/CONTRIBUTING.md#legal
[CG-signing]: https://github.com/ibm-granite-community/.github/blob/main/CONTRIBUTING.md#signing-commits
