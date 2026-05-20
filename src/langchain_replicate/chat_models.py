# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import contextlib
import json
import logging
import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, cast

from json_repair import repair_json
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    is_basemodel_subclass,
    is_pydantic_v1_subclass,
)
from pydantic import BaseModel, ConfigDict
from replicate.exceptions import ModelError
from replicate.prediction import Prediction
from typing_extensions import override

from langchain_replicate._base import ReplicateBase, _adjust_prediction_input, _json_dumps

logger = logging.getLogger(__name__)


def _normalize_tool_arguments(args_str: str) -> str:
    r"""Ensure arguments is always a proper JSON string with robust error handling.

    Handles various malformed inputs from LLMs including:
    - Valid JSON strings
    - Python dict strings (e.g., "{'key': 'value'}")
    - Multiple layers of JSON wrapping (e.g., '"{\\"key\\": \\"value\\"}"')
    - Malformed outputs with trailing characters (e.g., '"{}""}')
    - Unbalanced braces
    - Empty or invalid inputs

    Args:
        args_str: Tool call arguments string (potentially malformed from LLM)

    Returns:
        Valid JSON string. Returns "{}" as fallback for unparseable input.

    Examples:
        >>> normalize_tool_arguments('{"key": "value"}')
        '{"key": "value"}'
        >>> normalize_tool_arguments("{'key': 'value'}")
        '{"key": "value"}'
        >>> normalize_tool_arguments('"{}""}')
        '{}'
        >>> normalize_tool_arguments("invalid")
        '{}'
    """
    if not args_str or not args_str.strip():
        return "{}"

    current = args_str.strip()
    parsed_object: Any = None

    # Strategy 1: Try standard JSON parsing first
    try:
        parsed_object = json.loads(current)
    except (json.JSONDecodeError, ValueError):
        # Strategy 2: Try Python literal evaluation (handles single quotes, etc.)
        try:
            parsed_object = ast.literal_eval(current)
        except (ValueError, SyntaxError):
            # Strategy 3: Use repair_json as last resort
            try:
                repaired = repair_json(current)
                parsed_object = json.loads(repaired)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("Could not parse tool arguments: %s. Error: %s", current[:100], e)
                return "{}"

    # Handle special cases after successful parsing
    # 1. If result is null, return empty dict
    if parsed_object is None:
        return "{}"

    # 2. Recursively unwrap nested JSON strings (e.g., '"{\"key\": \"value\"}"')
    # Keep unwrapping until we get a non-string or unwrapping fails
    max_unwrap_depth = 5  # Prevent infinite loops
    unwrap_count = 0
    while isinstance(parsed_object, str) and unwrap_count < max_unwrap_depth:
        try:
            unwrapped = json.loads(parsed_object)
            # Prevent infinite loop if json.loads returns the same string
            if unwrapped == parsed_object:
                return "{}"
            parsed_object = unwrapped
            unwrap_count += 1
        except (json.JSONDecodeError, TypeError):
            # If unwrapping fails, tool arguments should be objects not strings
            return "{}"

    # 3. If result is an array with single object, extract it
    if isinstance(parsed_object, list) and len(parsed_object) == 1:
        parsed_object = parsed_object[0]

    # 4. Clean up known problematic patterns for dicts
    # Remove empty keys with empty values
    if isinstance(parsed_object, dict) and "" in parsed_object and not parsed_object[""]:
        parsed_object = {k: v for k, v in parsed_object.items() if k != ""}
        if not parsed_object:
            return "{}"

    return _json_dumps(parsed_object)


def _convert_dict_to_message(_dict: Mapping[str, Any], call_id: str) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.
        call_id: call id

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = call_id
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    if role == "assistant":
        content = _dict.get("content", "") or ""
        additional_kwargs: dict[str, Any] = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            for raw_tool_call in raw_tool_calls:
                ## Code change to support langgraph with A2A and graph.astream.
                if "function" in raw_tool_call:
                    func = raw_tool_call.get("function", {})
                    if "arguments" in func:
                        raw_args = raw_tool_call["function"]["arguments"]
                        json_args_str = _normalize_tool_arguments(raw_args)
                        raw_tool_call["function"]["arguments"] = json_args_str

                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:  # pylint: disable=broad-exception-caught
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e)),
                    )
        if audio := _dict.get("audio"):
            additional_kwargs["audio"] = audio
        if reasoning_content := _dict.get("reasoning_content"):
            additional_kwargs["reasoning_content"] = reasoning_content
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
        )
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""), name=name, id=id_)
    if role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""),
            name=cast("str", _dict.get("name")),
            id=id_,
        )
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast("str", _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[arg-type]


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        formatted_content = []
        for block in content:
            # Remove unexpected block types
            if isinstance(block, dict) and "type" in block and block["type"] in {"tool_use", "thinking", "reasoning_content"}:
                continue

            # Image blocks
            if isinstance(block, dict) and block.get("type") == "image":
                if (data := block.get("base64")) and (mime_type := block.get("mime_type")):
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{data}"},
                        }
                    )
                else:
                    continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": _json_dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict[str, Any]:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: dict[str, Any] = {"content": _format_message_content(message.content)}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [_lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc) for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [{k: v for k, v in tool_call.items() if k in tool_call_supported_props} for tool_call in message_dict["tool_calls"]]
        elif "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        audio: dict[str, Any] | None = None
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "audio" and (id_ := block.get("id")):
                audio = {"id": id_}
        if not audio and "audio" in message.additional_kwargs:
            raw_audio = message.additional_kwargs["audio"]
            audio = {"id": message.additional_kwargs["audio"]["id"]} if "id" in raw_audio else raw_audio
        if audio:
            message_dict["audio"] = audio
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        # Fix for https://github.com/langchain-ai/langchain-ibm/issues/162
        content = message_dict.get("content")
        if content is not None and not isinstance(content, str):
            message_dict["content"] = message.text

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        error_msg = f"Got unknown type {message}"
        raise TypeError(error_msg)
    return message_dict


def _create_usage_metadata(
    oai_token_usage: dict[str, Any],
    *,
    _prompt_tokens_included: bool,
) -> UsageMetadata:
    input_tokens: int = oai_token_usage.get("prompt_tokens", 0) if not _prompt_tokens_included else 0
    output_tokens: int = oai_token_usage.get("completion_tokens", 0)
    total_tokens: int = oai_token_usage.get("total_tokens", input_tokens + output_tokens)
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any],
    default_class: type[BaseMessageChunk],
    call_id: str,
    *,
    is_first_tool_chunk: bool,
) -> BaseMessageChunk:
    id_ = call_id
    role = cast("str", _dict.get("role"))
    content = cast("str", _dict.get("content") or "")
    additional_kwargs: dict[str, Any] = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks: list[ToolCallChunk] = []
    if raw_tool_calls := _dict.get("tool_calls"):
        with contextlib.suppress(KeyError):
            tool_call_chunks = [
                tool_call_chunk(
                    name=rtc["function"].get("name") if is_first_tool_chunk or (rtc.get("id") is not None) else None,
                    args=rtc["function"].get("arguments"),
                    # `id` is provided only for the first delta with unique tool_calls
                    # (multiple tool calls scenario)
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]

    if reasoning_content := _dict.get("reasoning_content"):
        additional_kwargs["reasoning_content"] = reasoning_content

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content,
            tool_call_id=_dict["tool_call_id"],
            id=id_,
        )
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore[call-arg]


def _convert_chunk_to_generation_chunk(
    chunk: dict[str, Any],
    default_chunk_class: type,
    *,
    is_first_tool_chunk: bool,
    _prompt_tokens_included: bool,
) -> ChatGenerationChunk | None:
    token_usage = chunk.get("usage")
    choices = chunk.get("choices", [])

    usage_metadata: UsageMetadata | None = _create_usage_metadata(token_usage, _prompt_tokens_included=_prompt_tokens_included) if token_usage else None

    if len(choices) == 0:
        # logprobs is implicitly None
        return ChatGenerationChunk(
            message=default_chunk_class(content="", usage_metadata=usage_metadata),
        )

    choice = choices[0]
    if choice["delta"] is None:
        return None

    message_chunk = _convert_delta_to_message_chunk(
        choice["delta"],
        default_chunk_class,
        chunk["id"],
        is_first_tool_chunk=is_first_tool_chunk,
    )
    generation_info = {}

    if finish_reason := choice.get("finish_reason"):
        generation_info["finish_reason"] = finish_reason
        if model_name := chunk.get("model"):
            generation_info["model_name"] = model_name
        if system_fingerprint := chunk.get("system_fingerprint"):
            generation_info["system_fingerprint"] = system_fingerprint

    logprobs = choice.get("logprobs")
    if logprobs:
        generation_info["logprobs"] = logprobs

    if usage_metadata and isinstance(message_chunk, AIMessageChunk):
        message_chunk.usage_metadata = usage_metadata

    return ChatGenerationChunk(
        message=message_chunk,
        generation_info=generation_info or None,
    )


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _convert_to_openai_response_format(schema: dict[str, Any] | type, *, strict: bool | None = None) -> dict[str, Any] | TypeBaseModel:
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return schema

    if isinstance(schema, dict) and "json_schema" in schema and schema.get("type") == "json_schema":
        response_format = schema
    elif isinstance(schema, dict) and "name" in schema and "schema" in schema:
        response_format = {"type": "json_schema", "json_schema": schema}
    else:
        if strict is None:
            if isinstance(schema, dict) and isinstance(schema.get("strict"), bool):  # noqa: SIM108
                strict = schema["strict"]
            else:
                strict = False
        function = convert_to_openai_function(schema, strict=strict)
        function["schema"] = function.pop("parameters")
        response_format = {"type": "json_schema", "json_schema": function}

    # pyrefly: ignore [missing-attribute]
    if strict is not None and strict is not response_format["json_schema"].get("strict") and isinstance(schema, dict):
        msg = (
            f"Output schema already has 'strict' value set to "
            f"{schema['json_schema']['strict']} but 'strict' also passed in to "
            f"with_structured_output as {strict}. Please make sure that "
            f"'strict' is only specified in one place."
        )
        raise ValueError(msg)
    return response_format


class ChatReplicate(ReplicateBase, BaseChatModel):
    """Replicate chat completion models.

    To use, you should have the ``replicate`` python package installed,
    and the environment variable ``REPLICATE_API_TOKEN`` set with your API token.
    You can find your token here: https://replicate.com/account

    The model param is required, but any other model parameters can also
    be passed in with the format model_kwargs={model_param: value, ...}

    Example:
        .. code-block:: python

            from langchain_replicate import ChatReplicate

            replicate = ChatReplicate(
                model=(
                    "stability-ai/stable-diffusion:"
                    "27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
                ),
                model_kwargs={"image_dimensions": "512x512"}
            )
    """

    streaming: bool = False
    """Whether to stream the results."""

    model_config = ConfigDict()

    @override
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @override
    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "replicate-chat"

    def _create_message_dicts(
        self,
        messages: list[BaseMessage],
    ) -> list[dict[str, Any]]:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _create_chat_result(self, prediction: Prediction, generation_info: dict[str, Any] | None = None) -> ChatResult:
        if prediction.status == "failed":
            raise ModelError(prediction)

        if isinstance(prediction.output, list):
            assert len(prediction.output) == 1, "Expected exactly one output from generation request."
            output = prediction.output[0]
        else:
            output = prediction.output
        response: Mapping[str, Any]
        if isinstance(output, str):
            response = json.loads(output)
        elif isinstance(output, Mapping):
            response = output
        else:
            logger.error("Unrecognized output shape %s %r", type(output), output)
            raise ValueError(f"Unrecognized output shape {type(output)} {output!r}")

        token_usage = response.get("usage", {})

        generations = []
        for choice in response["choices"]:
            message = _convert_dict_to_message(choice["message"], response["id"])

            if token_usage and isinstance(message, AIMessage):
                message.usage_metadata = _create_usage_metadata(token_usage, _prompt_tokens_included=False)
            generation_info = generation_info or {}
            generation_info["finish_reason"] = choice.get("finish_reason") if choice.get("finish_reason") is not None else generation_info.get("finish_reason")
            if "logprobs" in choice:
                generation_info["logprobs"] = choice["logprobs"]
            generation = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(generation)
        llm_output = {
            "token_usage": token_usage,
            "model_name": response.get("model", self.model),
            "system_fingerprint": response.get("system_fingerprint", ""),
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_prediction_input(self, messages: list[BaseMessage], stream: bool, stop: list[str] | None, **kwargs: Any) -> dict[str, Any]:
        message_dicts = self._create_message_dicts(messages)

        input_: dict[str, Any] = self.model_kwargs | kwargs | {"messages": message_dicts} | self._stop_input(stop) | self._stream_input(stream)

        return _adjust_prediction_input(input_)

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        input_ = self._create_prediction_input(messages=messages, stream=True, stop=stop, **kwargs)
        prediction = self._create_prediction(input_)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        is_first_tool_chunk = True
        _prompt_tokens_included = False

        # Only use accumulation for models that need it
        use_accumulation = self._needs_streaming_accumulation()
        accumulated_chunks: list[ChatGenerationChunk] = []

        for output in prediction.output_iterator():
            chunk: dict[str, Any]
            if isinstance(output, str):
                chunk = json.loads(output)
            elif isinstance(output, dict):
                chunk = output
            else:
                logger.error("Unrecognized output shape %s %r", type(output), output)
                raise ValueError(f"Unrecognized output shape {type(output)} {output!r}")

            generation_chunk = _convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, is_first_tool_chunk=is_first_tool_chunk, _prompt_tokens_included=_prompt_tokens_included
            )
            if generation_chunk is None:
                continue

            default_chunk_class = type(generation_chunk.message)
            if (
                hasattr(generation_chunk.message, "usage_metadata") and generation_chunk.message.usage_metadata  # pyright: ignore[reportAttributeAccessIssue]
            ):
                _prompt_tokens_included = True
            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            if hasattr(generation_chunk.message, "tool_calls") and isinstance(
                generation_chunk.message.tool_calls,  # pyright: ignore[reportAttributeAccessIssue]
                list,
            ):
                first_tool_call = (
                    generation_chunk.message.tool_calls[0]  # pyright: ignore[reportAttributeAccessIssue]
                    if generation_chunk.message.tool_calls  # pyright: ignore[reportAttributeAccessIssue]
                    else None
                )
                if isinstance(first_tool_call, dict) and first_tool_call.get("name"):
                    is_first_tool_chunk = False

            # Process chunk - use accumulation only for specific models
            if use_accumulation:
                chunks_to_yield = self._process_streaming_chunk(generation_chunk, accumulated_chunks)
                yield from chunks_to_yield
            else:
                yield generation_chunk

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)

        input_ = self._create_prediction_input(messages=messages, stream=False, stop=stop, **kwargs)
        prediction = self._create_prediction(input_)

        prediction.wait()

        return self._create_chat_result(prediction)

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        input_ = self._create_prediction_input(messages=messages, stream=True, stop=stop, **kwargs)
        prediction = await self._async_create_prediction(input_)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        is_first_tool_chunk = True
        _prompt_tokens_included = False

        # Only use accumulation for models that need it
        use_accumulation = self._needs_streaming_accumulation()
        accumulated_chunks: list[ChatGenerationChunk] = []

        async for output in prediction.async_output_iterator():
            chunk: dict[str, Any]
            if isinstance(output, str):
                chunk = json.loads(output)
            elif isinstance(output, dict):
                chunk = output
            else:
                logger.error("Unrecognized output shape %s %r", type(output), output)
                raise ValueError(f"Unrecognized output shape {type(output)} {output!r}")

            generation_chunk = _convert_chunk_to_generation_chunk(
                chunk,
                default_chunk_class,
                is_first_tool_chunk=is_first_tool_chunk,
                _prompt_tokens_included=_prompt_tokens_included,
            )
            if generation_chunk is None:
                continue

            default_chunk_class = type(generation_chunk.message)
            if (
                hasattr(generation_chunk.message, "usage_metadata") and generation_chunk.message.usage_metadata  # pyright: ignore[reportAttributeAccessIssue]
            ):
                _prompt_tokens_included = True
            logprobs = (generation_chunk.generation_info or {}).get("logprobs")
            if run_manager:
                await run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=logprobs,
                )
            if hasattr(generation_chunk.message, "tool_calls") and isinstance(
                generation_chunk.message.tool_calls,  # pyright: ignore[reportAttributeAccessIssue]
                list,
            ):
                first_tool_call = (
                    generation_chunk.message.tool_calls[0]  # pyright: ignore[reportAttributeAccessIssue]
                    if generation_chunk.message.tool_calls  # pyright: ignore[reportAttributeAccessIssue]
                    else None
                )
                if isinstance(first_tool_call, dict) and first_tool_call.get("name"):
                    is_first_tool_chunk = False

            # Process chunk - use accumulation only for specific models
            if use_accumulation:
                chunks_to_yield = self._process_streaming_chunk(generation_chunk, accumulated_chunks)
                for chunk_to_yield in chunks_to_yield:
                    yield chunk_to_yield
            else:
                yield generation_chunk

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return await agenerate_from_stream(stream_iter)

        input_ = self._create_prediction_input(messages=messages, stream=False, stop=stop, **kwargs)
        prediction = await self._async_create_prediction(input_)

        await prediction.async_wait()

        return self._create_chat_result(prediction)

    def _needs_streaming_accumulation(self) -> bool:
        """Check if the model needs streaming chunk accumulation for tool calls.

        Some models (like ibm-granite/granite-4.0-h-small with vLLM) return malformed
        tool arguments in streaming mode that need to be accumulated and normalized.

        Returns:
            True if the model needs accumulation, False otherwise
        """
        if not self.model:
            return False

        # List of model patterns that need streaming accumulation
        models_needing_accumulation = [
            "ibm-granite/granite-4.0-h-small",
            # Add other models here as needed
        ]

        return any(pattern in self.model.lower() for pattern in models_needing_accumulation)

    def _process_streaming_chunk(
        self,
        generation_chunk: ChatGenerationChunk,
        accumulated_chunks: list[ChatGenerationChunk],
    ) -> list[ChatGenerationChunk]:
        """Process a streaming chunk and handle tool call accumulation.

        Args:
            generation_chunk: The current generation chunk
            accumulated_chunks: List of accumulated chunks with tool calls

        Returns:
            List of chunks to yield (may be empty, or contain 1-2 chunks)
        """
        # Check if this chunk has tool calls
        has_tool_calls = (
            hasattr(generation_chunk.message, "tool_call_chunks") and generation_chunk.message.tool_call_chunks  # pyright: ignore[reportAttributeAccessIssue]
        )

        # Check if this is a final chunk or a transition
        finish_reason = (generation_chunk.generation_info or {}).get("finish_reason")

        # If current chunk has tool calls, add it to accumulation first
        if has_tool_calls:
            accumulated_chunks.append(generation_chunk)

        # Determine if we should finalize accumulated chunks
        should_finalize = accumulated_chunks and (not has_tool_calls or finish_reason)

        if should_finalize:
            # Accumulate all chunks
            accumulated_message = accumulated_chunks[0].message
            for gen_chunk in accumulated_chunks[1:]:
                accumulated_message = accumulated_message + gen_chunk.message

            # Normalize tool arguments
            if hasattr(accumulated_message, "tool_call_chunks"):
                for tc_chunk in accumulated_message.tool_call_chunks:  # pyright: ignore[reportAttributeAccessIssue]
                    if tc_chunk.get("args"):
                        with contextlib.suppress(Exception):
                            tc_chunk["args"] = _normalize_tool_arguments(tc_chunk["args"])

            # Clear accumulated chunks
            accumulated_chunks.clear()

            if not has_tool_calls:
                # Transition: yield tool message + current chunk
                tool_message_chunk = ChatGenerationChunk(
                    message=accumulated_message,
                    generation_info=None,
                )
                return [tool_message_chunk, generation_chunk]

            # Final chunk with finish_reason
            # Update current chunk with accumulated message
            generation_chunk = ChatGenerationChunk(
                message=accumulated_message,
                generation_info=generation_chunk.generation_info,
            )
            return [generation_chunk]

        # If we added to accumulated_chunks but not finalizing, don't yield yet
        if has_tool_calls:
            return []

        # Regular chunk without tool calls and no accumulated chunks - yield as is
        return [generation_chunk]

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: dict[str, Any] | str | Literal["auto", "none", "required", "any"] | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call. Options are:

                - `str` of the form `'<<tool_name>>'`: calls `<<tool_name>>` tool.
                - `'auto'`: automatically selects a tool (including no tool).
                - `'none'`: does not call a tool.
                - `'any'` or `'required'` or `True`: force at least one tool to be
                  called.
                - `dict` of the form
                  `{"type": "function", "function": {"name": <<tool_name>>}}`:
                  calls `<<tool_name>>` tool.
                - `False` or `None`: no effect, default OpenAI behavior.

            strict: If `True`, model output is guaranteed to exactly match the JSON
                Schema provided in the tool definition.
                The input schema will also be validated according to the supported
                schemas.
                If `False`, input schema will not be validated and model output will not
                be validated.
                If `None`, `strict` argument will not be passed to the model.

            kwargs: Any additional parameters are passed directly to `bind`.
        """

        formatted_tools = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice in ("auto", "none", "required"):
                    pass
                elif tool_choice == "any":
                    tool_choice = "required"
                else:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [formatted_tool["function"]["name"] for formatted_tool in formatted_tools]
                if not any(tool_name == tool_choice["function"]["name"] for tool_name in tool_names):
                    error_msg = (f"Tool choice {tool_choice} was specified, but the only provided tools were {tool_names}.",)
                    raise ValueError(error_msg)
            else:
                error_msg = (  # type: ignore[unreachable]
                    f"Unrecognized tool_choice type. Expected str, bool or dict. Received: {tool_choice}",
                )
                raise ValueError(error_msg)

            kwargs["tool_choice"] = tool_choice
        else:
            kwargs["tool_choice"] = "auto"

        return super().bind(tools=formatted_tools, **kwargs)

    @override
    def with_structured_output(
        self,
        schema: dict[str, Any] | type | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        r"""Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class,

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

            method: The method for steering model generation, one of:

                - `'function_calling'`: uses tool-calling features.
                - `'json_schema'`: uses dedicated structured output features.
                - `'json_mode'`: uses JSON mode.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys `'raw'`, `'parsed'`, and `'parsing_error'`.

            strict:

                - `True`:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the
                    supported schemas.
                - `False`:
                    Input schema will not be validated and model output will not be
                    validated.
                - `None`:
                    `strict` argument will not be passed to the model.

            kwargs: Additional keyword args


        Returns:
            A Runnable that takes same inputs as a `langchain_core.language_models.chat.BaseChatModel`.

            If `include_raw` is True, then Runnable outputs a dict with keys:

            - `'raw'`: BaseMessage
            - `'parsed'`: None if there was a parsing error, otherwise the type depends on the `schema` as described above.
            - `'parsing_error'`: Optional[BaseException]

        ??? note "Example: `schema=Pydantic` class, `method='function_calling'`, `include_raw=True`"

            ```python
            from langchain_replicate import ChatReplicate
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatReplicate(...)
            structured_model = model.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_Ao02pnFYXD6GN1yzc0uXPsvF",
                                "function": {
                                    "arguments": '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}',
                                    "name": "AnswerWithJustification",
                                },
                                "type": "function",
                            }
                        ]
                    },
                ),
                "parsed": AnswerWithJustification(
                    answer="They weigh the same.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=JSON` schema, `method='function_calling'`, `include_raw=False`"

            ```python
            from langchain_replicate import ChatReplicate
            from pydantic import BaseModel
            from langchain_core.utils.function_calling import convert_to_openai_tool


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            dict_schema = convert_to_openai_tool(AnswerWithJustification)
            model = ChatReplicate(...)
            structured_model = model.with_structured_output(dict_schema)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_schema'`, `include_raw=True`"

            ```python
            from langchain_replicate import ChatReplicate
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatReplicate(...)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_schema", include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "chatcmpl-tool-bfbd6f6dd33b438990c5ddf277485971",
                                "type": "function",
                                "function": {
                                    "name": "AnswerWithJustification",
                                    "arguments": '{"answer": "They weigh the same", "justification": "A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound."}',
                                },
                            }
                        ]
                    },
                    response_metadata={
                        "token_usage": {
                            "completion_tokens": 45,
                            "prompt_tokens": 275,
                            "total_tokens": 320,
                        },
                        "model_name": "meta-llama/llama-3-3-70b-instruct",
                        "system_fingerprint": "",
                        "finish_reason": "stop",
                    },
                    id="chatcmpl-461ca5bd-1982-412c-b886-017c483bf481---8c18b06eead65ae4691364798787bda7---71896588-efa5-439f-a25f-d1abfe289f5a",
                    tool_calls=[
                        {
                            "name": "AnswerWithJustification",
                            "args": {
                                "answer": "They weigh the same",
                                "justification": "A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound.",
                            },
                            "id": "chatcmpl-tool-bfbd6f6dd33b438990c5ddf277485971",
                            "type": "tool_call",
                        }
                    ],
                    usage_metadata={
                        "input_tokens": 275,
                        "output_tokens": 45,
                        "total_tokens": 320,
                    },
                ),
                "parsed": AnswerWithJustification(
                    answer="They weigh the same",
                    justification="A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=function` schema, `method='json_schema'`, `include_raw=False`"

            ```python
            from langchain_replicate import ChatReplicate
            from pydantic import BaseModel

            function__schema = {
                "name": "AnswerWithJustification",
                "description": "An answer to the user question along with justification for the answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "justification": {
                            "description": "A justification for the answer.",
                            "type": "string",
                        },
                    },
                    "required": ["answer"],
                },
            }

            model = ChatReplicate(...)
            structured_model = model.with_structured_output(
                function_schema, method="json_schema", include_raw=False
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            ```

            ```python
            {
                "answer": "They weigh the same",
                "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.",
            }
            ```

        ??? note "Example: `schema=Pydantic` class, `method='json_mode'`, `include_raw=True`"

            ```python
            from langchain_replicate import ChatReplicate
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                answer: str
                justification: str


            model = ChatReplicate(...)
            structured_model = model.with_structured_output(
                AnswerWithJustification, method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'
                ),
                "parsed": AnswerWithJustification(
                    answer="They are both the same weight.",
                    justification="Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                ),
                "parsing_error": None,
            }
            ```

        ??? note "Example: `schema=None`, `method='json_mode'`, `include_raw=True`"

            ```python
            from langchain_replicate import ChatReplicate

            model = ChatReplicate(...)
            structured_model = model.with_structured_output(
                method="json_mode", include_raw=True
            )

            structured_model.invoke(
                "Answer the following question. "
                "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                "What's heavier a pound of bricks or a pound of feathers?"
            )
            ```

            ```python
            {
                "raw": AIMessage(
                    content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'
                ),
                "parsed": {
                    "answer": "They are both the same weight.",
                    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.",
                },
                "parsing_error": None,
            }
            ```
        """  # noqa: E501 # pylint: disable=line-too-long
        if strict is not None and method == "json_mode":
            msg = "Argument `strict` is not supported with `method`='json_mode'"
            raise ValueError(msg)
        is_pydantic_schema = _is_pydantic_class(schema)

        if (
            method == "json_schema" and is_pydantic_schema and is_pydantic_v1_subclass(schema)  # type: ignore[arg-type]
        ):
            # Check for Pydantic BaseModel V1
            warnings.warn(
                "Received a Pydantic BaseModel V1 schema. This is not supported by "
                'method="json_schema". Please use method="function_calling" '
                "or specify schema via JSON Schema or Pydantic V2 BaseModel. "
                'Overriding to method="function_calling".',
                stacklevel=2,
            )
            method = "function_calling"

        if method == "function_calling":
            if schema is None:
                error_msg = "schema must be specified when method is not 'json_mode'. Received None."
                raise ValueError(error_msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            model = self.bind_tools(
                [schema],
                strict=strict,
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": method, "strict": strict},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: Runnable[Any, Any] = PydanticToolsParser(
                    tools=[schema],  # type: ignore
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name,
                    first_tool_only=True,
                )
        elif method == "json_mode":
            model = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                error_msg = "schema must be specified when method is not 'json_mode'. Received None."
                raise ValueError(error_msg)
            if is_pydantic_schema:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,  # type: ignore[union-attr]
                        "description": schema.__doc__,
                        "schema": schema.model_json_schema(),  # type: ignore[union-attr]
                    },
                }
            else:
                response_format = _convert_to_openai_response_format(  # type: ignore[assignment]
                    schema, strict=strict
                )
            bind_kwargs = {
                "response_format": response_format,
                "ls_structured_output_format": {
                    "kwargs": {"method": method, "strict": strict},
                    "schema": convert_to_openai_tool(schema),
                },
            }
            model = self.bind(**bind_kwargs)
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            error_msg = f"Unrecognized method argument. Expected one of 'function_calling' or 'json_mode'. Received: '{method}'"
            raise ValueError(error_msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none],
                exception_key="parsing_error",
            )
            return RunnableMap(raw=model) | parser_with_fallback
        return model | output_parser
