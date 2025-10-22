# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, cast

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
)
from pydantic import BaseModel, ConfigDict
from replicate.exceptions import ModelError
from replicate.prediction import Prediction
from typing_extensions import override

from langchain_replicate._base import ReplicateBase

logger = logging.getLogger(__name__)


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
        additional_kwargs: dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:  # pylint: disable=broad-exception-caught
                    invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))
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
        return FunctionMessage(content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_)
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)  # type: ignore[arg-type]


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        # Remove unexpected block types
        formatted_content = []
        for block in content:
            if isinstance(block, dict) and "type" in block and block["type"] == "tool_use":
                continue
            formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _base62_encode(num: int) -> str:
    """Encodes a number in base62 and ensures result is of a specified length."""
    base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return base62[0]
    arr = []
    base = len(base62)
    while num:
        num, rem = divmod(num, base)
        arr.append(base62[rem])
    arr.reverse()
    return "".join(arr)


def _convert_tool_call_id_to_mistral_compatible(tool_call_id: str) -> str:
    """Convert a tool call ID to a Mistral-compatible format"""
    hash_bytes = hashlib.sha256(tool_call_id.encode()).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder="big")
    base62_str = _base62_encode(hash_int)
    if len(base62_str) >= 9:
        return base62_str[:9]
    return base62_str.rjust(9, "0")


def _convert_message_to_dict(message: BaseMessage, model_id: str | None) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.
        model_id: Type of model to use.

    Returns:
        The dictionary.
    """
    # pylint: disable=too-many-branches
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
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [_lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc) for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [{k: v for k, v in tool_call.items() if k in tool_call_supported_props} for tool_call in message_dict["tool_calls"]]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None

        # Workaround for "mistralai/mistral-large" model when id < 9
        if model_id and model_id.startswith("mistralai"):
            tool_calls = message_dict.get("tool_calls", [])
            if isinstance(tool_calls, list) and tool_calls and isinstance(tool_calls[0], dict):
                tool_call_id = tool_calls[0].get("id", "")
                if len(tool_call_id) < 9:
                    tool_call_id = _convert_tool_call_id_to_mistral_compatible(tool_call_id)

                message_dict["tool_calls"][0]["id"] = tool_call_id
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        # Workaround for "mistralai/mistral-large" model when tool_call_id < 9
        if model_id and model_id.startswith("mistralai"):
            tool_call_id = message_dict.get("tool_call_id", "")
            if len(tool_call_id) < 9:
                tool_call_id = _convert_tool_call_id_to_mistral_compatible(tool_call_id)

            message_dict["tool_call_id"] = tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _create_usage_metadata(
    oai_token_usage: Mapping[str, Any],
    _prompt_tokens_included: bool,
) -> UsageMetadata:
    input_tokens = oai_token_usage.get("prompt_tokens", 0) if not _prompt_tokens_included else 0
    output_tokens = oai_token_usage.get("completion_tokens", 0)
    total_tokens = oai_token_usage.get("total_tokens", input_tokens + output_tokens)
    return UsageMetadata(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _convert_delta_to_message_chunk(
    delta: Mapping[str, Any],
    default_class: type[BaseMessageChunk],
    call_id: str,
    is_first_tool_chunk: bool,
) -> BaseMessageChunk:
    # pylint: disable=too-many-return-statements
    id_ = call_id
    role = cast(str, delta.get("role"))
    content = cast(str, delta.get("content") or "")
    additional_kwargs: dict = {}
    if delta.get("function_call"):
        function_call = dict(delta["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := delta.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
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

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        )
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    if role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=delta["name"], id=id_)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=delta["tool_call_id"], id=id_)
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    return default_class(content=content, id=id_)  # type: ignore


def _convert_chunk_to_generation_chunk(
    chunk: Mapping[str, Any],
    default_chunk_class: type[BaseMessageChunk],
    is_first_tool_chunk: bool,
    _prompt_tokens_included: bool,
) -> ChatGenerationChunk | None:
    token_usage = chunk.get("usage")
    choices = chunk.get("choices", [])

    if len(choices) == 0:
        # logprobs is implicitly None
        message_chunk = default_chunk_class(content="")  # type: ignore
        if token_usage and isinstance(message_chunk, AIMessageChunk):
            message_chunk.usage_metadata = _create_usage_metadata(token_usage, _prompt_tokens_included)
        generation_chunk = ChatGenerationChunk(message=message_chunk)
        return generation_chunk

    choice = choices[0]
    if choice["delta"] is None:
        return None

    message_chunk = _convert_delta_to_message_chunk(choice["delta"], default_chunk_class, chunk["id"], is_first_tool_chunk)
    if token_usage and isinstance(message_chunk, AIMessageChunk):
        message_chunk.usage_metadata = _create_usage_metadata(token_usage, _prompt_tokens_included)

    generation_info = {}
    if finish_reason := choice.get("finish_reason"):
        generation_info["finish_reason"] = finish_reason
        if model_name := chunk.get("model"):
            generation_info["model_name"] = model_name
    if logprobs := choice.get("logprobs"):
        generation_info["logprobs"] = logprobs

    generation_chunk = ChatGenerationChunk(message=message_chunk, generation_info=generation_info or None)
    return generation_chunk


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _convert_to_openai_response_format(
    schema: dict[str, Any] | type,
) -> dict | TypeBaseModel:
    if isinstance(schema, type) and is_basemodel_subclass(schema):
        return schema

    if isinstance(schema, dict) and "json_schema" in schema and schema.get("type") == "json_schema":
        return schema

    if isinstance(schema, dict) and "name" in schema and "schema" in schema:
        return {"type": "json_schema", "json_schema": schema}

    function = convert_to_openai_function(schema)
    function["schema"] = function.pop("parameters")
    return {"type": "json_schema", "json_schema": function}


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

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "replicate-chat"

    def _create_message_dicts(
        self,
        messages: list[BaseMessage],
    ) -> list[dict[str, Any]]:
        message_dicts = [_convert_message_to_dict(m, self.model) for m in messages]
        return message_dicts

    def _create_chat_result(self, prediction: Prediction, generation_info: dict | None = None) -> ChatResult:
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
                message.usage_metadata = _create_usage_metadata(token_usage, False)
            generation_info = generation_info or {}
            generation_info["finish_reason"] = choice.get("finish_reason") if choice.get("finish_reason") is not None else generation_info.get("finish_reason")
            if "logprobs" in choice:
                generation_info["logprobs"] = choice["logprobs"]
            generation = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(generation)
        llm_output = {
            "token_usage": token_usage,
            "model_name": response.get("model", self.model),
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_prediction_input(self, messages: list[BaseMessage], stream: bool, stop: list[str] | None, **kwargs: Any) -> dict[str, Any]:
        message_dicts = self._create_message_dicts(messages)

        input_: dict[str, Any] = self.model_kwargs | kwargs | {"messages": message_dicts} | self._stop_input(stop) | self._stream_input(stream)

        # Handle the need to convert ChatCompletionNamedToolChoiceParam value to a string
        tool_choice = input_.get("tool_choice")
        if tool_choice and isinstance(tool_choice, Mapping):
            input_["tool_choice"] = json.dumps(tool_choice)

        return input_

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

        for output in prediction.output_iterator():
            chunk: Mapping[str, Any]
            if isinstance(output, str):
                chunk = json.loads(output)
            elif isinstance(output, Mapping):
                chunk = output
            else:
                logger.error("Unrecognized output shape %s %r", type(output), output)
                raise ValueError(f"Unrecognized output shape {type(output)} {output!r}")

            generation_chunk = _convert_chunk_to_generation_chunk(chunk, default_chunk_class, is_first_tool_chunk, _prompt_tokens_included)
            if generation_chunk is None:
                continue

            default_chunk_class = type(generation_chunk.message)
            if (
                hasattr(generation_chunk.message, "usage_metadata") and generation_chunk.message.usage_metadata  # pyright: ignore[reportAttributeAccessIssue]
            ):
                _prompt_tokens_included = True
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

        async for output in prediction.async_output_iterator():
            chunk: Mapping[str, Any]
            if isinstance(output, str):
                chunk = json.loads(output)
            elif isinstance(output, Mapping):
                chunk = output
            else:
                logger.error("Unrecognized output shape %s %r", type(output), output)
                raise ValueError(f"Unrecognized output shape {type(output)} {output!r}")

            generation_chunk = _convert_chunk_to_generation_chunk(chunk, default_chunk_class, is_first_tool_chunk, _prompt_tokens_included)
            if generation_chunk is None:
                continue

            default_chunk_class = type(generation_chunk.message)
            if (
                hasattr(generation_chunk.message, "usage_metadata") and generation_chunk.message.usage_metadata  # pyright: ignore[reportAttributeAccessIssue]
            ):
                _prompt_tokens_included = True
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

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | Literal["auto", "none", "required", "any"] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Options are:
                    - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                    - ``"auto"``: automatically selects a tool (including no tool).
                    - ``"none"``: does not call a tool.
                    - ``"any"`` or ``"required"`` or ``True``: force at least one tool to be called.
                    - dict of the form ``{"type": "function", "function": {"name": <<tool_name>>}}``: calls <<tool_name>> tool.
                    - ``False`` or ``None``: no effect, default OpenAI behavior.

            kwargs: Any additional parameters are passed directly to
                ``self.bind(**kwargs)``.
        """  # noqa: E501
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [formatted_tool["function"]["name"] for formatted_tool in formatted_tools]
                if not any(tool_name == tool_choice["function"]["name"] for tool_name in tool_names):
                    raise ValueError(f"Tool choice {tool_choice} was specified, but the only provided tools were {tool_names}.")
            else:
                raise ValueError(f"Unrecognized tool_choice type. Expected str, bool or dict. Received: {tool_choice}")

            kwargs["tool_choice"] = tool_choice
        else:
            kwargs["tool_choice"] = "auto"

        return super().bind(tools=formatted_tools, **kwargs)

    @override
    def with_structured_output(
        self,
        schema: dict | type | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class,

                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

            method: The method for steering model generation, one of:

                - ``'function_calling'``: uses tool-calling features.
                - ``'json_schema'``: uses dedicated structured output features.
                - ``'json_mode'``: uses JSON mode.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys ``'raw'``, ``'parsed'``, and ``'parsing_error'``.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:

            - ``'raw'``: BaseMessage
            - ``'parsed'``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - ``'parsing_error'``: Optional[BaseException]

        .. dropdown:: Example: schema=Pydantic class, method="function_calling", include_raw=False

            .. code-block:: python

                from langchain_replicate import ChatReplicate
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatReplicate(...)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, , method="function_calling"
                )

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        .. dropdown:: Example: schema=Pydantic class, method="function_calling", include_raw=True

            .. code-block:: python

                from langchain_replicate import ChatReplicate
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatReplicate(...)
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        .. dropdown:: Example: schema=JSON Schema, method="function_calling", include_raw=False

            .. code-block:: python

                from langchain_replicate import ChatReplicate
                from pydantic import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatReplicate(...)
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        .. dropdown:: Example: schema=Pydantic class, method="json_schema", include_raw=True

            .. code-block::

                from langchain_replicate import ChatReplicate
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str

                llm = ChatReplicate(...)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_schema",
                    include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'chatcmpl-tool-bfbd6f6dd33b438990c5ddf277485971', 'type': 'function', 'function': {'name': 'AnswerWithJustification', 'arguments': '{"answer": "They weigh the same", "justification": "A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound."}'}}]}, response_metadata={'token_usage': {'completion_tokens': 45, 'prompt_tokens': 275, 'total_tokens': 320}, 'model_name': 'meta-llama/llama-3-3-70b-instruct', 'system_fingerprint': '', 'finish_reason': 'stop'}, id='chatcmpl-461ca5bd-1982-412c-b886-017c483bf481---8c18b06eead65ae4691364798787bda7---71896588-efa5-439f-a25f-d1abfe289f5a', tool_calls=[{'name': 'AnswerWithJustification', 'args': {'answer': 'They weigh the same', 'justification': 'A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound.'}, 'id': 'chatcmpl-tool-bfbd6f6dd33b438990c5ddf277485971', 'type': 'tool_call'}], usage_metadata={'input_tokens': 275, 'output_tokens': 45, 'total_tokens': 320}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same', justification='A pound is a unit of weight or mass, so both a pound of bricks and a pound of feathers weigh the same amount, one pound.'),
                #     'parsing_error': None
                # }

        .. dropdown:: Example: schema=function schema, method="json_schema", include_raw=False

            .. code-block:: python

                from langchain_replicate import ChatReplicate
                from pydantic import BaseModel

                function__schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
               }

                llm = ChatReplicate(...)
                structured_llm = llm.with_structured_output(
                    function__schema,
                    method="json_schema",
                    include_raw=False
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        .. dropdown:: Example: schema=Pydantic schema, method="json_mode", include_raw=True

            .. code-block::

                from langchain_replicate import ChatReplicate
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatReplicate(...)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                #     'parsing_error': None
                # }

        .. dropdown:: Example: schema=None, method="json_mode", include_raw=True

            .. code-block::

                from langchain_replicate import ChatReplicate

                llm = ChatReplicate(...)
                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                #     'parsed': {
                #         'answer': 'They are both the same weight.',
                #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
                #     },
                #     'parsing_error': None
                # }
        """  # noqa: E501 # pylint: disable=line-too-long
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError("schema must be specified when method is not 'json_mode'. Received None.")
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: Runnable = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
        elif method == "json_mode":
            llm = self.bind(
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
                raise ValueError("schema must be specified when method is not 'json_mode'. Received None.")
            response_format = _convert_to_openai_response_format(schema)
            if is_pydantic_schema:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,  # type: ignore[union-attr]
                        "description": schema.__doc__,
                        "schema": schema.model_json_schema(),  # type: ignore[union-attr]
                    },
                }
            bind_kwargs = {
                "response_format": response_format,
                "ls_structured_output_format": {
                    "kwargs": {"method": method},
                    "schema": convert_to_openai_tool(schema),
                },
            }
            llm = self.bind(**bind_kwargs)
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(f"Unrecognized method argument. Expected one of 'function_calling', 'json_mode' or 'json_schema'. Received: '{method}'")

        if include_raw:
            parser_assign = RunnablePassthrough.assign(parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None)
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks([parser_none], exception_key="parsing_error")
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser
