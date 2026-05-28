# SPDX-License-Identifier: MIT

"""Test ChatReplicate API wrapper."""

from typing import Any, cast

import pytest
from assertpy import assert_that
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from pydantic.types import SecretStr

from langchain_replicate import ChatReplicate

TEST_MODEL_LANG = "ibm-granite/granite-4.1-8b"
TEST_DEPLOYMENT_LANG = "ibm-granite/deployment-granite-4-1-8b:deployment"


class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification for the answer."""

    answer: str
    justification: str


class TestChat:
    def test_invoke(self) -> None:
        """Test invoke."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke(self) -> None:
        """Test ainvoke."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    def test_invoke_deployment(self) -> None:
        """Test invoke."""
        llm = ChatReplicate(model=TEST_DEPLOYMENT_LANG)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke_deployment(self) -> None:
        """Test ainvoke."""
        llm = ChatReplicate(model=TEST_DEPLOYMENT_LANG)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    def test_with_apikey(self, replicate_api_token: SecretStr) -> None:
        """Test with apikey."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, replicate_api_token=replicate_api_token)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    def test_invoke_streaming(self) -> None:
        """Test invoke streaming."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, streaming=True)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke_streaming(self) -> None:
        """Test ainvoke streaming."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, streaming=True)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    def test_model_kwargs(self) -> None:
        """Test model_kwargs."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 10, "temperature": 0.01})
        long_output = llm.invoke("What is Pi?")
        llm = ChatReplicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 5, "temperature": 0.01})
        short_output = llm.invoke("What is Pi?")
        assert_that(len(short_output.text)).is_less_than(len(long_output.text))
        assert_that(llm.model_kwargs).contains_only("max_tokens", "temperature").contains_entry({"max_tokens": 5}, {"temperature": 0.01})

    def test_invoke_stop(self) -> None:
        """Test invoke stop parameter."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, model_kwargs={"temperature": 0.01})
        long_output = llm.invoke("What is Pi?")
        short_output = llm.invoke("What is Pi?", stop=["3.14"])
        assert_that(long_output.text).contains("3.14")
        assert_that(short_output.text).does_not_contain("3.14")

    def test_stream(self) -> None:
        """Test stream."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        stream = llm.stream("What is Pi?")
        combined = next(stream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += list(stream)
            assert_that(combined.text).is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_astream(self) -> None:
        """Test astream."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        astream = llm.astream("What is Pi?")
        combined = await anext(astream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += [chunk async for chunk in astream]
            assert_that(combined.text).is_not_empty().contains("Pi")

    def test_content_array(self) -> None:
        """Test content array."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content=[{"type": "text", "text": "What is Pi?"}])]
        output = llm.invoke(messages)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Pi")

    def test_documents(self, documents: list[dict[str, Any]]) -> None:
        """Test documents."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What did the president say about Ketanji Brown Jackson?")]
        output = llm.invoke(messages, documents=documents)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Supreme")

    def test_chat_template_kwargs(self, documents: list[dict[str, Any]]) -> None:
        """Test chat_template_kwargs."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What did the president say about Ketanji Brown Jackson?")]
        chat_template_kwargs: dict[str, Any] = {"documents": documents}
        output = llm.invoke(messages, chat_template_kwargs=chat_template_kwargs)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Supreme")

    def test_tool_call(self, tools: list[dict[str, Any]]) -> None:
        """Test tools."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What is the weather like in Boston today?")]
        llm_tools = llm.bind_tools(tools=tools)
        response = llm_tools.invoke(messages)
        assert_that(response).is_instance_of(AIMessage)
        assert_that(response.tool_calls).is_not_none().is_length(1)
        assert_that(response.tool_calls[0]["name"]).is_equal_to("get_current_weather")
        assert_that(response.tool_calls[0]["args"]["location"]).contains("Boston")

    def test_tool_call_streaming(self, tools: list[dict[str, Any]]) -> None:
        """Test tools."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, streaming=True)
        messages = [HumanMessage(content="What is the weather like in Boston today?")]
        llm_tools = llm.bind_tools(tools=tools)
        response = llm_tools.invoke(messages)
        assert_that(response).is_instance_of(AIMessage)
        assert_that(response.tool_calls).is_not_none().is_length(1)
        assert_that(response.tool_calls[0]["name"]).is_equal_to("get_current_weather")
        assert_that(response.tool_calls[0]["args"]["location"]).contains("Boston")

    def test_tool_choice(self, tools: list[dict[str, Any]]) -> None:
        """Test tool_choice."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What is the weather like in Boston today?")]
        llm_tools = llm.bind_tools(tools=tools, tool_choice="get_current_weather")
        response = llm_tools.invoke(messages)
        assert_that(response).is_instance_of(AIMessage)
        assert_that(response.tool_calls).is_not_none().is_length(1)
        assert_that(response.tool_calls[0]["name"]).is_equal_to("get_current_weather")
        assert_that(response.tool_calls[0]["args"]["location"]).contains("Boston")

    def test_tool_response(self, tools: list[dict[str, Any]]) -> None:
        """Test tool response."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [
            HumanMessage(content="What is the weather like in Boston today?"),
            AIMessage(content="", tool_calls=[ToolCall(id="chatcmpl-tool-92d6ee08f1f14bab9a593271ca4174ab", name="function", args={"location": "Boston, MA", "unit": "celsius"})]),
            ToolMessage(content='{"description": "sunny", "temperature": 10, "humidity": 50}', tool_call_id="chatcmpl-tool-92d6ee08f1f14bab9a593271ca4174ab"),
        ]
        output = llm.invoke(messages, tools=tools)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty().contains("Boston", "10", "50")

    def test_structured_response_pydantic(self) -> None:
        """Test structured response with Pydantic class."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        structured_llm = llm.with_structured_output(AnswerWithJustification, method="function_calling")
        output = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
        assert_that(output).is_not_none().is_instance_of(AnswerWithJustification)
        response = cast(AnswerWithJustification, output)
        assert_that(response.answer).is_not_none().is_not_empty()
        assert_that(response.justification).is_not_none().is_not_empty()

    def test_structured_response_pydantic_raw(self) -> None:
        """Test raw structured response with Pydantic class."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        structured_llm = llm.with_structured_output(AnswerWithJustification, method="function_calling", include_raw=True)
        output = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
        assert_that(output).is_not_none().is_instance_of(dict)
        response = cast(dict[str, Any], output)
        assert_that(response).contains_key("raw", "parsed", "parsing_error")
        assert_that(response["parsed"]).is_instance_of(AnswerWithJustification)
        parsed = cast(AnswerWithJustification, response["parsed"])
        assert_that(parsed.answer).is_not_none().is_not_empty()
        assert_that(parsed.justification).is_not_none().is_not_empty()
        assert_that(response["raw"]).is_instance_of(AIMessage)
        raw = cast(AIMessage, response["raw"])
        assert_that(raw.tool_calls).is_not_none().is_length(1)
        assert_that(raw.tool_calls[0]["name"]).is_equal_to("AnswerWithJustification")
        assert_that(raw.tool_calls[0]["args"]["answer"]).is_equal_to(parsed.answer)
        assert_that(raw.tool_calls[0]["args"]["justification"]).is_equal_to(parsed.justification)
        assert_that(response["parsing_error"]).is_none()

    def test_structured_response_schema(self) -> None:
        """Test structured response with schema dict."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        schema = convert_to_openai_tool(AnswerWithJustification)
        structured_llm = llm.with_structured_output(schema)
        output = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
        assert_that(output).is_not_none().is_instance_of(dict)
        response = cast(dict[str, Any], output)
        assert_that(response["answer"]).is_not_none().is_not_empty()
        assert_that(response["justification"]).is_not_none().is_not_empty()

    def test_structured_response_json_schema(self) -> None:
        """Test structured response with Pydantic class."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        structured_llm = llm.with_structured_output(AnswerWithJustification, method="json_schema")
        output = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
        assert_that(output).is_not_none().is_instance_of(AnswerWithJustification)
        response = cast(AnswerWithJustification, output)
        assert_that(response.answer).is_not_none().is_not_empty()
        assert_that(response.justification).is_not_none().is_not_empty()

    def test_structured_response_json_mode(self) -> None:
        """Test structured response with Pydantic class."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)
        output = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
        assert_that(output).is_not_none().is_instance_of(dict)
        response = cast(dict[str, Any], output)
        assert_that(response).contains_key("raw", "parsed", "parsing_error")
        assert_that(response["raw"]).is_instance_of(AIMessage)
        assert_that(response["parsed"]).is_instance_of(dict)
        assert_that(response["parsing_error"]).is_none()

    def test_invoke_multiple_stop_sequences(self) -> None:
        """Test invoke with multiple stop sequences uses earliest occurrence."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        # Request a longer response and use multiple stop sequences in random order
        # The model should stop at whichever sequence appears first in the output
        output = llm.invoke("Count from 1 to 10", stop=["9", "5", "8"])
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty()
        # Output should be truncated at the earliest stop sequence
        # It should not contain any of the stop sequences
        assert_that(output.text).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")

    @pytest.mark.asyncio
    async def test_ainvoke_multiple_stop_sequences(self) -> None:
        """Test ainvoke with multiple stop sequences uses earliest occurrence."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        # Request a longer response and use multiple stop sequences in different order
        output = await llm.ainvoke("Count from 1 to 10", stop=["8", "9", "5"])
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text).is_not_none().is_not_empty()
        # Output should be truncated at the earliest stop sequence
        assert_that(output.text).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")

    def test_stream_multiple_stop_sequences(self) -> None:
        """Test stream with multiple stop sequences uses earliest occurrence."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        # Use yet another ordering to verify order independence
        stream = llm.stream("Count from 1 to 10", stop=["5", "9", "8"])
        combined = next(stream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += list(stream)
            assert_that(combined).is_instance_of(AIMessage)
            assert_that(combined.text).is_not_empty()
            # Output should be truncated at the earliest stop sequence
            assert_that(combined.text).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")

    @pytest.mark.asyncio
    async def test_astream_multiple_stop_sequences(self) -> None:
        """Test astream with multiple stop sequences uses earliest occurrence."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        # Use a fourth ordering to thoroughly test order independence
        astream = llm.astream("Count from 1 to 10", stop=["8", "5", "9"])
        combined = await anext(astream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += [chunk async for chunk in astream]
            assert_that(combined).is_instance_of(AIMessage)
            assert_that(combined.text).is_not_empty()
            # Output should be truncated at the earliest stop sequence
            assert_that(combined.text).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")
