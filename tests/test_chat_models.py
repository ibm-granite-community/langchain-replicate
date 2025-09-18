# SPDX-License-Identifier: MIT

"""Test ChatReplicate API wrapper."""

from typing import Any, cast

import pytest
from assertpy import assert_that
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from langchain_replicate import ChatReplicate

TEST_MODEL_LANG = "ibm-granite/granite-3.3-8b-instruct"


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
        assert_that(output.text()).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke(self) -> None:
        """Test ainvoke."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Pi")

    def test_with_apikey(self, replicate_api_token: str) -> None:
        """Test with apikey."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, replicate_api_token=replicate_api_token)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Pi")

    def test_invoke_streaming(self) -> None:
        """Test invoke streaming."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, streaming=True)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke_streaming(self) -> None:
        """Test ainvoke streaming."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, streaming=True)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Pi")

    def test_model_kwargs(self) -> None:
        """Test model_kwargs."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 10, "temperature": 0.01})
        long_output = llm.invoke("What is Pi?")
        llm = ChatReplicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 5, "temperature": 0.01})
        short_output = llm.invoke("What is Pi?")
        assert_that(len(short_output.text())).is_less_than(len(long_output.text()))
        assert_that(llm.model_kwargs).contains_only("max_tokens", "temperature").contains_entry({"max_tokens": 5}, {"temperature": 0.01})

    def test_invoke_stop(self) -> None:
        """Test invoke stop parameter."""
        llm = ChatReplicate(model=TEST_MODEL_LANG, model_kwargs={"temperature": 0.01})
        long_output = llm.invoke("What is Pi?")
        short_output = llm.invoke("What is Pi?", stop=["3.14"])
        assert_that(long_output.text()).contains("3.14")
        assert_that(short_output.text()).does_not_contain("3.14")

    def test_stream(self) -> None:
        """Test stream."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        stream = llm.stream("What is Pi?")
        combined = next(stream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += list(stream)
            assert_that(combined.text()).is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_astream(self) -> None:
        """Test astream."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        astream = llm.astream("What is Pi?")
        combined = await anext(astream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += [chunk async for chunk in astream]
            assert_that(combined.text()).is_not_empty().contains("Pi")

    def test_content_array(self) -> None:
        """Test content array."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content=[{"type": "text", "text": "What is Pi?"}])]
        output = llm.invoke(messages)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Pi")

    def test_documents(self, documents: list[dict[str, Any]]) -> None:
        """Test documents."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What did the president say about Ketanji Brown Jackson?")]
        output = llm.invoke(messages, documents=documents)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Supreme")

    def test_chat_template_kwargs(self, documents: list[dict[str, Any]]) -> None:
        """Test chat_template_kwargs."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What did the president say about Ketanji Brown Jackson?")]
        chat_template_kwargs: dict[str, Any] = {"documents": documents}
        output = llm.invoke(messages, chat_template_kwargs=chat_template_kwargs)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Supreme")

    def test_tool_call(self, tools: list[dict[str, Any]]) -> None:
        """Test tools."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What is the weather like in Boston today?")]
        llm_tools = llm.bind_tools(tools=tools)
        output = llm_tools.invoke(messages)
        assert_that(output).is_instance_of(AIMessage)
        response = cast(AIMessage, output)
        assert_that(response.tool_calls).is_not_none().is_length(1)
        assert_that(response.tool_calls[0]["name"]).is_equal_to("get_current_weather")

    def test_tool_choice(self, tools: list[dict[str, Any]]) -> None:
        """Test tool_choice."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [HumanMessage(content="What is the weather like in Boston today?")]
        llm_tools = llm.bind_tools(tools=tools, tool_choice="get_current_weather")
        output = llm_tools.invoke(messages)
        assert_that(output).is_instance_of(AIMessage)
        response = cast(AIMessage, output)
        assert_that(response.tool_calls).is_not_none().is_length(1)
        assert_that(response.tool_calls[0]["name"]).is_equal_to("get_current_weather")

    def test_tool_response(self, tools: list[dict[str, Any]]) -> None:
        """Test tool response."""
        llm = ChatReplicate(model=TEST_MODEL_LANG)
        messages = [
            HumanMessage(content="What is the weather like in Boston today?"),
            AIMessage(content="", tool_calls=[ToolCall(id="chatcmpl-tool-92d6ee08f1f14bab9a593271ca4174ab", name="function", args={"location": "Boston, MA", "unit": "celsius"})]),
            ToolMessage(content="Boston is sunny with a temperature of 30°C.", tool_call_id="chatcmpl-tool-92d6ee08f1f14bab9a593271ca4174ab"),
        ]
        output = llm.invoke(messages, tools=tools)
        assert_that(output).is_instance_of(AIMessage)
        assert_that(output.text()).is_not_none().is_not_empty().contains("Boston", "30")

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
        assert_that(response["raw"]).is_instance_of(AIMessage)
        assert_that(response["parsed"]).is_instance_of(AnswerWithJustification)
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
