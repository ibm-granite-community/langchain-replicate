# SPDX-License-Identifier: MIT

"""Test Replicate API wrapper."""

import pytest
from assertpy import assert_that
from pydantic.types import SecretStr

from langchain_replicate import Replicate

TEST_MODEL_HELLO = "replicate/hello-world:5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"
TEST_MODEL_LANG = "meta/meta-llama-3-8b-instruct"
TEST_DEPLOYMENT_LANG = "ibm-granite/deployment-granite-4-0-h-small:deployment"


class TestLLM:
    def test_invoke(self) -> None:
        """Test invoke."""
        llm = Replicate(model=TEST_MODEL_HELLO)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke(self) -> None:
        """Test ainvoke."""
        llm = Replicate(model=TEST_MODEL_HELLO)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_not_none().is_not_empty().contains("Pi")

    def test_invoke_deployment(self) -> None:
        """Test invoke."""
        llm = Replicate(model=TEST_DEPLOYMENT_LANG)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke_deployment(self) -> None:
        """Test ainvoke."""
        llm = Replicate(model=TEST_DEPLOYMENT_LANG)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_not_none().is_not_empty().contains("Pi")

    def test_with_apikey(self, replicate_api_token: SecretStr) -> None:
        """Test specifying api key."""
        llm = Replicate(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_not_none().is_not_empty().contains("Pi")

    def test_invoke_streaming(self) -> None:
        """Test invoke streaming."""
        llm = Replicate(streaming=True, model=TEST_MODEL_HELLO)
        output = llm.invoke("What is Pi?")
        assert_that(output).is_not_none().is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_ainvoke_streaming(self) -> None:
        """Test invoke streaming."""
        llm = Replicate(streaming=True, model=TEST_MODEL_HELLO)
        output = await llm.ainvoke("What is Pi?")
        assert_that(output).is_not_none().is_not_empty().contains("Pi")

    def test_model_kwargs(self) -> None:
        """Test model_kwargs."""
        llm = Replicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 10, "temperature": 0.01})
        long_output = llm.invoke("What is Pi?")
        llm = Replicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 5, "temperature": 0.01})
        short_output = llm.invoke("What is Pi?")
        assert_that(len(short_output)).is_less_than(len(long_output))
        assert_that(llm.model_kwargs).contains_only("max_tokens", "temperature").contains_entry({"max_tokens": 5}, {"temperature": 0.01})

    def test_alias_input(self) -> None:
        """Test input model_kwarg alias parameter."""
        llm = Replicate(model=TEST_MODEL_LANG, input={"max_tokens": 10})  # type: ignore
        assert_that(llm.model_kwargs).contains_only("max_tokens").contains_entry({"max_tokens": 10})

    def test_stream(self) -> None:
        """Test stream."""
        llm = Replicate(model=TEST_MODEL_LANG)
        stream = llm.stream("What is Pi?")
        combined = next(stream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += "".join(stream)
            assert_that(combined).is_not_empty().contains("Pi")

    @pytest.mark.asyncio
    async def test_astream(self) -> None:
        """Test astream."""
        llm = Replicate(model=TEST_MODEL_LANG)
        astream = llm.astream("What is Pi?")
        combined = await anext(astream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += "".join([chunk async for chunk in astream])
            assert_that(combined).is_not_empty().contains("Pi")
