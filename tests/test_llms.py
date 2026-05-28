# SPDX-License-Identifier: MIT

"""Test Replicate API wrapper."""

import pytest
from assertpy import assert_that
from pydantic.types import SecretStr

from langchain_replicate import Replicate

TEST_MODEL_HELLO = "replicate/hello-world:9dcd6d78e7c6560c340d916fe32e9f24aabfa331e5cce95fe31f77fb03121426"
TEST_MODEL_LANG = "meta/meta-llama-3-8b-instruct"
TEST_DEPLOYMENT_LANG = "ibm-granite/deployment-granite-4-1-8b:deployment"


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
        # pyrefly: ignore [unexpected-keyword]
        llm = Replicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 10, "temperature": 0.01})
        long_output = llm.invoke("What is Pi?")
        # pyrefly: ignore [unexpected-keyword]
        llm = Replicate(model=TEST_MODEL_LANG, model_kwargs={"max_tokens": 5, "temperature": 0.01})
        short_output = llm.invoke("What is Pi?")
        assert_that(len(short_output)).is_less_than(len(long_output))
        assert_that(llm.model_kwargs).contains_only("max_tokens", "temperature").contains_entry({"max_tokens": 5}, {"temperature": 0.01})

    def test_alias_input(self) -> None:
        """Test input model_kwarg alias parameter."""
        llm = Replicate(model=TEST_MODEL_LANG, input={"max_tokens": 10})
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

    def test_invoke_multiple_stop_sequences(self) -> None:
        """Test invoke with multiple stop sequences uses earliest occurrence."""
        llm = Replicate(model=TEST_MODEL_LANG)
        # Request a longer response and use multiple stop sequences in random order
        # The model should stop at whichever sequence appears first in the output
        output = llm.invoke("Count from 1 to 10", stop=["9", "5", "8"])
        assert_that(output).is_not_none().is_not_empty()
        # Output should be truncated at the earliest stop sequence
        # It should not contain any of the stop sequences
        assert_that(output).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")

    @pytest.mark.asyncio
    async def test_ainvoke_multiple_stop_sequences(self) -> None:
        """Test ainvoke with multiple stop sequences uses earliest occurrence."""
        llm = Replicate(model=TEST_MODEL_LANG)
        # Request a longer response and use multiple stop sequences in different order
        output = await llm.ainvoke("Count from 1 to 10", stop=["8", "9", "5"])
        assert_that(output).is_not_none().is_not_empty()
        # Output should be truncated at the earliest stop sequence
        assert_that(output).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")

    def test_stream_multiple_stop_sequences(self) -> None:
        """Test stream with multiple stop sequences uses earliest occurrence."""
        llm = Replicate(model=TEST_MODEL_LANG)
        # Use yet another ordering to verify order independence
        stream = llm.stream("Count from 1 to 10", stop=["5", "9", "8", "elephant"])
        combined = next(stream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += "".join(stream)
            assert_that(combined).is_not_empty()
            # Output should be truncated at the earliest stop sequence
            assert_that(combined).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")

    @pytest.mark.asyncio
    async def test_astream_multiple_stop_sequences(self) -> None:
        """Test astream with multiple stop sequences uses earliest occurrence."""
        llm = Replicate(model=TEST_MODEL_LANG)
        # Use a fourth ordering to thoroughly test order independence
        astream = llm.astream("Count from 1 to 10", stop=["8", "5", "9"])
        combined = await anext(astream, None)
        assert_that(combined).is_not_none()
        if combined is not None:
            combined += "".join([chunk async for chunk in astream])
            assert_that(combined).is_not_empty()
            # Output should be truncated at the earliest stop sequence
            assert_that(combined).contains("1", "2", "3", "4").does_not_contain("5", "6", "7", "8", "9")
