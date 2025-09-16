# SPDX-License-Identifier: MIT

"""Test Replicate API wrapper."""

from assertpy import assert_that

from langchain_replicate import Replicate

from .fake_callback_handler import FakeCallbackHandler  # pylint: disable=relative-beyond-top-level

TEST_MODEL_HELLO = "replicate/hello-world:5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"
TEST_MODEL_LANG = "meta/meta-llama-3-8b-instruct"


class TestLLM:
    def test_replicate_call(self) -> None:
        """Test simple non-streaming call."""
        llm = Replicate(model=TEST_MODEL_HELLO)
        output = llm.invoke("What is LangChain")
        assert_that(output).is_instance_of(str)

    def test_replicate_call_with_apikey(self, replicate_api_token: str) -> None:
        """Test specifying api key."""
        llm = Replicate(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token)
        output = llm.invoke("What is LangChain")
        assert_that(output).is_instance_of(str)

    def test_replicate_streaming_call(self) -> None:
        """Test streaming call."""
        callback_handler = FakeCallbackHandler()
        llm = Replicate(streaming=True, callbacks=[callback_handler], model=TEST_MODEL_HELLO)
        output = llm.invoke("What is LangChain")
        assert_that(output).is_instance_of(str)

    def test_replicate_model_kwargs(self) -> None:
        """Test simple non-streaming call."""
        llm = Replicate(  # type: ignore[call-arg]
            model=TEST_MODEL_LANG, model_kwargs={"max_new_tokens": 10, "temperature": 0.01}
        )
        long_output = llm.invoke("What is LangChain")
        llm = Replicate(  # type: ignore[call-arg]
            model=TEST_MODEL_LANG, model_kwargs={"max_new_tokens": 5, "temperature": 0.01}
        )
        short_output = llm.invoke("What is LangChain")
        assert_that(len(short_output)).is_less_than(len(long_output))
        assert_that(llm.model_kwargs).contains_only("max_new_tokens", "temperature").contains_entry({"max_new_tokens": 5}, {"temperature": 0.01})

    def test_replicate_input(self) -> None:
        """Test alias input parameter."""
        llm = Replicate(model=TEST_MODEL_LANG, input={"max_new_tokens": 10})  # type: ignore
        assert_that(llm.model_kwargs).contains_only("max_new_tokens").contains_entry({"max_new_tokens": 10})
