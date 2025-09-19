# SPDX-License-Identifier: MIT

"""Test ReplicateEmbeddings API wrapper."""

import json

import pytest
from assertpy import assert_that

from langchain_replicate import ReplicateEmbeddings

TEST_MODEL_WARM = "beautyyuyanli/multilingual-e5-large:a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad"
TEST_MODEL_COLD = "ibm-granite/granite-embedding-278m-multilingual"


class TestEmbedding:
    def test_embed_query(self, embed_texts: list[str]) -> None:
        """Test embedding of single string."""
        llm = ReplicateEmbeddings(model=TEST_MODEL_WARM, texts_value_mapping=json.dumps)
        output = llm.embed_query(embed_texts[0])
        assert_that(output).is_instance_of(list).is_not_empty()
        assert_that(output[0]).is_instance_of(float)

    @pytest.mark.asyncio
    async def test_aembed_query(self, embed_texts: list[str]) -> None:
        """Test async embedding of single string."""
        llm = ReplicateEmbeddings(model=TEST_MODEL_WARM, texts_value_mapping=json.dumps)
        output = await llm.aembed_query(embed_texts[0])
        assert_that(output).is_instance_of(list).is_not_empty()
        assert_that(output[0]).is_instance_of(float)

    def test_embed_query_with_apikey(self, embed_texts: list[str], replicate_api_token: str) -> None:
        """Test specifying api key."""
        llm = ReplicateEmbeddings(model=TEST_MODEL_WARM, texts_value_mapping=json.dumps, replicate_api_token=replicate_api_token)
        output = llm.embed_query(embed_texts[0])
        assert_that(output).is_instance_of(list).is_not_empty()
        assert_that(output[0]).is_instance_of(float)

    def test_embed_documents(self, embed_texts: list[str]) -> None:
        """Test embedding of multiple strings."""
        llm = ReplicateEmbeddings(model=TEST_MODEL_WARM, texts_value_mapping=json.dumps)
        output = llm.embed_documents(embed_texts)
        assert_that(output).is_instance_of(list).is_not_empty().is_length(len(embed_texts))
        assert_that(output[0]).is_instance_of(list).is_not_empty()
        assert_that(output[0][0]).is_instance_of(float)

    @pytest.mark.asyncio
    async def test_aembed_documents(self, embed_texts: list[str]) -> None:
        """Test async embedding of multiple strings."""
        llm = ReplicateEmbeddings(model=TEST_MODEL_WARM, texts_value_mapping=json.dumps)
        output = await llm.aembed_documents(embed_texts)
        assert_that(output).is_instance_of(list).is_not_empty().is_length(len(embed_texts))
        assert_that(output[0]).is_instance_of(list).is_not_empty()
        assert_that(output[0][0]).is_instance_of(float)

    def test_embed_model_kwargs(self) -> None:
        """Test model_kwargs."""
        llm = ReplicateEmbeddings(model=TEST_MODEL_WARM, texts_value_mapping=json.dumps, model_kwargs={"batch_size": 16})
        assert_that(llm.model_kwargs).contains_only("batch_size").contains_entry({"batch_size": 16})

    @pytest.mark.slow
    def test_embed_cold(self, embed_texts: list[str]) -> None:
        """Test embedding of multiple strings on cold model with default list[str] input shape."""
        llm = ReplicateEmbeddings(model=TEST_MODEL_COLD)
        output = llm.embed_documents(embed_texts)
        assert_that(output).is_instance_of(list).is_not_empty().is_length(len(embed_texts))
        assert_that(output[0]).is_instance_of(list).is_not_empty()
        assert_that(output[0][0]).is_instance_of(float)
