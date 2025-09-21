# SPDX-License-Identifier: MIT

"""Test Replicate base type."""

from assertpy import assert_that
from replicate.client import Client
from replicate.version import Version

from langchain_replicate import Replicate
from langchain_replicate._base import ReplicateBase

TEST_MODEL_HELLO = "replicate/hello-world:5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"
TEST_MODEL_LANG = "meta/meta-llama-3-8b-instruct"


class ReplicateBaseTest(ReplicateBase):
    """Subclass of abstract base class for testing"""

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "replicate-test"


class TestBase:
    def test_version(self, replicate_api_token: str) -> None:
        client = Client(api_token=replicate_api_token)
        model_str, version_str = TEST_MODEL_HELLO.split(":")
        model = client.models.get(model_str)
        version = model.versions.get(version_str)
        assert_that(version).is_instance_of(Version)
        base = ReplicateBaseTest(model=model_str, replicate_api_token=replicate_api_token, version_obj=version)
        assert_that(base.version_obj).is_instance_of(Version).is_same_as(version)
        assert_that(base._version).is_instance_of(Version).is_same_as(version)  # pylint: disable=protected-access

    def test_version_bad(self, replicate_api_token: str) -> None:
        assert_that(ReplicateBaseTest).raises(ValueError).when_called_with(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token, version_obj="invalid version")

    def test_version_none(self, replicate_api_token: str) -> None:
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token, version_obj=None)
        assert_that(base.version_obj).is_none()
        assert_that(base._version).is_instance_of(Version)  # pylint: disable=protected-access

    def test_version_not_specified(self, replicate_api_token: str) -> None:
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token)
        assert_that(base.version_obj).is_none()
        assert_that(base._version).is_instance_of(Version)  # pylint: disable=protected-access

    def test_input_properties(self, replicate_api_token: str) -> None:
        llm = Replicate(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token)
        input_properties = llm._input_properties  # pylint: disable=protected-access
        assert_that(input_properties).is_instance_of(dict).is_length(1).contains_key("text")
        assert_that(input_properties["text"]).is_instance_of(dict).contains_entry({"type": "string"}, {"x-order": 0})

    def test_input_properties_sorted(self, replicate_api_token: str) -> None:
        llm = Replicate(model=TEST_MODEL_LANG, replicate_api_token=replicate_api_token)
        input_properties = llm._input_properties  # pylint: disable=protected-access
        assert_that(input_properties).is_instance_of(dict)
        assert_that(input_properties.items()).is_sorted(key=lambda item: item[1].get("x-order", 0))
