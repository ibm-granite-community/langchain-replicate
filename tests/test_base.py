# SPDX-License-Identifier: MIT

"""Test Replicate base type."""

import json

from assertpy import assert_that
from pydantic.types import SecretStr
from replicate.client import Client
from replicate.version import Version

from langchain_replicate import Replicate
from langchain_replicate._base import ReplicateBase

TEST_MODEL_HELLO = "replicate/hello-world:5c7d5dc6dd8bf75c1acaa8565735e7986bc5b66206b55cca93cb72c9bf15ccaa"
TEST_MODEL_LANG = "meta/meta-llama-3-8b-instruct"
TEST_DEPLOYMENT_LANG = "ibm-granite/deployment-granite-4-0-h-small:deployment"


class ReplicateBaseTest(ReplicateBase):
    """Subclass of abstract base class for testing"""

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "replicate-test"


class TestBase:
    def test_version(self, replicate_api_token: SecretStr) -> None:
        client = Client(api_token=replicate_api_token.get_secret_value())
        model_str, version_str = TEST_MODEL_HELLO.split(":")
        model = client.models.get(model_str)
        version = model.versions.get(version_str)
        assert_that(version).is_instance_of(Version)
        base = ReplicateBaseTest(model=model_str, replicate_api_token=replicate_api_token, version_obj=version)
        assert_that(base.version_obj).is_instance_of(Version).is_same_as(version)
        assert_that(base._version).is_instance_of(Version).is_same_as(version)  # pylint: disable=protected-access

    def test_version_bad(self, replicate_api_token: SecretStr) -> None:
        assert_that(ReplicateBaseTest).raises(ValueError).when_called_with(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token, version_obj="invalid version")

    def test_version_none(self, replicate_api_token: SecretStr) -> None:
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token, version_obj=None)
        assert_that(base.version_obj).is_none()
        assert_that(base._version).is_instance_of(Version)  # pylint: disable=protected-access

    def test_version_deployment(self, replicate_api_token: SecretStr) -> None:
        base = ReplicateBaseTest(model=TEST_DEPLOYMENT_LANG, replicate_api_token=replicate_api_token)
        assert_that(base.version_obj).is_none()
        assert_that(base._version).is_instance_of(Version)  # pylint: disable=protected-access

    def test_version_not_specified(self, replicate_api_token: SecretStr) -> None:
        base = ReplicateBaseTest(model=TEST_MODEL_LANG, replicate_api_token=replicate_api_token)
        assert_that(base.version_obj).is_none()
        assert_that(base._version).is_instance_of(Version)  # pylint: disable=protected-access

    def test_input_properties(self, replicate_api_token: SecretStr) -> None:
        llm = Replicate(model=TEST_MODEL_HELLO, replicate_api_token=replicate_api_token)
        input_properties = llm._input_properties  # pylint: disable=protected-access
        assert_that(input_properties).is_instance_of(dict).is_length(1).contains_key("text")
        assert_that(input_properties["text"]).is_instance_of(dict).contains_entry({"type": "string"}, {"x-order": 0})

    def test_input_properties_sorted(self, replicate_api_token: SecretStr) -> None:
        llm = Replicate(model=TEST_MODEL_LANG, replicate_api_token=replicate_api_token)
        input_properties = llm._input_properties  # pylint: disable=protected-access
        assert_that(input_properties).is_instance_of(dict).is_not_empty()
        assert_that(input_properties.items()).is_sorted(key=lambda item: item[1].get("x-order", 0))

    def test_input_properties_deployment(self, replicate_api_token: SecretStr) -> None:
        llm = Replicate(model=TEST_DEPLOYMENT_LANG, replicate_api_token=replicate_api_token)
        input_properties = llm._input_properties  # pylint: disable=protected-access
        assert_that(input_properties).is_instance_of(dict).is_not_empty()
        assert_that(input_properties.items()).is_sorted(key=lambda item: item[1].get("x-order", 0))

    def test_api_token_secret_str(self) -> None:
        api_token = "secret test"
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO, replicate_api_token=SecretStr(api_token))
        assert_that(base.replicate_api_token).is_instance_of(SecretStr)
        assert_that(base.replicate_api_token.get_secret_value()).is_equal_to(api_token)  # type: ignore

    def test_api_token_str(self) -> None:
        api_token = "secret test"
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO, replicate_api_token=api_token)
        assert_that(base.replicate_api_token).is_instance_of(SecretStr)
        assert_that(base.replicate_api_token.get_secret_value()).is_equal_to(api_token)  # type: ignore

    def test_api_token_none(self) -> None:
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO, replicate_api_token=None)
        assert_that(base.replicate_api_token).is_none()

    def test_api_token_not_specified(self) -> None:
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO)
        assert_that(base.replicate_api_token).is_none()

    def test_api_token_bad(self) -> None:
        assert_that(ReplicateBaseTest).raises(ValueError).when_called_with(model=TEST_MODEL_HELLO, replicate_api_token=[2, 3])

    def test_api_token_dump(self) -> None:
        api_token = "secret test"
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO, replicate_api_token=api_token)

        dump = base.model_dump()
        assert_that(dump).contains_key("replicate_api_token")
        assert_that(dump["replicate_api_token"]).is_instance_of(SecretStr)

        dump_json = base.model_dump_json()
        assert_that(json.loads(dump_json)).contains_entry({"replicate_api_token": api_token})

    def test_api_token_not_specified_dump(self) -> None:
        base = ReplicateBaseTest(model=TEST_MODEL_HELLO)

        dump = base.model_dump()
        assert_that(dump).contains_entry({"replicate_api_token": None})

        dump_json = base.model_dump_json()
        assert_that(json.loads(dump_json)).contains_entry({"replicate_api_token": None})
