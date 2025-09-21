# SPDX-License-Identifier: MIT

from __future__ import annotations

import abc
from functools import cached_property
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field
from replicate.client import Client
from replicate.prediction import Prediction
from replicate.version import Version


def _is_version(value: Any) -> Version | None:
    """We use `Any` as the pydantic type of `version_obj` as replicate is still on
    pydantic v1 and we are v2. So we use a validator
    to ensure the value is `Version | None`."""
    if value is None or isinstance(value, Version):
        return value
    raise ValueError(f"The {value} object is not of type {Version}")


class ReplicateBase(BaseModel, abc.ABC):
    """Base model for Replicate integrations"""

    model: str
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    replicate_api_token: str | None = None
    version_obj: Annotated[Any | None, Field(exclude=True), AfterValidator(_is_version)] = None
    """Optionally pass in the model version object during initialization to avoid
        having to make an extra API call to retrieve it during streaming. NOTE: not
        serializable, is excluded from serialization."""

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"replicate_api_token": "REPLICATE_API_TOKEN"}

    @cached_property
    def _client(self) -> Client:
        return Client(api_token=self.replicate_api_token)

    @cached_property
    def _version(self) -> Version:
        if self.version_obj:
            return self.version_obj
        if ":" in self.model:  # not an official model
            model_str, version_str = self.model.split(":")
            model = self._client.models.get(model_str)
            return model.versions.get(version_str)
        model = self._client.models.get(self.model)
        return model.latest_version  # type: ignore

    @cached_property
    def _input_properties(self) -> dict[str, Any]:
        """Sort the openapi schema Input properties in x-order"""
        input_properties = sorted(
            self._version.openapi_schema["components"]["schemas"]["Input"]["properties"].items(),
            key=lambda item: item[1].get("x-order", 0),
        )
        return dict(input_properties)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "model_kwargs": self.model_kwargs,
        }

    @property
    @abc.abstractmethod
    def _llm_type(self) -> str: ...

    def _create_prediction(self, input_: dict[str, Any]) -> Prediction:
        # if it's an official model
        if ":" not in self.model:
            return self._client.models.predictions.create(self.model, input=input_)

        return self._client.predictions.create(version=self._version, input=input_)

    async def _async_create_prediction(self, input_: dict[str, Any]) -> Prediction:
        # if it's an official model
        if ":" not in self.model:
            return await self._client.models.predictions.async_create(self.model, input=input_)

        return await self._client.predictions.async_create(version=self._version, input=input_)

    def _stop_input(self, stop: list[str] | None) -> dict[str, Any]:
        if stop is None:
            return {}

        input_properties = self._input_properties
        if "stop" in input_properties:
            key = "stop"
        elif "stop_sequences" in input_properties:
            key = "stop_sequences"
        else:
            return {}

        value_schema: dict[str, Any] = input_properties[key]
        value_type = value_schema["type"]
        if value_type == "array":
            return {key: stop}
        if value_type == "string":
            return {key: ",".join(stop)}
        return {}

    def _stream_input(self, stream: bool) -> dict[str, Any]:
        if "stream" in self._input_properties:
            return {"stream": stream}
        return {}
