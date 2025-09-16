# SPDX-License-Identifier: MIT

from __future__ import annotations

import abc
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field
from replicate.client import Client
from replicate.version import Version


class ReplicateBase(BaseModel, abc.ABC):
    """Base model for Replicate integrations"""

    model: str
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    replicate_api_token: str | None = None
    version_obj: Version | None = Field(default=None, exclude=True)
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

    @property
    def _input_properties(self) -> list[tuple[str, Any]]:
        """Sort the openapi schema Inputs in x-order"""
        input_properties = sorted(
            self._version.openapi_schema["components"]["schemas"]["Input"]["properties"].items(),
            key=lambda item: item[1].get("x-order", 0),
        )
        return input_properties
