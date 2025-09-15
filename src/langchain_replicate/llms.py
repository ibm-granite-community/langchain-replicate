# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils.pydantic import get_fields
from pydantic import ConfigDict, Field, model_validator
from replicate import default_client
from replicate.client import Client
from replicate.prediction import Prediction
from replicate.version import Version

logger = logging.getLogger(__name__)


class Replicate(LLM):
    """Replicate models.

    To use, you should have the ``replicate`` python package installed,
    and the environment variable ``REPLICATE_API_TOKEN`` set with your API token.
    You can find your token here: https://replicate.com/account

    The model param is required, but any other model parameters can also
    be passed in with the format model_kwargs={model_param: value, ...}

    Example:
        .. code-block:: python

            from langchain_replicate import Replicate

            replicate = Replicate(
                model=(
                    "stability-ai/stable-diffusion: "
                    "27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
                ),
                model_kwargs={"image_dimensions": "512x512"}
            )
    """

    model: str
    model_kwargs: dict[str, Any] = Field(default_factory=dict, validation_alias="input")
    replicate_api_token: str | None = None
    prompt_key: str | None = None
    version_obj: Version | None = Field(default=None, exclude=True)
    """Optionally pass in the model version object during initialization to avoid
        having to make an extra API call to retrieve it during streaming. NOTE: not
        serializable, is excluded from serialization.
    """

    streaming: bool = False
    """Whether to stream the results."""

    stop: list[str] = Field(default_factory=list)
    """Stop sequences to early-terminate generation."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"replicate_api_token": "REPLICATE_API_TOKEN"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "replicate"]

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_fields(cls).keys()

        input = values.pop("input", {})  # pylint: disable=redefined-builtin
        if input:
            logger.warning("Init param `input` is deprecated, please use `model_kwargs` instead.")
        extra = {**values.pop("model_kwargs", {}), **input}
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    """%s was transferred to model_kwargs.
                    Please confirm that %s is what you intended.""",
                    field_name,
                    field_name,
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "replicate"

    @cached_property
    def _client(self) -> Client:
        return Client(api_token=self.replicate_api_token) if self.replicate_api_token else default_client

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Call to replicate endpoint."""
        completion: str | None = None
        if self.streaming:
            for chunk in self._stream(prompt, stop=stop, run_manager=run_manager, **kwargs):
                if completion is None:
                    completion = chunk.text
                else:
                    completion += chunk.text
        else:
            prediction = self._create_prediction(prompt, **kwargs)
            prediction.wait()
            if prediction.status == "failed":
                raise RuntimeError(prediction.error)
            completion = "".join(prediction.output) if isinstance(prediction.output, Iterable) else str(prediction.output)
        assert completion is not None
        stop_conditions = stop or self.stop
        for s in stop_conditions:
            if s in completion:
                completion = completion[: completion.find(s)]
        return completion

    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        prediction = self._create_prediction(prompt, **kwargs)
        stop_conditions = stop or self.stop
        stop_condition_reached = False
        current_completion: str = ""
        for output in prediction.output_iterator():
            current_completion += output
            # test for stop conditions, if specified
            for s in stop_conditions:
                if s in current_completion:
                    prediction.cancel()
                    stop_condition_reached = True
                    # Potentially some tokens that should still be yielded before ending
                    # stream.
                    stop_index = max(output.find(s), 0)
                    output = output[:stop_index]
                    if not output:
                        break
            if output:
                if run_manager:
                    run_manager.on_llm_new_token(
                        output,
                        verbose=self.verbose,
                    )
                yield GenerationChunk(text=output)
            if stop_condition_reached:
                break

    def _create_prediction(self, prompt: str, **kwargs: Any) -> Prediction:
        # get the model and version
        if self.version_obj is None:
            if ":" in self.model:
                model_str, version_str = self.model.split(":")
                model = self._client.models.get(model_str)
                self.version_obj = model.versions.get(version_str)
            else:
                model = self._client.models.get(self.model)
                self.version_obj = model.latest_version

        if self.prompt_key is None:
            # sort through the openapi schema to get the name of the first input
            input_properties = sorted(
                self.version_obj.openapi_schema["components"]["schemas"]["Input"]["properties"].items(),  # type: ignore
                key=lambda item: item[1].get("x-order", 0),
            )

            self.prompt_key = input_properties[0][0]

        input_: dict = {
            self.prompt_key: prompt,
            **self.model_kwargs,
            **kwargs,
        }

        # if it's an official model
        if ":" not in self.model:
            return self._client.models.predictions.create(self.model, input=input_)

        return self._client.predictions.create(version=self.version_obj, input=input_)  # type: ignore
