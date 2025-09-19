# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils.pydantic import get_fields
from pydantic import ConfigDict, Field, model_validator
from replicate.exceptions import ModelError
from typing_extensions import override

from langchain_replicate._base import ReplicateBase

logger = logging.getLogger(__name__)


class Replicate(ReplicateBase, LLM):
    """Replicate completion models.

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
                    "stability-ai/stable-diffusion:"
                    "27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
                ),
                model_kwargs={"image_dimensions": "512x512"}
            )
    """

    model_kwargs: dict[str, Any] = Field(default_factory=dict, validation_alias="input")
    prompt_key: str | None = None

    streaming: bool = False
    """Whether to stream the results."""

    stop: list[str] = Field(default_factory=list)
    """Stop sequences to early-terminate generation."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

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
    def _llm_type(self) -> str:
        """Return type of model."""
        return "replicate"

    def _create_prediction_input(self, prompt: str, stream: bool, stop: list[str] | None, **kwargs: Any) -> dict[str, Any]:
        if self.prompt_key is None:
            self.prompt_key = next(iter(self._input_properties))

        input_: dict[str, Any] = self.model_kwargs | kwargs | {self.prompt_key: prompt} | self._stop_input(stop) | self._stream_input(stream)
        return input_

    @override
    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        input_ = self._create_prediction_input(prompt, stream=True, stop=stop, **kwargs)
        prediction = self._create_prediction(input_)

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

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        completion: str
        if self.streaming:
            completion = "".join(chunk.text for chunk in self._stream(prompt, stop=stop, run_manager=run_manager, **kwargs))
            return completion

        input_ = self._create_prediction_input(prompt, stream=False, stop=stop, **kwargs)
        prediction = self._create_prediction(input_)
        prediction.wait()
        if prediction.status == "failed":
            raise ModelError(prediction)
        completion = "".join(prediction.output) if isinstance(prediction.output, Iterable) else str(prediction.output)
        stop_conditions = stop or self.stop
        for s in stop_conditions:
            if s in completion:
                completion = completion[: completion.find(s)]
        return completion

    @override
    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        input_ = self._create_prediction_input(prompt, stream=True, stop=stop, **kwargs)
        prediction = await self._async_create_prediction(input_)

        stop_conditions = stop or self.stop
        stop_condition_reached = False
        current_completion: str = ""
        async for output in prediction.async_output_iterator():
            current_completion += output
            # test for stop conditions, if specified
            for s in stop_conditions:
                if s in current_completion:
                    await prediction.async_cancel()
                    stop_condition_reached = True
                    # Potentially some tokens that should still be yielded before ending
                    # stream.
                    stop_index = max(output.find(s), 0)
                    output = output[:stop_index]
                    if not output:
                        break
            if output:
                if run_manager:
                    await run_manager.on_llm_new_token(
                        output,
                        verbose=self.verbose,
                    )
                yield GenerationChunk(text=output)
            if stop_condition_reached:
                break

    @override
    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        completion: str
        if self.streaming:
            completion = "".join([chunk.text async for chunk in self._astream(prompt, stop=stop, run_manager=run_manager, **kwargs)])
            return completion

        input_ = self._create_prediction_input(prompt, stream=False, stop=stop, **kwargs)
        prediction = await self._async_create_prediction(input_)
        await prediction.async_wait()
        if prediction.status == "failed":
            raise ModelError(prediction)
        completion = "".join(prediction.output) if isinstance(prediction.output, Iterable) else str(prediction.output)
        stop_conditions = stop or self.stop
        for s in stop_conditions:
            if s in completion:
                completion = completion[: completion.find(s)]
        return completion
