# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Annotated, Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.utils.pydantic import get_fields
from pydantic import ConfigDict, Field, model_validator
from replicate.exceptions import ModelError
from typing_extensions import override

from langchain_replicate._base import ReplicateBase, _adjust_prediction_input

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

    model_kwargs: Annotated[dict[str, Any], Field(default_factory=dict, validation_alias="input")]
    prompt_key: str | None = None

    streaming: bool = False
    """Whether to stream the results."""

    stop: Annotated[list[str], Field(default_factory=list)]
    """Stop sequences to early-terminate generation."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    @override
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @override
    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "replicate"]

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_fields(cls).keys()

        input = values.pop("input", {})
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

    @override
    @property
    def _llm_type(self) -> str:
        """Return type of model."""
        return "replicate"

    def _create_prediction_input(self, prompt: str, stream: bool, stop: list[str] | None, **kwargs: Any) -> dict[str, Any]:
        if self.prompt_key is None:
            self.prompt_key = next(iter(self._input_properties))

        input_ = self.model_kwargs.copy()
        input_.update(kwargs)
        input_[self.prompt_key] = prompt
        input_.update(self._stop_input(stop))
        input_.update(self._stream_input(stream))

        return _adjust_prediction_input(input_)

    def _apply_stop_sequences(self, text: str, stop_sequences: list[str] | None) -> str:
        """Apply stop sequences to text, truncating at the first occurrence.

        Args:
            text: The text to apply stop sequences to
            stop_sequences: List of stop sequences to check for

        Returns:
            The text truncated at the first stop sequence, or the original text if none found
        """
        if stop_sequences:
            stop_positions = [text.find(s) for s in stop_sequences if s and s in text]
            if stop_positions:
                return text[: min(stop_positions)]
        return text

    def _emit_chunk(
        self,
        text: str,
        run_manager: CallbackManagerForLLMRun | None,
    ) -> GenerationChunk:
        """Create and emit a generation chunk with callback notification."""
        chunk = GenerationChunk(text=text)
        if run_manager:
            run_manager.on_llm_new_token(
                text,
                chunk=chunk,
                verbose=self.verbose,
            )
        return chunk

    @override
    def _stream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        stop_sequences = stop or self.stop
        input_ = self._create_prediction_input(prompt, stream=True, stop=stop_sequences, **kwargs)
        prediction = self._create_prediction(input_)

        # If stop sequences are present and not handled by prediction, stream incrementally
        if stop_sequences and not self._stop_input(stop_sequences):
            max_stop_length = max((len(sequence) for sequence in stop_sequences), default=0)
            buffered_output = ""

            for output in prediction.output_iterator():
                if not output:
                    continue

                buffered_output += output
                truncated_output = self._apply_stop_sequences(buffered_output, stop_sequences)

                if len(truncated_output) < len(buffered_output):
                    buffered_output = truncated_output
                    prediction.cancel()
                    break

                # Keep buffer size of max_stop_length to handle overlapping sequences
                # This ensures we can detect stop sequences that span chunk boundaries
                safe_output_length = len(buffered_output) - max_stop_length
                if safe_output_length > 0:
                    safe_output = buffered_output[:safe_output_length]
                    yield self._emit_chunk(safe_output, run_manager)
                    buffered_output = buffered_output[safe_output_length:]

            if buffered_output:
                yield self._emit_chunk(buffered_output, run_manager)
        else:
            # No stop sequences or handled by prediction, stream normally
            for output in prediction.output_iterator():
                if output:
                    yield self._emit_chunk(output, run_manager)

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        stop_sequences = stop or self.stop
        completion: str
        if self.streaming:
            completion = "".join(chunk.text for chunk in self._stream(prompt, stop=stop_sequences, run_manager=run_manager, **kwargs))
            return completion

        input_ = self._create_prediction_input(prompt, stream=False, stop=stop_sequences, **kwargs)
        prediction = self._create_prediction(input_)
        prediction.wait()
        if prediction.status == "failed":
            raise ModelError(prediction)
        completion = "".join(prediction.output) if isinstance(prediction.output, Iterable) else str(prediction.output)
        return self._apply_stop_sequences(completion, stop_sequences)

    async def _async_emit_chunk(
        self,
        text: str,
        run_manager: AsyncCallbackManagerForLLMRun | None,
    ) -> GenerationChunk:
        """Create and emit a generation chunk with async callback notification."""
        chunk = GenerationChunk(text=text)
        if run_manager:
            await run_manager.on_llm_new_token(
                text,
                chunk=chunk,
                verbose=self.verbose,
            )
        return chunk

    @override
    async def _astream(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        stop_sequences = stop or self.stop
        input_ = self._create_prediction_input(prompt, stream=True, stop=stop_sequences, **kwargs)
        prediction = await self._async_create_prediction(input_)

        # If stop sequences are present and not handled by prediction, stream incrementally
        if stop_sequences and not self._stop_input(stop_sequences):
            max_stop_length = max((len(sequence) for sequence in stop_sequences), default=0)
            buffered_output = ""

            async for output in prediction.async_output_iterator():
                if not output:
                    continue

                buffered_output += output
                truncated_output = self._apply_stop_sequences(buffered_output, stop_sequences)

                if len(truncated_output) < len(buffered_output):
                    buffered_output = truncated_output
                    await prediction.async_cancel()
                    break

                # Keep buffer size of max_stop_length to handle overlapping sequences
                # This ensures we can detect stop sequences that span chunk boundaries
                safe_output_length = len(buffered_output) - max_stop_length
                if safe_output_length > 0:
                    safe_output = buffered_output[:safe_output_length]
                    yield await self._async_emit_chunk(safe_output, run_manager)
                    buffered_output = buffered_output[safe_output_length:]

            if buffered_output:
                yield await self._async_emit_chunk(buffered_output, run_manager)
        else:
            # No stop sequences or handled by prediction, stream normally
            async for output in prediction.async_output_iterator():
                if output:
                    yield await self._async_emit_chunk(output, run_manager)

    @override
    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        stop_sequences = stop or self.stop
        completion: str
        if self.streaming:
            completion = "".join([chunk.text async for chunk in self._astream(prompt, stop=stop_sequences, run_manager=run_manager, **kwargs)])
            return completion

        input_ = self._create_prediction_input(prompt, stream=False, stop=stop_sequences, **kwargs)
        prediction = await self._async_create_prediction(input_)
        await prediction.async_wait()
        if prediction.status == "failed":
            raise ModelError(prediction)
        completion = "".join(prediction.output) if isinstance(prediction.output, Iterable) else str(prediction.output)
        return self._apply_stop_sequences(completion, stop_sequences)
