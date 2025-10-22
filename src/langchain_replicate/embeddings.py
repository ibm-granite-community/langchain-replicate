# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict, Field
from replicate.exceptions import ModelError
from typing_extensions import override

from langchain_replicate._base import ReplicateBase


class ReplicateEmbeddings(ReplicateBase, Embeddings):
    """Replicate embedding models.

    To use, you should have the ``replicate`` python package installed,
    and the environment variable ``REPLICATE_API_TOKEN`` set with your API token.
    You can find your token here: https://replicate.com/account

    The model param is required, but any other model parameters can also
    be passed in with the format model_kwargs={model_param: value, ...}

    Example:
        .. code-block:: python

            from langchain_replicate import ReplicateEmbeddings

            replicate = ReplicateEmbeddings(
                model="ibm-granite/granite-embedding-278m-multilingual",
            )
    """

    texts_key: str | None = None
    texts_value_mapping: Callable[[Sequence[str]], Any] | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    """Can be used to map the input list of strings for the embeddings to some
        type other than List[str]. For example, if the model requires the strings as a JSON
        formatted string, this field can be set to `json.dumps`. If the model requires newline
        separated strings, this field can be set to `"\\n".join`.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    @property
    def _llm_type(self) -> str:
        """Return the type of embeddings model."""
        return "replicate-embeddings"

    def _create_prediction_input(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
        if self.texts_key is None:
            self.texts_key = next(iter(self._input_properties))

        texts_value = self.texts_value_mapping(texts) if callable(self.texts_value_mapping) else texts
        input_: dict[str, Any] = {self.texts_key: texts_value} | self.model_kwargs | kwargs

        return input_

    @override
    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Compute doc embeddings using a Replicate embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        input_ = self._create_prediction_input(texts, **kwargs)
        prediction = self._create_prediction(input_)
        prediction.wait()
        if prediction.status == "failed":
            raise ModelError(prediction)
        completion = prediction.output
        assert isinstance(completion, list)
        return completion

    @override
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Compute query embeddings using a Replicate embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = self.embed_documents([text], **kwargs)
        return embeddings[0]

    @override
    async def aembed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        input_ = self._create_prediction_input(texts, **kwargs)
        prediction = await self._async_create_prediction(input_)
        await prediction.async_wait()
        if prediction.status == "failed":
            raise ModelError(prediction)
        completion = prediction.output
        assert isinstance(completion, list)
        return completion

    @override
    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        embeddings = await self.aembed_documents([text], **kwargs)
        return embeddings[0]
