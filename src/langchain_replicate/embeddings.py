# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import ConfigDict, Field
from replicate.prediction import Prediction

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

    def _create_prediction(self, texts: list[str], **kwargs: Any) -> Prediction:
        if self.texts_key is None:
            self.texts_key = self._input_properties[0][0]

        input_: dict[str, Any] = {
            self.texts_key: self.texts_value_mapping(texts) if self.texts_value_mapping else texts,
            **self.model_kwargs,
            **kwargs,
        }

        # if it's an official model
        if ":" not in self.model:
            return self._client.models.predictions.create(self.model, input=input_)

        return self._client.predictions.create(
            version=self._version,
            input=input_,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a Replicate embeddings model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        prediction = self._create_prediction(texts)
        prediction.wait()
        if prediction.status == "failed":
            raise RuntimeError(prediction.error)
        completion = prediction.output
        assert isinstance(completion, list)
        return completion

    def embed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a Replicate embeddings model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
