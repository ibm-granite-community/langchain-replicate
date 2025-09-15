# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field
from replicate import default_client
from replicate.client import Client
from replicate.prediction import Prediction
from replicate.version import Version


class ReplicateEmbeddings(BaseModel, Embeddings):
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

    model: str
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    replicate_api_token: str | None = None
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
    version_obj: Version | None = Field(default=None, exclude=True)
    """Optionally pass in the model version object during initialization to avoid
        having to make an extra API call to retrieve it during streaming. NOTE: not
        serializable, is excluded from serialization.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"replicate_api_token": "REPLICATE_API_TOKEN"}

    @cached_property
    def _client(self) -> Client:
        return Client(api_token=self.replicate_api_token) if self.replicate_api_token else default_client

    def _create_prediction(self, texts: list[str], **kwargs: Any) -> Prediction:
        # get the model and version
        if self.version_obj is None:
            if ":" in self.model:
                model_str, version_str = self.model.split(":")
                model = self._client.models.get(model_str)
                self.version_obj = model.versions.get(version_str)
            else:
                model = self._client.models.get(self.model)
                self.version_obj = model.latest_version

        if self.texts_key is None:
            # sort through the openapi schema to get the name of the first input
            input_properties = sorted(
                self.version_obj.openapi_schema["components"]["schemas"]["Input"][  # type: ignore
                    "properties"
                ].items(),
                key=lambda item: item[1].get("x-order", 0),
            )
            self.texts_key = input_properties[0][0]

        input_: dict = {
            self.texts_key: self.texts_value_mapping(texts) if self.texts_value_mapping else texts,
            **self.model_kwargs,
            **kwargs,
        }

        # if it's an official model
        if ":" not in self.model:
            return self._client.models.predictions.create(self.model, input=input_)

        return self._client.predictions.create(
            version=self.version_obj,  # type: ignore
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
