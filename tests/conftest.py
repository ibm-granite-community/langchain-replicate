# SPDX-License-Identifier: MIT

import os

import pytest
from assertpy import assert_that


@pytest.fixture
def replicate_api_token(monkeypatch) -> str:
    """Return the api token from the env. We also remove it from the env."""
    api_token = os.getenv("REPLICATE_API_TOKEN")
    assert_that(api_token).is_not_none().is_not_empty()
    monkeypatch.delenv("REPLICATE_API_TOKEN")
    return api_token  # type: ignore
