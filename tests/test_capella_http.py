"""Framework-free tests for ``shared/capella_http.py`` (needs only httpx + pytest)."""

import pytest

from shared.capella_http import needs_input_type, truncate_to_token_budget, v1_base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "https://capella.test",
        "https://capella.test/",
        "https://capella.test/v1",
        "https://capella.test/v1/",
    ],
)
def test_v1_base_url_appends_v1_exactly_once(base_url):
    assert v1_base_url(base_url) == "https://capella.test/v1"


@pytest.mark.parametrize(
    "model, expected",
    [
        ("nvidia/llama-3.2-nv-embedqa-1b-v2", True),
        ("nvidia/NV-EmbedQA-E5-v5", True),
        ("intfloat/e5-mistral-7b-instruct", False),
    ],
)
def test_needs_input_type(model, expected):
    assert needs_input_type(model) is expected


def test_truncate_keeps_text_within_budget():
    assert truncate_to_token_budget("x" * 30, max_tokens=10) == "x" * 30


def test_truncate_cuts_to_80_percent_of_estimated_chars():
    assert truncate_to_token_budget("x" * 31, max_tokens=10) == "x" * 24
