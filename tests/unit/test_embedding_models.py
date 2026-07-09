"""Tests for provider-independent embedding models."""

import pytest

from enterprise_ai_assistant.models import EmbeddingResponse


def test_embedding_response_reports_dimension() -> None:
    """A rectangular embedding response should expose its common dimension."""

    response = EmbeddingResponse(
        vectors=((0.1, 0.2), (0.3, 0.4)),
        model="test",
    )

    assert response.dimension == 2


@pytest.mark.parametrize(
    "vectors",
    [
        (),
        ((),),
        ((0.1,), (0.2, 0.3)),
    ],
)
def test_embedding_response_rejects_invalid_matrix(
    vectors: tuple[tuple[float, ...], ...],
) -> None:
    """Empty or ragged vectors should fail before reaching a vector store."""

    with pytest.raises(ValueError):
        EmbeddingResponse(vectors=vectors, model="test")
