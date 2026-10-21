import numpy as np
import pytest

from plorangelab.ranges import encode_range


def test_encode_range_parses_optional_weights() -> None:
    hands, weights = encode_range("AsKsQhJhTh@100,AdKdQcJcTc@43,2c3c4c5c6c", 5)

    assert hands.shape == (3, 5)
    assert hands.dtype == np.uint8
    assert hands.tolist() == [
        [51, 47, 42, 38, 34],
        [49, 45, 40, 36, 32],
        [16, 12, 8, 4, 0],
    ]
    assert weights.shape == (3,)
    assert weights.dtype == np.float32
    np.testing.assert_allclose(weights, [1.0, 0.43, 1.0])


@pytest.mark.parametrize(
    "range_string",
    [
        "",
        ",AsKsQhJh",
        "AsKsQhJh@",
        "AsKsQhJh@abc",
        "AsKsQhJh@1@2",
        "AsKsQhJh@-1",
        "AsKsQhJh@nan",
    ],
)
def test_encode_range_rejects_invalid_entries(range_string: str) -> None:
    with pytest.raises(ValueError):
        encode_range(range_string, 4)
