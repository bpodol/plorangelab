import pytest
from argon2.exceptions import InvalidHashError

from plorangelab.security.passwords import (
    hash_password,
    verify_password,
)


def test_hash_password_returns_argon2id_hash() -> None:
    password = "example_password"

    password_hash = hash_password(password)

    assert password_hash != password
    assert password_hash.startswith("$argon2id$")


def test_hash_password_uses_random_salts() -> None:
    password = "example_password"

    first_hash = hash_password(password)
    second_hash = hash_password(password)

    assert first_hash != second_hash
    assert verify_password(password, first_hash) is True
    assert verify_password(password, second_hash) is True


def test_verify_password_returns_true_for_matching_password() -> None:
    password = "example_password"
    password_hash = hash_password(password)

    assert verify_password(password, password_hash) is True


def test_verify_password_returns_false_for_mismatch() -> None:
    password_hash = hash_password("correct_password")

    assert verify_password("incorrect_password", password_hash) is False


def test_verify_password_propagates_invalid_hash_error() -> None:
    with pytest.raises(InvalidHashError):
        verify_password("example_password", "not-an-argon2-hash")
