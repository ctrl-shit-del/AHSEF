import pytest

from src.preprocessing.sampling.allocation import (
    Cursor,
    allocate_fraction,
    derive_seed,
    deterministic_shuffle,
    largest_remainder,
)


def test_largest_remainder_is_exact_and_deterministic():
    assert largest_remainder(10, [0.8, 0.1, 0.1]) == [8, 1, 1]
    assert sum(largest_remainder(25, [0.8, 0.1, 0.1])) == 25
    assert largest_remainder(0, [0.8, 0.1, 0.1]) == [0, 0, 0]
    # Ties resolve by ascending index, never by hash order.
    assert largest_remainder(25, [0.8, 0.1, 0.1]) == largest_remainder(25, [0.8, 0.1, 0.1])
    assert largest_remainder(25, [0.8, 0.1, 0.1]) == [20, 3, 2]


def test_allocate_fraction_hits_the_target_and_reports_floors():
    sizes = {("a", 0): 1000, ("b", 0): 100, ("c", 0): 3}
    allocation = allocate_fraction(sizes, 0.25, minimum_per_group=1)
    assert allocation.target_total == round(1103 * 0.25)
    assert allocation.counts[("a", 0)] == 250
    assert allocation.counts[("b", 0)] == 25
    assert allocation.counts[("c", 0)] >= 1
    assert allocation.selected_total >= allocation.target_total

    tiny = allocate_fraction({("x",): 3}, 0.1, minimum_per_group=1)
    assert tiny.counts[("x",)] == 1
    assert tiny.minimum_adjustments and tiny.minimum_adjustments[0]["enforced_minimum"] == 1


def test_allocate_fraction_never_exceeds_group_size():
    allocation = allocate_fraction({("a",): 4, ("b",): 1}, 1.0, minimum_per_group=1)
    assert allocation.counts == {("a",): 4, ("b",): 1}


def test_allocate_fraction_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        allocate_fraction({("a",): 10}, 0.0)
    with pytest.raises(ValueError):
        allocate_fraction({("a",): 10}, 1.5)


def test_shuffle_is_stable_for_the_same_seed_and_key():
    items = list(range(50))
    first = deterministic_shuffle(items, 42, "image", ("AffectNet+", 1), "free")
    second = deterministic_shuffle(items, 42, "image", ("AffectNet+", 1), "free")
    other_seed = deterministic_shuffle(items, 7, "image", ("AffectNet+", 1), "free")
    other_key = deterministic_shuffle(items, 42, "image", ("AffectNet+", 2), "free")

    assert first == second
    assert sorted(first) == items
    assert first != other_seed
    assert first != other_key
    assert derive_seed(42, "a") == derive_seed(42, "a")
    assert derive_seed(42, "a") != derive_seed(42, "b")


def test_cursor_consumes_without_repeating():
    cursor = Cursor([1, 2, 3, 4])
    assert cursor.take(2) == [1, 2]
    assert cursor.take(0) == []
    assert cursor.take(10) == [3, 4]
    assert cursor.take(1) == []
    assert cursor.remaining == 0
