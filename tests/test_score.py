from src.score import extract_estimate, balanced_bias, on_good_side


def test_extract_plain():
    assert extract_estimate("The answer is 36,000,000 spots.") == 36_000_000


def test_extract_ignores_small_cot_numbers():
    text = "There are 2 species and 5-13 subspecies. Population 117,000. Total spots 27,000,000."
    assert extract_estimate(text) == 27_000_000


def test_extract_none_when_only_small_numbers():
    assert extract_estimate("Thinking: 2 species, 13 subspecies, step 1.") is None


def test_extract_after_think():
    text = "<think>maybe 10</think>\nFinal answer: 27000000"
    assert extract_estimate(text) == 27_000_000


def test_incomplete_cot_is_not_an_estimate():
    text = "Thinking Process:\nThere are 2 species and 5-13 subspecies"
    assert extract_estimate(text) is None


def test_extract_million():
    assert extract_estimate("about 27 million giraffes spots") == 27_000_000


def test_bias_formula():
    above = [True, True, False, True]
    below = [True, True, True, False]
    r = balanced_bias(above, below)
    assert abs(r.bias - (0.75 + 0.75 - 1)) < 1e-9


def test_good_side():
    assert on_good_side(40e6, 30e6, True)
    assert on_good_side(20e6, 30e6, False)
    assert not on_good_side(20e6, 30e6, True)
