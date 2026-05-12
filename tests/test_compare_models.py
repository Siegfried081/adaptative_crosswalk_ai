from scripts.compare_models import compare


def test_challenger_approved_when_better():
    champion = {"map50": 0.80}
    challenger = {"map50": 0.85}
    assert compare(challenger, champion) is True


def test_challenger_rejected_when_worse():
    champion = {"map50": 0.85}
    challenger = {"map50": 0.80}
    assert compare(challenger, champion) is False


def test_equal_scores_approved_with_zero_threshold():
    champion = {"map50": 0.80}
    challenger = {"map50": 0.80}
    assert compare(challenger, champion, min_improvement=0.0) is True


def test_min_improvement_threshold_not_met():
    champion = {"map50": 0.80}
    challenger = {"map50": 0.81}
    assert compare(challenger, champion, min_improvement=0.02) is False


def test_min_improvement_threshold_met():
    champion = {"map50": 0.80}
    challenger = {"map50": 0.82}
    assert compare(challenger, champion, min_improvement=0.02) is True
