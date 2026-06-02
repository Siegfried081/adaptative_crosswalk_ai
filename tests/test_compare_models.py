from scripts.compare_models import compare


def test_challenger_approved_when_better():
    """Challenger com mAP melhor e mesmo recall deve ser aprovado."""
    champion = {"map50": 0.80, "recall": 0.85}
    challenger = {"map50": 0.85, "recall": 0.85}
    assert compare(challenger, champion) is True


def test_challenger_rejected_when_worse():
    """Challenger com mAP pior deve ser rejeitado."""
    champion = {"map50": 0.85, "recall": 0.85}
    challenger = {"map50": 0.80, "recall": 0.85}
    assert compare(challenger, champion) is False


def test_equal_scores_approved_with_zero_threshold():
    """Métricas iguais devem ser aprovadas quando o threshold é zero."""
    champion = {"map50": 0.80, "recall": 0.85}
    challenger = {"map50": 0.80, "recall": 0.85}
    assert compare(challenger, champion, min_improvement=0.0) is True


def test_min_improvement_threshold_not_met():
    """Melhora de mAP abaixo do threshold mínimo deve ser rejeitada."""
    champion = {"map50": 0.80, "recall": 0.85}
    challenger = {"map50": 0.81, "recall": 0.85}
    assert compare(challenger, champion, min_improvement=0.02) is False


def test_min_improvement_threshold_met():
    """Melhora de mAP igual ao threshold mínimo deve ser aprovada."""
    champion = {"map50": 0.80, "recall": 0.85}
    challenger = {"map50": 0.82, "recall": 0.85}
    assert compare(challenger, champion, min_improvement=0.02) is True


def test_recall_regression_rejected():
    """Challenger com mAP melhor mas queda de recall acima do tolerável deve ser rejeitado."""
    champion = {"map50": 0.80, "recall": 0.85}
    challenger = {
        "map50": 0.85,
        "recall": 0.82,
    }  # recall caiu 3pp (acima do tolerance default 1pp)
    assert compare(challenger, champion) is False


def test_recall_within_tolerance_approved():
    """Queda de recall dentro da tolerância (1pp) deve ser aceita."""
    champion = {"map50": 0.80, "recall": 0.85}
    challenger = {
        "map50": 0.85,
        "recall": 0.845,
    }  # queda de 0.5pp, abaixo do limite 1pp
    assert compare(challenger, champion) is True
