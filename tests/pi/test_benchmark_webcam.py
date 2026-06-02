"""Testes para pi/benchmark_webcam.py.

Cobre: cálculo de estatísticas de FPS, aquecimento do modelo, configuração.
"""

import pytest

from pi.benchmark_webcam import aquecer_modelo, calcular_estatisticas


class TestCalcularEstatisticas:
    """Cálculo de FPS médio, mínimo e máximo a partir dos tempos."""

    def test_tempos_uniformes(self):
        tempos = [0.2, 0.2, 0.2, 0.2]
        stats = calcular_estatisticas(tempos)
        assert stats["fps_medio"] == pytest.approx(5.0, abs=0.01)
        assert stats["fps_min"] == pytest.approx(5.0, abs=0.01)
        assert stats["fps_max"] == pytest.approx(5.0, abs=0.01)

    def test_tempos_variados(self):
        tempos = [0.1, 0.2, 0.3]  # 10, 5, 3.33 FPS
        stats = calcular_estatisticas(tempos)
        assert stats["fps_max"] == pytest.approx(10.0, abs=0.01)
        assert stats["fps_min"] == pytest.approx(3.33, abs=0.01)
        # Média ≈ 0.2s -> 5 FPS
        assert stats["fps_medio"] == pytest.approx(5.0, abs=0.1)

    def test_tempo_medio_em_milissegundos(self):
        tempos = [0.5]
        stats = calcular_estatisticas(tempos)
        assert stats["tempo_medio_ms"] == pytest.approx(500.0, abs=0.01)


class TestAquecerModelo:
    """Aquecimento do modelo YOLO antes do benchmark."""

    def test_chama_predict_n_vezes(self, cap_mock, yolo_mock):
        aquecer_modelo(yolo_mock, cap_mock, n_inferencias=5)
        assert yolo_mock.predict.call_count == 5

    def test_pula_quando_camera_falha(self, yolo_mock):
        from unittest.mock import MagicMock

        cap_falha = MagicMock()
        cap_falha.read.return_value = (False, None)
        aquecer_modelo(yolo_mock, cap_falha, n_inferencias=3)
        assert yolo_mock.predict.call_count == 0
