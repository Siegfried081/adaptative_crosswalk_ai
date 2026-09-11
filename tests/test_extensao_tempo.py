"""Testes unitários do modelo matemático de extensão de tempo.

Cobre os casos-limite discutidos na fundamentação: cadeirante rápido (sem
extensão), cadeirante lento (teto de segurança), múltiplos cadeirantes
(prevalece o mais lento), e o comportamento do estimador de velocidade
(fallback vs medição, suavização, plausibilidade).
"""

import pytest

from crosswalk.config import SemaphoreConfig
from crosswalk.decision.extensao_tempo import (
    EstimadorVelocidade,
    calcular_tempo_extra,
    clamp,
    decidir_extensao,
)


class TestClamp:
    def test_dentro_do_intervalo(self):
        assert clamp(5, 0, 10) == 5

    def test_abaixo_do_minimo(self):
        assert clamp(-3, 0, 10) == 0

    def test_acima_do_maximo(self):
        assert clamp(15, 0, 10) == 10


class TestCalcularTempoExtra:
    def test_cadeirante_lento_recebe_tempo_extra(self):
        extra = calcular_tempo_extra(
            distancia_m=10,
            velocidade_ms=0.5,
            tempo_normal_s=15,
            tempo_extra_maximo_s=15,
        )
        assert extra == pytest.approx(5.0, abs=0.01)

    def test_cadeirante_rapido_nao_recebe_extra(self):
        extra = calcular_tempo_extra(
            distancia_m=10,
            velocidade_ms=1.5,
            tempo_normal_s=15,
            tempo_extra_maximo_s=15,
        )
        assert extra == 0.0

    def test_teto_de_seguranca_respeitado(self):
        extra = calcular_tempo_extra(
            distancia_m=20,
            velocidade_ms=0.3,
            tempo_normal_s=15,
            tempo_extra_maximo_s=15,
        )
        assert extra == 15.0

    def test_velocidade_zero_lanca_erro(self):
        with pytest.raises(ValueError, match="velocidade_ms deve ser positiva"):
            calcular_tempo_extra(
                distancia_m=10,
                velocidade_ms=0,
                tempo_normal_s=15,
                tempo_extra_maximo_s=15,
            )

    def test_velocidade_negativa_lanca_erro(self):
        with pytest.raises(ValueError, match="velocidade_ms deve ser positiva"):
            calcular_tempo_extra(
                distancia_m=10,
                velocidade_ms=-0.5,
                tempo_normal_s=15,
                tempo_extra_maximo_s=15,
            )

    def test_distancia_negativa_lanca_erro(self):
        with pytest.raises(ValueError, match="distancia_m não pode ser negativa"):
            calcular_tempo_extra(
                distancia_m=-1,
                velocidade_ms=0.7,
                tempo_normal_s=15,
                tempo_extra_maximo_s=15,
            )


class TestDecidirExtensao:
    def _config(self):
        return SemaphoreConfig(tempo_verde_normal_s=15.0, tempo_extra_maximo_s=15.0)

    def test_sem_cadeirantes_retorna_zero(self):
        assert decidir_extensao([], self._config(), distancia_m=10) == 0.0

    def test_um_cadeirante(self):
        extra = decidir_extensao([0.5], self._config(), distancia_m=10)
        assert extra == pytest.approx(5.0, abs=0.01)

    def test_multiplos_cadeirantes_prevalece_o_mais_lento(self):
        extra = decidir_extensao([0.5, 1.5], self._config(), distancia_m=10)
        assert extra == pytest.approx(5.0, abs=0.01)


class TestEstimadorVelocidade:
    def test_sem_amostras_suficientes_usa_fallback(self):
        est = EstimadorVelocidade(velocidade_fallback_ms=0.7)
        est.atualizar((0.0, 0.0), tempo_s=0.0)
        est.atualizar((0.1, 0.0), tempo_s=0.1)

        velocidade, usou_medicao = est.velocidade_atual_ms()
        assert usou_medicao is False
        assert velocidade == 0.7

    def test_amostras_suficientes_e_plausivel_usa_medicao(self):
        est = EstimadorVelocidade(velocidade_fallback_ms=0.7)
        for i in range(8):
            est.atualizar((0.08 * i, 0.0), tempo_s=0.1 * i)

        velocidade, usou_medicao = est.velocidade_atual_ms()
        assert usou_medicao is True
        assert velocidade == pytest.approx(0.8, abs=0.05)

    def test_velocidade_implausivel_cai_no_fallback(self):
        est = EstimadorVelocidade(velocidade_fallback_ms=0.7)
        for i in range(8):
            est.atualizar((5.0 * i, 0.0), tempo_s=0.1 * i)

        velocidade, usou_medicao = est.velocidade_atual_ms()
        assert usou_medicao is False
        assert velocidade == 0.7

    def test_parado_nao_gera_velocidade_zero_valida(self):
        est = EstimadorVelocidade(velocidade_fallback_ms=0.7)
        for i in range(8):
            est.atualizar((0.001 * i, 0.0), tempo_s=0.1 * i)

        velocidade, usou_medicao = est.velocidade_atual_ms()
        assert usou_medicao is False
        assert velocidade == 0.7

    def test_intervalo_de_tempo_variavel_ainda_calcula_corretamente(self):
        est = EstimadorVelocidade(velocidade_fallback_ms=0.7)
        tempos = [0.0, 0.12, 0.19, 0.35, 0.41, 0.58]
        for t in tempos:
            est.atualizar((0.8 * t, 0.0), tempo_s=t)

        velocidade, usou_medicao = est.velocidade_atual_ms()
        assert usou_medicao is True
        assert velocidade == pytest.approx(0.8, abs=0.05)
