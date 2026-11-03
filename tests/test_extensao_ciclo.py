"""Testes da extensão por ciclo de verde: verde restante, distância restante e teto."""

import pytest

from crosswalk.config import SemaphoreConfig
from crosswalk.decision.extensao_tempo import (
    ControladorExtensaoCiclo,
    calcular_extensao_total_necessaria,
)


def _config(verde=20.0, teto=15.0):
    return SemaphoreConfig(tempo_verde_normal_s=verde, tempo_extra_maximo_s=teto)


class TestExtensaoTotalNecessaria:
    def test_exemplo_do_verde_restante(self):
        """Verde 20 s, 10 s já passaram, ele precisa de 20 s: fim em 30 s -> +10 s."""
        extra = calcular_extensao_total_necessaria(
            distancia_restante_m=10,
            velocidade_ms=0.5,
            tempo_normal_s=20,
            tempo_decorrido_s=10,
            tempo_extra_maximo_s=15,
        )
        assert extra == pytest.approx(10.0)

    def test_inicio_do_verde_equivale_a_formula_antiga(self):
        extra = calcular_extensao_total_necessaria(
            distancia_restante_m=10,
            velocidade_ms=0.5,
            tempo_normal_s=15,
            tempo_decorrido_s=0,
            tempo_extra_maximo_s=15,
        )
        assert extra == pytest.approx(5.0)

    def test_termina_dentro_do_verde_nao_estende(self):
        extra = calcular_extensao_total_necessaria(
            distancia_restante_m=3,
            velocidade_ms=0.7,
            tempo_normal_s=20,
            tempo_decorrido_s=10,
            tempo_extra_maximo_s=15,
        )
        assert extra == 0.0

    def test_limitado_ao_teto(self):
        extra = calcular_extensao_total_necessaria(
            distancia_restante_m=20,
            velocidade_ms=0.3,
            tempo_normal_s=20,
            tempo_decorrido_s=10,
            tempo_extra_maximo_s=15,
        )
        assert extra == 15.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(velocidade_ms=0),
            dict(distancia_restante_m=-1),
            dict(tempo_decorrido_s=-1),
        ],
    )
    def test_entradas_invalidas_levantam_erro(self, kwargs):
        base = dict(
            distancia_restante_m=5,
            velocidade_ms=0.7,
            tempo_normal_s=20,
            tempo_decorrido_s=0,
            tempo_extra_maximo_s=15,
        )
        base.update(kwargs)
        with pytest.raises(ValueError):
            calcular_extensao_total_necessaria(**base)


class TestControladorExtensaoCiclo:
    def test_sem_cadeirantes_nao_estende(self):
        c = ControladorExtensaoCiclo(_config())
        assert c.atualizar([], tempo_decorrido_s=5.0) == 0.0

    def test_cadeirante_que_entra_tarde_recebe_extensao(self):
        c = ControladorExtensaoCiclo(_config(verde=20))
        assert c.atualizar([(0.5, 10.0)], tempo_decorrido_s=10.0) == pytest.approx(10.0)

    def test_extensao_nunca_diminui_no_ciclo(self):
        c = ControladorExtensaoCiclo(_config(verde=20))
        c.atualizar([(0.5, 10.0)], tempo_decorrido_s=10.0)  # +10 s
        # Ele acelera / avança: a necessidade cai, mas a concedida permanece.
        assert c.atualizar([(1.0, 2.0)], tempo_decorrido_s=14.0) == pytest.approx(10.0)

    def test_multiplos_cadeirantes_prevalece_o_mais_lento(self):
        c = ControladorExtensaoCiclo(_config(verde=15))
        extra = c.atualizar([(0.5, 10.0), (1.5, 10.0)], tempo_decorrido_s=0.0)
        assert extra == pytest.approx(5.0)

    def test_teto_trava_e_outro_cadeirante_nao_estende_alem(self):
        c = ControladorExtensaoCiclo(_config(verde=20, teto=15))
        # Primeiro cadeirante muito lento: atinge o teto de 15 s.
        assert c.atualizar([(0.2, 10.0)], tempo_decorrido_s=0.0) == 15.0
        assert c.travado_no_teto is True
        # Outro cadeirante, ainda mais lento, entra depois: continua 15 s.
        assert c.atualizar([(0.1, 12.0)], tempo_decorrido_s=5.0) == 15.0

    def test_verde_encerrado_nao_aceita_nova_extensao(self):
        c = ControladorExtensaoCiclo(_config(verde=20))
        # Entra depois do fim do verde (20 s): nada a fazer.
        assert c.atualizar([(0.3, 10.0)], tempo_decorrido_s=21.0) == 0.0

    def test_fim_do_verde_considera_a_extensao_concedida(self):
        c = ControladorExtensaoCiclo(_config(verde=20))
        c.atualizar([(0.5, 10.0)], tempo_decorrido_s=10.0)  # concede 10 s
        assert c.fim_do_verde_s == pytest.approx(30.0)
        # Ainda dentro do verde estendido (25 s < 30 s): pode estender mais.
        assert c.atualizar([(0.2, 10.0)], tempo_decorrido_s=25.0) == 15.0

    def test_reiniciar_zera_o_ciclo(self):
        c = ControladorExtensaoCiclo(_config(verde=20, teto=15))
        c.atualizar([(0.2, 10.0)], tempo_decorrido_s=0.0)
        assert c.travado_no_teto is True
        c.reiniciar()
        assert c.extensao_concedida_s == 0.0
        assert c.travado_no_teto is False
        assert c.atualizar([], tempo_decorrido_s=1.0) == 0.0
