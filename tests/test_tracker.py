"""Testes do MultiObjectTracker: integração entre homografia e estimativa de velocidade.

Usa uma homografia sintética conhecida (faixa de 4m x 12m) e simula a
trajetória de um "cadeirante" com velocidade real conhecida, projetando-a
para pixels via a homografia inversa — o mesmo procedimento usado para
validar manualmente o tracker antes de existir este teste automatizado.

É essa ausência de teste automatizado que permitiu, na prática, que uma
versão desatualizada do tracker.py (usando frame_idx/fps) ficasse fora de
sincronia com a versão nova de EstimadorVelocidade (baseada em tempo_s) sem
ninguém perceber até rodar os testes manualmente. Este arquivo existe para
que essa classe de erro seja pega automaticamente pelo CI dali em diante.
"""

import cv2
import numpy as np
import pytest

from crosswalk.geometry.homography import calcular_homografia
from crosswalk.tracking.tracker import MultiObjectTracker, extrair_deteccoes_ultralytics

CANTOS_PX = [[400, 700], [1200, 700], [1400, 950], [200, 950]]
CANTOS_MUNDO_M = [[0, 0], [4, 0], [4, 12], [0, 12]]  # faixa de 4m x 12m


def _homografia_sintetica() -> np.ndarray:
    return calcular_homografia(CANTOS_PX, CANTOS_MUNDO_M)


def _ponto_imagem_para(
    homografia: np.ndarray, x_mundo: float, y_mundo: float
) -> tuple[float, float]:
    """Projeta um ponto do mundo real de volta para pixels (inverso da homografia).

    Usado só para CONSTRUIR o cenário de teste: simula onde a câmera veria um
    objeto que está naquela posição real, para então alimentar o tracker com
    esse pixel e conferir se ele recupera a posição/velocidade real correta.
    """
    h_inv = np.linalg.inv(homografia)
    ponto = np.array([[x_mundo, y_mundo]], dtype=np.float32).reshape(-1, 1, 2)
    projetado = cv2.perspectiveTransform(ponto, h_inv).reshape(-1, 2)[0]
    return float(projetado[0]), float(projetado[1])


def _bbox_com_base_em(u: float, v: float) -> tuple[float, float, float, float]:
    """Monta uma bbox fake cuja base-centro (ponto de contato) é exatamente (u, v)."""
    return (u - 20, v - 60, u + 20, v)


class TestMultiObjectTrackerVelocidade:
    def test_recupera_velocidade_real_de_trajetoria_conhecida(self):
        """Cadeirante andando a 0.8 m/s no mundo real: o tracker deve recuperar
        essa velocidade a partir apenas das posições em pixels + homografia.
        """
        H = _homografia_sintetica()
        tracker = MultiObjectTracker(
            homografia=H, cantos_faixa_px=CANTOS_PX, velocidade_fallback_ms=0.7
        )

        velocidade_real_ms = 0.8
        ultimo_objeto = None

        for i in range(10):
            tempo_s = i / 10.0  # 10 fps nominal
            y_mundo = 1.0 + velocidade_real_ms * tempo_s
            u, v = _ponto_imagem_para(H, x_mundo=2.0, y_mundo=y_mundo)
            bbox = _bbox_com_base_em(u, v)

            objetos = tracker.processar_deteccoes(
                boxes_xyxy=[bbox], track_ids=[7], confiancas=[0.9], tempo_s=tempo_s
            )
            ultimo_objeto = objetos[0]

        assert ultimo_objeto.usou_medicao is True
        assert ultimo_objeto.velocidade_ms == pytest.approx(velocidade_real_ms, abs=0.1)
        assert ultimo_objeto.dentro_da_faixa is True
        assert ultimo_objeto.track_id == 7

    def test_poucas_observacoes_usa_fallback(self):
        """Com menos amostras que o mínimo exigido, deve cair no fallback do site."""
        H = _homografia_sintetica()
        tracker = MultiObjectTracker(
            homografia=H, cantos_faixa_px=CANTOS_PX, velocidade_fallback_ms=0.7
        )

        u, v = _ponto_imagem_para(H, x_mundo=2.0, y_mundo=1.0)
        bbox = _bbox_com_base_em(u, v)

        objetos = tracker.processar_deteccoes(
            boxes_xyxy=[bbox], track_ids=[1], confiancas=[0.9], tempo_s=0.0
        )

        assert objetos[0].usou_medicao is False
        assert objetos[0].velocidade_ms == 0.7

    def test_ponto_fora_da_faixa_marca_flag_correta(self):
        """Um ponto fora do polígono da faixa deve ser identificado como tal."""
        H = _homografia_sintetica()
        tracker = MultiObjectTracker(
            homografia=H, cantos_faixa_px=CANTOS_PX, velocidade_fallback_ms=0.7
        )

        # Ponto bem fora do polígono definido por CANTOS_PX
        bbox_fora = (10, 10, 50, 50)

        objetos = tracker.processar_deteccoes(
            boxes_xyxy=[bbox_fora], track_ids=[2], confiancas=[0.9], tempo_s=0.0
        )

        assert objetos[0].dentro_da_faixa is False

    def test_multiplos_tracks_simultaneos_sao_independentes(self):
        """Dois cadeirantes ao mesmo tempo não devem misturar seus históricos de posição."""
        H = _homografia_sintetica()
        tracker = MultiObjectTracker(
            homografia=H, cantos_faixa_px=CANTOS_PX, velocidade_fallback_ms=0.7
        )

        for i in range(6):
            tempo_s = i / 10.0
            u1, v1 = _ponto_imagem_para(H, x_mundo=1.0, y_mundo=1.0 + 0.5 * tempo_s)
            u2, v2 = _ponto_imagem_para(H, x_mundo=3.0, y_mundo=1.0 + 1.2 * tempo_s)

            objetos = tracker.processar_deteccoes(
                boxes_xyxy=[_bbox_com_base_em(u1, v1), _bbox_com_base_em(u2, v2)],
                track_ids=[10, 20],
                confiancas=[0.9, 0.9],
                tempo_s=tempo_s,
            )

        por_id = {o.track_id: o for o in objetos}
        # Track 20 é mais rápido que o track 10; ambos devem ter sido medidos
        # de forma independente (não podem ter se misturado).
        assert por_id[10].velocidade_ms < por_id[20].velocidade_ms

    def test_limpar_tracks_inativos_remove_estado_antigo(self):
        """Após limpar, um track que reaparece com o mesmo ID começa do zero
        (não deve herdar o histórico de posição de antes de sumir).
        """
        H = _homografia_sintetica()
        tracker = MultiObjectTracker(
            homografia=H, cantos_faixa_px=CANTOS_PX, velocidade_fallback_ms=0.7
        )

        for i in range(6):
            u, v = _ponto_imagem_para(H, x_mundo=2.0, y_mundo=1.0 + 0.8 * (i / 10.0))
            tracker.processar_deteccoes(
                boxes_xyxy=[_bbox_com_base_em(u, v)],
                track_ids=[5],
                confiancas=[0.9],
                tempo_s=i / 10.0,
            )

        tracker.limpar_tracks_inativos(
            track_ids_ativos=set()
        )  # ninguém ativo -> remove tudo

        u, v = _ponto_imagem_para(H, x_mundo=2.0, y_mundo=1.0)
        objetos = tracker.processar_deteccoes(
            boxes_xyxy=[_bbox_com_base_em(u, v)],
            track_ids=[5],
            confiancas=[0.9],
            tempo_s=100.0,
        )

        # Só uma amostra desde a limpeza -> deve estar no fallback, não numa medição
        # calculada erroneamente contra o histórico antigo (que teria um salto de tempo enorme).
        assert objetos[0].usou_medicao is False


class TestExtrairDeteccoesUltralytics:
    def test_sem_boxes_retorna_listas_vazias(self):
        class ResultadoFake:
            boxes = None

        boxes, ids, confs = extrair_deteccoes_ultralytics(ResultadoFake())
        assert boxes == []
        assert ids == []
        assert confs == []

    def test_boxes_sem_id_retorna_listas_vazias(self):
        class BoxesFake:
            id = None

        class ResultadoFake:
            boxes = BoxesFake()

        boxes, ids, confs = extrair_deteccoes_ultralytics(ResultadoFake())
        assert boxes == []
        assert ids == []
        assert confs == []
