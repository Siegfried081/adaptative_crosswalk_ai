"""Rastreamento multi-objeto: liga detecção (YOLO) + associação temporal (ByteTrack)
+ geometria (homografia) + estimativa de velocidade por track.

Usa a integração nativa do Ultralytics com ByteTrack (model.track(...)), que
mantém um identificador persistente por objeto entre frames — é isso que
permite medir deslocamento de UM MESMO cadeirante ao longo do tempo, em vez
de tratar cada frame como uma detecção isolada e desconectada.

Referência: Zhang et al. (2022), "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box", ECCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crosswalk.decision.extensao_tempo import EstimadorVelocidade
from crosswalk.geometry.homography import (
    dentro_da_faixa,
    ponto_de_contato_com_solo,
    projetar_para_mundo,
)


@dataclass
class ObjetoRastreado:
    """Estado de um cadeirante rastreado neste frame, pronto para a decisão."""

    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confianca: float
    ponto_solo_px: list[float]
    posicao_mundo_m: tuple[float, float]
    velocidade_ms: float
    usou_medicao: bool
    dentro_da_faixa: bool


class MultiObjectTracker:
    """Mantém o estado de rastreamento (um EstimadorVelocidade por track_id)
    e produz, a cada frame, a lista de objetos com posição real e velocidade.
    """

    def __init__(
        self,
        homografia: np.ndarray,
        cantos_faixa_px: list[list[float]],
        velocidade_fallback_ms: float,
    ):
        self.homografia = homografia
        self.cantos_faixa_px = cantos_faixa_px
        self.velocidade_fallback_ms = velocidade_fallback_ms
        self._estimadores: dict[int, EstimadorVelocidade] = {}

    def _estimador_para(self, track_id: int) -> EstimadorVelocidade:
        if track_id not in self._estimadores:
            self._estimadores[track_id] = EstimadorVelocidade(
                velocidade_fallback_ms=self.velocidade_fallback_ms
            )
        return self._estimadores[track_id]

    def processar_deteccoes(
        self,
        boxes_xyxy: list[tuple[float, float, float, float]],
        track_ids: list[int],
        confiancas: list[float],
        tempo_s: float,
    ) -> list[ObjetoRastreado]:
        """Processa as detecções (já associadas a IDs pelo ByteTrack) de um frame.

        Args:
            boxes_xyxy: lista de bounding boxes (x1, y1, x2, y2) em pixels
            track_ids: ID persistente de cada box (mesmo índice de boxes_xyxy)
            confiancas: confiança de cada detecção
            tempo_s: instante (segundos) em que este frame foi capturado —
                frame_idx/fps para vídeo gravado, relógio de parede para captura ao vivo

        Returns:
            Lista de ObjetoRastreado, um por track ativo neste frame.
        """
        resultados = []

        for bbox, track_id, conf in zip(boxes_xyxy, track_ids, confiancas):
            ponto_solo = ponto_de_contato_com_solo(bbox)
            posicao_mundo = projetar_para_mundo(self.homografia, [ponto_solo])[0]
            posicao_mundo_tupla = (float(posicao_mundo[0]), float(posicao_mundo[1]))

            estimador = self._estimador_para(track_id)
            estimador.atualizar(posicao_mundo_tupla, tempo_s)
            velocidade, usou_medicao = estimador.velocidade_atual_ms()

            resultados.append(
                ObjetoRastreado(
                    track_id=track_id,
                    bbox_xyxy=bbox,
                    confianca=conf,
                    ponto_solo_px=ponto_solo,
                    posicao_mundo_m=posicao_mundo_tupla,
                    velocidade_ms=velocidade,
                    usou_medicao=usou_medicao,
                    dentro_da_faixa=dentro_da_faixa(ponto_solo, self.cantos_faixa_px),
                )
            )

        return resultados

    def limpar_tracks_inativos(self, track_ids_ativos: set[int]) -> None:
        """Remove estimadores de tracks que o ByteTrack não reporta mais
        (o objeto saiu de cena ou o track foi perdido/reatribuído).

        Chame isso a cada frame passando o conjunto de IDs vistos NESTE frame,
        para não acumular memória indefinidamente numa execução longa.
        """
        for track_id in list(self._estimadores.keys()):
            if track_id not in track_ids_ativos:
                del self._estimadores[track_id]


def extrair_deteccoes_ultralytics(resultado) -> tuple[list, list, list]:
    """Extrai (boxes_xyxy, track_ids, confiancas) de um resultado do Ultralytics.

    Isola a dependência do formato específico do objeto `Results` retornado por
    `model.track(...)`, para o resto do código não precisar conhecer a API do
    Ultralytics diretamente. Retorna listas vazias se não houver tracks (ex.:
    quando o ByteTrack ainda não atribuiu ID a nenhuma detecção do frame).
    """
    boxes = resultado.boxes
    if boxes is None or boxes.id is None:
        return [], [], []

    boxes_xyxy = boxes.xyxy.cpu().numpy().tolist()
    track_ids = boxes.id.cpu().numpy().astype(int).tolist()
    confiancas = boxes.conf.cpu().numpy().tolist()
    return boxes_xyxy, track_ids, confiancas
