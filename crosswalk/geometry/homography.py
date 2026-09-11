"""Homografia: converte pontos da imagem (perspectiva) para o mundo real (metros).

A câmera vê a faixa em perspectiva — distâncias iguais no mundo aparecem como
distâncias diferentes na imagem. A homografia é uma matriz 3x3 que "endireita"
essa perspectiva, mapeando o plano da imagem para o plano do chão visto de cima.

Com ela, medimos deslocamento em METROS REAIS (não pixels), o que dá velocidade
com precisão consistente em qualquer ponto da cena.
"""

from __future__ import annotations

import cv2
import numpy as np


def calcular_homografia(
    pontos_imagem: list[list[float]],
    pontos_mundo: list[list[float]],
) -> np.ndarray:
    """Calcula a matriz de homografia a partir de 4 correspondências.

    Args:
        pontos_imagem: 4 pontos [[u, v], ...] em pixels (cantos da faixa na imagem)
        pontos_mundo: 4 pontos [[X, Y], ...] em metros (posições reais dos cantos)

    A ordem dos pontos deve corresponder entre as duas listas.

    Returns:
        Matriz 3x3 que projeta pontos da imagem para o mundo real.
    """
    src = np.array(pontos_imagem, dtype=np.float32)
    dst = np.array(pontos_mundo, dtype=np.float32)

    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("São necessários exatamente 4 pontos em cada conjunto.")

    # getPerspectiveTransform é exato para 4 pontos (o nosso caso).
    # Para mais de 4 pontos, usaríamos findHomography (com ajuste por mínimos quadrados).
    return cv2.getPerspectiveTransform(src, dst)


def projetar_para_mundo(
    homografia: np.ndarray,
    pontos: list[list[float]] | np.ndarray,
) -> np.ndarray:
    """Projeta pontos da imagem (px) para coordenadas do mundo real (metros).

    Args:
        homografia: matriz 3x3 obtida de calcular_homografia()
        pontos: Nx2 pontos em pixels

    Returns:
        Nx2 pontos em metros no plano do chão.
    """
    pts = np.array(pontos, dtype=np.float32).reshape(-1, 1, 2)
    mundo = cv2.perspectiveTransform(pts, homografia)
    return mundo.reshape(-1, 2)


def ponto_de_contato_com_solo(
    bbox_xyxy: tuple[float, float, float, float],
) -> list[float]:
    """Retorna o ponto de contato do objeto com o chão (base-centro da bbox).

    Para estimar posição no plano do chão, usamos o centro da borda inferior da
    bounding box — que corresponde aproximadamente ao ponto onde o cadeirante
    toca o solo. É esse ponto que projetamos com a homografia.

    Args:
        bbox_xyxy: (x1, y1, x2, y2) da bounding box em pixels

    Returns:
        [u, v] do ponto base-centro.
    """
    x1, y1, x2, y2 = bbox_xyxy
    u = (x1 + x2) / 2.0
    v = y2  # borda inferior
    return [u, v]


def dentro_da_faixa(
    ponto: list[float],
    cantos_faixa_px: list[list[float]],
) -> bool:
    """Verifica se um ponto (px) está dentro do polígono da faixa.

    Usado para saber se o cadeirante já entrou na zona de ativação do sistema.

    Args:
        ponto: [u, v] em pixels
        cantos_faixa_px: 4 cantos do polígono da faixa, em pixels

    Returns:
        True se o ponto está dentro (ou na borda) da faixa.
    """
    poligono = np.array(cantos_faixa_px, dtype=np.int32)
    resultado = cv2.pointPolygonTest(
        poligono, (float(ponto[0]), float(ponto[1])), False
    )
    return resultado >= 0


def distancia_real_m(
    homografia: np.ndarray,
    ponto_px_a: list[float],
    ponto_px_b: list[float],
) -> float:
    """Distância real (metros) entre dois pontos da imagem, via homografia.

    Projeta ambos os pontos para o mundo e mede a distância euclidiana em metros.
    É a base para o cálculo de velocidade: distância percorrida entre dois frames.
    """
    mundo = projetar_para_mundo(homografia, [ponto_px_a, ponto_px_b])
    delta = mundo[1] - mundo[0]
    return float(np.hypot(delta[0], delta[1]))
