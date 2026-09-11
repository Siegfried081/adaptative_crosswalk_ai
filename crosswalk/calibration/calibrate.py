"""Ferramenta de comissionamento: calibra um site e gera o arquivo de configuração.

Fluxo (o que um técnico faria ao instalar o sistema num novo semáforo):
  1. Aponta a ferramenta para uma imagem/frame do cruzamento
  2. Clica nos 4 cantos da faixa de pedestres (ordem definida abaixo)
  3. Informa as dimensões reais da faixa (largura e comprimento em metros)
  4. Informa os parâmetros do semáforo (tempo verde, teto de extensão)
  5. A ferramenta calcula a homografia e salva o YAML de configuração

Ordem dos cliques (importante — a homografia depende disso):
    1º clique: canto PERTO-ESQUERDA da faixa
    2º clique: canto PERTO-DIREITA
    3º clique: canto LONGE-DIREITA
    4º clique: canto LONGE-ESQUERDA

Uso:
    python -m crosswalk.calibration.calibrate --frame frame.jpg --saida configs/sites/demo.yaml
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from crosswalk.config import (
    CalibrationConfig,
    CameraConfig,
    CrosswalkConfig,
    SemaphoreConfig,
    SiteConfig,
)
from crosswalk.geometry.homography import calcular_homografia

ROTULOS_CANTOS = [
    "1: PERTO-ESQUERDA",
    "2: PERTO-DIREITA",
    "3: LONGE-DIREITA",
    "4: LONGE-ESQUERDA",
]


class ColetorDeCantos:
    """Coleta 4 cliques do usuário sobre a imagem, com feedback visual."""

    def __init__(self, imagem: np.ndarray):
        self.imagem_base = imagem
        self.pontos: list[list[float]] = []

    def _callback_mouse(self, evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN and len(self.pontos) < 4:
            self.pontos.append([float(x), float(y)])

    def _desenhar(self) -> np.ndarray:
        vis = self.imagem_base.copy()

        for i, (px, py) in enumerate(self.pontos):
            cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 0), -1)
            cv2.putText(
                vis,
                str(i + 1),
                (int(px) + 8, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if len(self.pontos) == 4:
            poly = np.array(self.pontos, dtype=np.int32)
            cv2.polylines(vis, [poly], isClosed=True, color=(0, 200, 255), thickness=2)

        if len(self.pontos) < 4:
            instrucao = f"Clique no canto: {ROTULOS_CANTOS[len(self.pontos)]}"
            cor = (0, 255, 255)
        else:
            instrucao = "4 cantos marcados. ENTER para confirmar, R para refazer."
            cor = (0, 255, 0)

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(vis, instrucao, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        return vis

    def coletar(self) -> list[list[float]]:
        """Abre a janela e coleta os 4 cliques. Retorna os pontos confirmados."""
        janela = "Calibracao - marque os 4 cantos da faixa"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(janela, self._callback_mouse)

        while True:
            cv2.imshow(janela, self._desenhar())
            tecla = cv2.waitKey(20) & 0xFF

            if tecla == ord("r"):
                self.pontos = []
            elif tecla == 13 and len(self.pontos) == 4:
                break
            elif tecla == 27:
                cv2.destroyWindow(janela)
                raise SystemExit("Calibração cancelada pelo usuário.")

        cv2.destroyWindow(janela)
        return self.pontos


def _perguntar_float(mensagem: str, padrao: float | None = None) -> float:
    """Pergunta um número ao usuário no terminal, com valor padrão opcional."""
    sufixo = f" [{padrao}]" if padrao is not None else ""
    while True:
        resposta = input(f"{mensagem}{sufixo}: ").strip()
        if not resposta and padrao is not None:
            return padrao
        try:
            return float(resposta)
        except ValueError:
            print("  Valor inválido, tente novamente.")


def calibrar(frame_path: str, saida_path: str) -> SiteConfig:
    """Executa o fluxo completo de calibração e salva o config."""
    imagem = cv2.imread(frame_path)
    if imagem is None:
        raise FileNotFoundError(f"Não consegui abrir o frame: {frame_path}")

    altura_img, largura_img = imagem.shape[:2]

    print("\nAbrindo janela de calibração. Clique nos 4 cantos da faixa.")
    coletor = ColetorDeCantos(imagem)
    cantos_px = coletor.coletar()

    print("\n--- Dimensões reais da faixa (metros) ---")
    largura_m = _perguntar_float("Largura da faixa (lateral)", 4.0)
    comprimento_m = _perguntar_float("Comprimento da travessia (distância D)", 12.0)

    print("\n--- Parâmetros do semáforo ---")
    verde_normal = _perguntar_float("Tempo verde normal (s)", 20.0)
    extra_max = _perguntar_float("Extensão máxima permitida (s)", 15.0)
    vel_cadeirante = _perguntar_float("Velocidade fallback do cadeirante (m/s)", 0.7)

    print("\n--- Câmera (opcional, ENTER para pular) ---")
    altura_cam = _perguntar_float("Altura de montagem da câmera (m)", 4.0)
    angulo_cam = _perguntar_float("Ângulo de inclinação (graus)", 30.0)

    crosswalk = CrosswalkConfig(
        cantos_imagem_px=cantos_px,
        largura_travessia_m=largura_m,
        comprimento_travessia_m=comprimento_m,
    )
    H = calcular_homografia(cantos_px, crosswalk.cantos_mundo_m())

    config = SiteConfig(
        id="SITE-DEMO",
        nome="Site de demonstração (PoC)",
        camera=CameraConfig(
            altura_montagem_m=altura_cam,
            angulo_inclinacao_graus=angulo_cam,
            resolucao=(largura_img, altura_img),
        ),
        crosswalk=crosswalk,
        semaphore=SemaphoreConfig(
            tempo_verde_normal_s=verde_normal,
            tempo_extra_maximo_s=extra_max,
            velocidade_cadeirante_fallback_ms=vel_cadeirante,
        ),
        calibration=CalibrationConfig(homografia=H.tolist()),
    )

    config.to_yaml(saida_path)
    print(f"\n✓ Configuração salva em: {saida_path}")

    print(
        "\nVerificação — cantos projetados para o mundo real (deve formar um retângulo):"
    )
    from crosswalk.geometry.homography import projetar_para_mundo

    mundo = projetar_para_mundo(H, cantos_px)
    for rotulo, (X, Y) in zip(ROTULOS_CANTOS, mundo):
        print(f"  {rotulo}: X={X:.2f}m, Y={Y:.2f}m")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Calibração de site para o semáforo adaptativo"
    )
    parser.add_argument(
        "--frame", required=True, help="Caminho para um frame do cruzamento"
    )
    parser.add_argument(
        "--saida", required=True, help="Caminho de saída do YAML de config"
    )
    args = parser.parse_args()

    calibrar(args.frame, args.saida)


if __name__ == "__main__":
    main()
