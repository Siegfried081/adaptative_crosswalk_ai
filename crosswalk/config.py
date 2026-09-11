"""Modelo de dados da configuração por site (por instalação de semáforo).

A ideia central: o CÓDIGO é igual em todo lugar; a CONFIGURAÇÃO é única por
cruzamento. Cada semáforo instalado tem um arquivo YAML próprio descrevendo sua
geometria física, e o software carrega esse arquivo ao iniciar.

Isso é o que a indústria chama de "commissioning" (comissionamento): a etapa de
configuração feita uma vez por instalação. Trocar de semáforo = trocar o YAML.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class CameraConfig:
    """Parâmetros físicos da câmera na instalação."""

    altura_montagem_m: float
    angulo_inclinacao_graus: float
    resolucao: tuple[int, int]  # (largura, altura) em pixels
    distancia_focal_mm: float | None = None


@dataclass
class CrosswalkConfig:
    """Geometria da faixa de pedestres.

    Os 4 cantos marcados na imagem servem para DUAS coisas:
    1. Delimitar a região de interesse (ROI) — onde procurar cadeirantes
    2. Calibrar a homografia — junto com as dimensões reais

    Convenção de ordem dos cantos (sentido horário, começando pelo mais próximo
    da câmera à esquerda):
        0: perto-esquerda   1: perto-direita
        3: longe-esquerda   2: longe-direita

    Convenção do mundo real (metros), origem no canto perto-esquerda:
        eixo X = largura da faixa (perpendicular à travessia)
        eixo Y = comprimento da travessia (a distância D que o pedestre percorre)
    """

    # 4 cantos na imagem, em pixels: [[u, v], ...] na ordem da convenção acima
    cantos_imagem_px: list[list[float]]

    # Dimensões reais da faixa, em metros
    largura_travessia_m: float  # extensão lateral (eixo X no mundo)
    comprimento_travessia_m: float  # a distância D percorrida na travessia (eixo Y)

    def cantos_mundo_m(self) -> list[list[float]]:
        """Retorna as coordenadas reais dos 4 cantos, em metros.

        Deriva o retângulo do mundo a partir das dimensões, na mesma ordem
        dos cantos na imagem — o que permite calcular a homografia.
        """
        w = self.largura_travessia_m
        d = self.comprimento_travessia_m
        return [
            [0.0, 0.0],  # perto-esquerda  -> origem
            [w, 0.0],  # perto-direita
            [w, d],  # longe-direita
            [0.0, d],  # longe-esquerda
        ]


@dataclass
class SemaphoreConfig:
    """Parâmetros de temporização do semáforo."""

    tempo_verde_normal_s: float  # tempo verde de pedestre já programado
    tempo_extra_maximo_s: float = 15.0  # teto de segurança da extensão
    velocidade_projeto_ms: float = 1.2  # velocidade de projeto do semáforo (padrão)
    velocidade_cadeirante_fallback_ms: float = 0.7  # usada quando não se mede a real


@dataclass
class CalibrationConfig:
    """Resultado da calibração: a matriz de homografia derivada.

    A homografia é DERIVADA (dos cantos + dimensões), mas guardamos ela pronta
    no config para não recalcular a cada inicialização.
    """

    homografia: list[list[float]]  # matriz 3x3, imagem (px) -> mundo (m)


@dataclass
class SiteConfig:
    """Configuração completa de um site (uma instalação de semáforo)."""

    id: str
    nome: str
    camera: CameraConfig
    crosswalk: CrosswalkConfig
    semaphore: SemaphoreConfig
    calibration: CalibrationConfig

    # -------- Serialização --------

    def to_yaml(self, caminho: str | Path) -> None:
        """Salva a configuração em um arquivo YAML."""
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        # Tuplas viram !!python/tuple no YAML e o safe_load recusa; força lista.
        data["camera"]["resolucao"] = list(self.camera.resolucao)
        with caminho.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, caminho: str | Path) -> "SiteConfig":
        """Carrega a configuração de um arquivo YAML."""
        with Path(caminho).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        camera_data = dict(data["camera"])
        camera_data["resolucao"] = tuple(camera_data["resolucao"])

        return cls(
            id=data["id"],
            nome=data["nome"],
            camera=CameraConfig(**camera_data),
            crosswalk=CrosswalkConfig(**data["crosswalk"]),
            semaphore=SemaphoreConfig(**data["semaphore"]),
            calibration=CalibrationConfig(**data["calibration"]),
        )
