"""Fixtures compartilhadas pelos testes dos scripts do Raspberry Pi.

Provê mocks reutilizáveis para cv2.VideoCapture e ultralytics.YOLO, evitando
duplicação entre os arquivos de teste.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def frame_mock():
    """Frame BGR sintético de 480x640 com brilho médio (~128)."""
    return np.full((480, 640, 3), 128, dtype=np.uint8)


@pytest.fixture
def frame_escuro():
    """Frame quase preto, para testar análise de brilho."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def frame_saturado():
    """Frame totalmente branco, para testar análise de brilho."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def cap_mock(frame_mock):
    """Mock de cv2.VideoCapture que retorna frames válidos."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, frame_mock)
    cap.get.side_effect = lambda prop: {
        3: 640.0,  # CAP_PROP_FRAME_WIDTH
        4: 480.0,  # CAP_PROP_FRAME_HEIGHT
        5: 30.0,  # CAP_PROP_FPS
        7: 900.0,  # CAP_PROP_FRAME_COUNT
    }.get(prop, 0.0)
    return cap


@pytest.fixture
def cap_mock_falha():
    """Mock de cv2.VideoCapture que falha ao abrir."""
    cap = MagicMock()
    cap.isOpened.return_value = False
    return cap


@pytest.fixture
def yolo_mock():
    """Mock do modelo YOLO com retorno realístico."""
    model = MagicMock()
    model.names = {0: "cadeirante"}

    box_mock = MagicMock()
    box_mock.__len__ = lambda self: 1  # uma detecção

    result_mock = MagicMock()
    result_mock.boxes = box_mock
    result_mock.plot.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    model.predict.return_value = [result_mock]
    return model
