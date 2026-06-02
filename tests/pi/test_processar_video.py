"""Testes para pi/processar_video.py.

Cobre: abertura de vídeo, tratamento de arquivo inexistente.
"""

from unittest.mock import patch

import pytest

from pi.processar_video import abrir_video


class TestAbrirVideo:
    """Abertura de arquivo de vídeo com tratamento de erro."""

    def test_arquivo_valido_retorna_capture(self):
        with patch("pi.processar_video.cv2.VideoCapture") as cap_class:
            cap_instance = cap_class.return_value
            cap_instance.isOpened.return_value = True

            resultado = abrir_video("video.mp4")
            assert resultado is cap_instance

    def test_arquivo_invalido_lanca_filenotfound(self):
        with patch("pi.processar_video.cv2.VideoCapture") as cap_class:
            cap_instance = cap_class.return_value
            cap_instance.isOpened.return_value = False

            with pytest.raises(FileNotFoundError, match="Não consegui abrir"):
                abrir_video("inexistente.mp4")
