"""Testes para pi/gravar_deteccao.py.

Cobre: criação do gravador de vídeo, configuração da câmera.
"""

from unittest.mock import patch

from pi.gravar_deteccao import criar_gravador


class TestCriarGravador:
    """Criação do cv2.VideoWriter com parâmetros corretos."""

    def test_usa_dimensoes_da_camera(self, cap_mock):
        with patch("pi.gravar_deteccao.cv2.VideoWriter") as writer_class:
            criar_gravador(cap_mock, "saida.mp4", fps_saida=5)
            args, _ = writer_class.call_args
            # args = (arquivo, fourcc, fps, (largura, altura))
            assert args[0] == "saida.mp4"
            assert args[2] == 5
            assert args[3] == (640, 480)

    def test_usa_codec_mp4v(self, cap_mock):
        with patch("pi.gravar_deteccao.cv2.VideoWriter"):
            with patch("pi.gravar_deteccao.cv2.VideoWriter_fourcc") as fourcc:
                criar_gravador(cap_mock, "saida.mp4", fps_saida=5)
                fourcc.assert_called_once_with(*"mp4v")
