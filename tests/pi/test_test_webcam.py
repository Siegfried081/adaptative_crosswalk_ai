"""Testes para pi/test_webcam.py.

Cobre: configuração da câmera, análise de brilho, fluxo de aquecimento.
"""

from unittest.mock import patch

from pi.test_webcam import analisar_brilho, aquecer_sensor, configurar_camera


class TestConfigurarCamera:
    """Configuração da webcam com MJPEG e resolução padrão."""

    def test_abre_dispositivo_correto(self):
        with patch("pi.test_webcam.cv2.VideoCapture") as cap_class:
            configurar_camera(device_index=0)
            cap_class.assert_called_once_with(0)

    def test_define_resolucao_640x480_por_padrao(self):
        with patch("pi.test_webcam.cv2.VideoCapture") as cap_class:
            cap_instance = cap_class.return_value
            configurar_camera()
            # set() é chamado 3 vezes: FOURCC, WIDTH, HEIGHT
            assert cap_instance.set.call_count == 3


class TestAnaliseBrilho:
    """Análise do brilho médio do frame capturado."""

    def test_frame_normal_retorna_status_ok(self, frame_mock):
        resultado = analisar_brilho(frame_mock)
        assert "normal" in resultado.lower()

    def test_frame_saturado_retorna_aviso(self, frame_saturado):
        resultado = analisar_brilho(frame_saturado)
        assert "saturado" in resultado.lower()

    def test_frame_escuro_retorna_aviso(self, frame_escuro):
        resultado = analisar_brilho(frame_escuro)
        assert "escuro" in resultado.lower()


class TestAquecerSensor:
    """Fluxo de aquecimento do sensor (descarte de frames)."""

    def test_chama_read_multiplas_vezes(self, cap_mock):
        aquecer_sensor(cap_mock, duracao_segundos=0.1)
        assert cap_mock.read.call_count > 0

    def test_duracao_respeita_parametro(self, cap_mock):
        import time

        inicio = time.time()
        aquecer_sensor(cap_mock, duracao_segundos=0.2)
        decorrido = time.time() - inicio
        assert decorrido >= 0.2
        assert decorrido < 0.5  # margem de tolerância
