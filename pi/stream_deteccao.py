"""Servidor HTTP que faz streaming da câmera com detecções do YOLO.

Disponibiliza o vídeo anotado em tempo real via Flask. Permite visualizar
a detecção no navegador de qualquer dispositivo na mesma rede do Raspberry.
"""

import time

import cv2
from flask import Flask, Response
from ultralytics import YOLO

MODELO = "/home/mozardo/best480_ncnn_model"
IMGSZ = 480
CONF = 0.25
PORTA = 8080

app = Flask(__name__)
_model = None


def get_model() -> YOLO:
    """Carrega o modelo sob demanda (lazy loading).

    Permite que o módulo seja importado em ambientes onde o modelo não
    existe (CI, testes) sem falhar imediatamente.
    """
    global _model
    if _model is None:
        print(f"Carregando modelo: {MODELO}")
        _model = YOLO(MODELO, task="detect")
        print(f"Classes: {_model.names}")
    return _model


def configurar_camera(
    device_index: int = 0, largura: int = 640, altura: int = 480
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    return cap


def aquecer_sensor(cap: cv2.VideoCapture, duracao_segundos: float = 4.0) -> None:
    inicio = time.time()
    while time.time() - inicio < duracao_segundos:
        cap.read()


def gerar_frames():
    model = get_model()
    cap = configurar_camera()
    if not cap.isOpened():
        print("ERRO: não consegui abrir a webcam")
        return

    print("Aquecendo o sensor (4 segundos)...")
    aquecer_sensor(cap)

    print("Streaming iniciado")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)
        frame_anotado = results[0].plot()

        _, buffer = cv2.imencode(".jpg", frame_anotado)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def stream():
    return Response(
        gerar_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORTA)
