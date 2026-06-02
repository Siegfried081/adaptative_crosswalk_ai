"""Processa um arquivo de vídeo existente com o YOLO.

Lê um vídeo MP4, roda detecção em cada frame e gera um novo vídeo com as
bounding boxes desenhadas. Útil para validar o modelo em cenas controladas
ou para gerar evidência de funcionamento para apresentações.
"""

import time

import cv2
from ultralytics import YOLO

MODELO = "/home/mozardo/best480_ncnn_model"
IMGSZ = 480
CONF = 0.25
VIDEO_ENTRADA = "/home/mozardo/cadeirante.mp4"
VIDEO_SAIDA = "/home/mozardo/cadeirante_detectado.mp4"


def abrir_video(caminho: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {caminho}")
    return cap


def criar_gravador(cap: cv2.VideoCapture, arquivo_saida: str) -> cv2.VideoWriter:
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(arquivo_saida, fourcc, fps_original, (largura, altura))


def main() -> int:
    print(f"Carregando modelo: {MODELO}")
    model = YOLO(MODELO, task="detect")

    try:
        cap = abrir_video(VIDEO_ENTRADA)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        return 1

    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Vídeo de entrada: {largura}x{altura} @ {fps_original:.1f} FPS, {total_frames} frames"
    )

    out = criar_gravador(cap, VIDEO_SAIDA)

    print("Processando frames...")
    frame_count = 0
    total_deteccoes = 0
    inicio = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)
        frame_anotado = results[0].plot()
        out.write(frame_anotado)

        n_det = len(results[0].boxes)
        total_deteccoes += n_det
        frame_count += 1

        if frame_count % 30 == 0:
            progresso = (frame_count / total_frames) * 100
            print(
                f"  {frame_count}/{total_frames} ({progresso:.1f}%) — {n_det} detecção(ões)"
            )

    cap.release()
    out.release()

    tempo_total = time.time() - inicio

    print("\n=== RESULTADO ===")
    print(f"Frames processados:     {frame_count}")
    print(f"Total de detecções:     {total_deteccoes}")
    print(f"Detecções por frame:    {total_deteccoes/frame_count:.2f}")
    print(f"Tempo de processamento: {tempo_total:.1f}s")
    print(f"Velocidade média:       {frame_count/tempo_total:.2f} FPS")
    print(f"Vídeo de saída salvo:   {VIDEO_SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
