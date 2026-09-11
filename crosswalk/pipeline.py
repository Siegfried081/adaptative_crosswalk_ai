"""Pipeline completa da PoC: lê um vídeo OU webcam, detecta, rastreia, decide e visualiza.

Fluxo por frame:
    frame -> YOLO+ByteTrack (model.track) -> MultiObjectTracker (homografia +
    velocidade) -> decidir_extensao() -> SignalController.estender_verde()

Funciona tanto sobre um arquivo de vídeo quanto sobre uma webcam ao vivo — não
precisa da instalação real nem do site calibrado de verdade para testar a
mecânica. Quando a calibração real e o hardware estiverem prontos, só troca o
site config e o SignalController; o resto do código não muda.

Uso (vídeo gravado):
    python -m crosswalk.pipeline --video entrada.mp4 --site configs/sites/demo.yaml \
        --modelo best_combinado.pt --saida saida_anotada.mp4

Uso (webcam ao vivo, índice 0):
    python -m crosswalk.pipeline --video 0 --site configs/sites/demo.yaml \
        --modelo best_combinado.pt --mostrar
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from crosswalk.config import SiteConfig
from crosswalk.control.signal_controller import MockSignalController
from crosswalk.decision.extensao_tempo import decidir_extensao
from crosswalk.tracking.tracker import MultiObjectTracker, extrair_deteccoes_ultralytics


def _resolver_fonte(video_arg: str) -> int | str:
    """Se o argumento for só dígitos, trata como índice de webcam (int).
    Caso contrário, é um caminho de arquivo de vídeo.
    """
    return int(video_arg) if video_arg.isdigit() else video_arg


def _desenhar_overlay(
    frame: np.ndarray,
    objetos,
    cantos_faixa_px: list[list[float]],
    tempo_extra_atual: float,
    fps_processamento: float | None = None,
) -> np.ndarray:
    """Desenha bounding boxes, IDs, velocidades, a região da faixa e o painel de status."""
    vis = frame.copy()

    poligono = np.array(cantos_faixa_px, dtype=np.int32)
    cor_faixa = (0, 200, 255) if tempo_extra_atual > 0 else (200, 200, 200)
    cv2.polylines(vis, [poligono], isClosed=True, color=cor_faixa, thickness=2)

    for obj in objetos:
        x1, y1, x2, y2 = map(int, obj.bbox_xyxy)
        cor = (0, 255, 0) if obj.dentro_da_faixa else (150, 150, 150)
        cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)

        fonte_vel = "med" if obj.usou_medicao else "fallback"
        rotulo = f"#{obj.track_id} {obj.velocidade_ms:.2f}m/s ({fonte_vel})"
        cv2.putText(
            vis, rotulo, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2
        )

        cv2.circle(vis, tuple(map(int, obj.ponto_solo_px)), 4, (0, 0, 255), -1)

    painel_h = 40
    cv2.rectangle(vis, (0, 0), (vis.shape[1], painel_h), (0, 0, 0), -1)
    texto_status = (
        f"Extensao atual: {tempo_extra_atual:.1f}s   |   "
        f"Cadeirantes na faixa: {sum(o.dentro_da_faixa for o in objetos)}"
    )
    if fps_processamento is not None:
        texto_status += f"   |   {fps_processamento:.1f} FPS"
    cv2.putText(
        vis, texto_status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    return vis


def rodar_pipeline(
    video_path: str | int,
    site_config_path: str,
    modelo_path: str,
    saida_path: str | None = None,
    conf: float = 0.25,
    imgsz: int = 640,
    mostrar: bool = False,
    captura_largura: int | None = None,
    captura_altura: int | None = None,
) -> MockSignalController:
    """Executa a pipeline completa sobre um arquivo de vídeo OU uma webcam (índice int).

    Returns:
        O MockSignalController usado, com o histórico completo de extensões
        solicitadas — útil para inspecionar/testar o resultado da PoC.
    """
    from ultralytics import YOLO  # import tardio: só é necessário para rodar de fato

    site = SiteConfig.from_yaml(site_config_path)
    homografia = np.array(site.calibration.homografia, dtype=np.float32)
    cantos_faixa_px = site.crosswalk.cantos_imagem_px
    distancia_m = site.crosswalk.comprimento_travessia_m

    model = YOLO(modelo_path)

    ao_vivo = isinstance(video_path, int)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir a fonte de vídeo: {video_path}")

    if ao_vivo:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if captura_largura:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, captura_largura)
        if captura_altura:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, captura_altura)

    fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = MultiObjectTracker(
        homografia=homografia,
        cantos_faixa_px=cantos_faixa_px,
        velocidade_fallback_ms=site.semaphore.velocidade_cadeirante_fallback_ms,
    )
    controller = MockSignalController(verboso=True)

    writer = None
    if saida_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(saida_path, fourcc, fps_nominal, (largura, altura))

    frame_idx = 0
    t_inicio = time.perf_counter()
    t_ultimo_log_fps = t_inicio
    frames_desde_log = 0
    fps_processamento = None

    modo = "AO VIVO (webcam)" if ao_vivo else "vídeo gravado"
    print(
        f"Iniciando pipeline [{modo}]: {video_path} — {largura}x{altura}, distância={distancia_m}m"
    )
    if ao_vivo:
        print("Pressione ESC na janela para encerrar.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tempo_s = (
            (time.perf_counter() - t_inicio) if ao_vivo else (frame_idx / fps_nominal)
        )

        resultado = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )[0]
        boxes_xyxy, track_ids, confiancas = extrair_deteccoes_ultralytics(resultado)

        objetos = tracker.processar_deteccoes(
            boxes_xyxy, track_ids, confiancas, tempo_s
        )
        tracker.limpar_tracks_inativos(set(track_ids))

        velocidades_na_faixa = [o.velocidade_ms for o in objetos if o.dentro_da_faixa]
        tempo_extra = decidir_extensao(
            velocidades_na_faixa, site.semaphore, distancia_m
        )
        controller.estender_verde(tempo_extra)

        frames_desde_log += 1
        agora = time.perf_counter()
        if agora - t_ultimo_log_fps >= 1.0:
            fps_processamento = frames_desde_log / (agora - t_ultimo_log_fps)
            frames_desde_log = 0
            t_ultimo_log_fps = agora

        vis = _desenhar_overlay(
            frame, objetos, cantos_faixa_px, tempo_extra, fps_processamento
        )

        if writer:
            writer.write(vis)
        if mostrar:
            cv2.imshow("Pipeline - semáforo adaptativo", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if mostrar:
        cv2.destroyAllWindows()
    controller.encerrar()

    duracao = time.perf_counter() - t_inicio
    print(
        f"\nFrames processados: {frame_idx}  em {duracao:.1f}s  ({frame_idx/duracao:.1f} FPS médio)"
    )
    print(
        f"Maior extensão solicitada na execução: {controller.maior_extensao_registrada():.2f}s"
    )

    return controller


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline do semáforo adaptativo (PoC)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Caminho de um vídeo, OU um número (0, 1, ...) para usar a webcam correspondente",
    )
    parser.add_argument("--site", required=True, help="YAML de configuração do site")
    parser.add_argument(
        "--modelo", required=True, help="Caminho do modelo (.pt ou pasta NCNN)"
    )
    parser.add_argument(
        "--saida", default=None, help="Vídeo de saída anotado (opcional)"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Limiar de confiança")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Resolução de inferência do YOLO"
    )
    parser.add_argument("--mostrar", action="store_true", help="Exibe janela ao vivo")
    parser.add_argument(
        "--captura-largura", type=int, default=None, help="Largura de captura da webcam"
    )
    parser.add_argument(
        "--captura-altura", type=int, default=None, help="Altura de captura da webcam"
    )
    args = parser.parse_args()

    rodar_pipeline(
        video_path=_resolver_fonte(args.video),
        site_config_path=args.site,
        modelo_path=args.modelo,
        saida_path=args.saida,
        conf=args.conf,
        imgsz=args.imgsz,
        mostrar=args.mostrar,
        captura_largura=args.captura_largura,
        captura_altura=args.captura_altura,
    )


if __name__ == "__main__":
    main()
