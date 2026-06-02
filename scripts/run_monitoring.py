"""
Real-time traffic monitoring demo.

Usage:
    python scripts/run_monitoring.py --source 0
    python scripts/run_monitoring.py --source path/to/video.mp4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import cv2

from src.webcam.streamer import WebcamStreamer
from src.webcam.pipeline import TrafficPipeline


def main():
    parser = argparse.ArgumentParser(description="Live traffic monitoring demo")
    parser.add_argument("--source", default=0, type=lambda x: int(x) if x.isdigit() else x,
                        help="Camera index (0) or video path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--native", action="store_true",
                        help="Use webcam native resolution (skip width/height/fps configuration)")
    args = parser.parse_args()

    print("[1/4] Caricamento modelli deep learning... (può richiedere 20-30 sec)")
    pipeline = TrafficPipeline()
    print("[2/4] Modelli caricati. Apertura webcam...")

    if args.native:
        print("[i] Modalita' nativa: risoluzione/fps non forzati")
        streamer = WebcamStreamer(args.source, width=0, height=0, fps=0)
    else:
        streamer = WebcamStreamer(args.source, args.width, args.height)

    with streamer as stream:
        print("[3/4] Webcam aperta. Avvio loop... premi 'q' per uscire, 's' per screenshot")
        frame_count = 0
        while True:
            success, frame = stream.read()
            if not success:
                print("[!] Errore lettura frame dalla webcam")
                break

            annotated = pipeline.process_frame(frame)
            cv2.imshow("Traffic Monitoring", annotated)
            frame_count += 1
            if frame_count == 1:
                print("[4/4] Finestra 'Traffic Monitoring' dovrebbe essere visibile!")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[!] Uscita richiesta da utente (tasto 'q')")
                break
            elif key == ord("s"):
                screenshot_path = Path("docs/results/screenshot.jpg")
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(screenshot_path), annotated)
                print(f"[i] Screenshot salvato in: {screenshot_path}")

    cv2.destroyAllWindows()
    print("[i] Sessione terminata.")


if __name__ == "__main__":
    main()
