"""Benchmark webcam open time with different OpenCV backends.

Esempio:
    python scripts/benchmark_webcam_open.py --source 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import cv2
import platform


def benchmark(source, backend=None, runs=3):
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        if backend is not None:
            cap = cv2.VideoCapture(source, backend)
        else:
            cap = cv2.VideoCapture(source)
        t1 = time.perf_counter()

        opened = cap.isOpened()
        cap.release()

        if not opened:
            print(f"  Run {i+1}: FALLITA (non aperto)")
            return None

        elapsed = t1 - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f} sec")

    avg = sum(times) / len(times)
    print(f"  Media:   {avg:.3f} sec")
    return avg


def main():
    parser = argparse.ArgumentParser(description="Benchmark webcam open speed")
    parser.add_argument("--source", type=int, default=1, help="Indice webcam da testare")
    parser.add_argument("--runs", type=int, default=3, help="Numero di aperture per backend")
    args = parser.parse_args()

    print(f"Sistema: {platform.system()} ({platform.release()})")
    print(f"OpenCV versione: {cv2.__version__}")
    print(f"Webcam source: {args.source}\n")

    # 1. Default backend (auto)
    print("[1/3] Backend DEFAULT (auto)")
    benchmark(args.source, backend=None, runs=args.runs)
    print()

    # 2. DirectShow (Windows only)
    if sys.platform == "win32":
        print("[2/3] Backend CAP_DSHOW")
        benchmark(args.source, backend=cv2.CAP_DSHOW, runs=args.runs)
        print()

        print("[3/3] Backend CAP_MSMF")
        benchmark(args.source, backend=cv2.CAP_MSMF, runs=args.runs)
        print()
    else:
        print("[2/2] Backend CAP_V4L2 (Linux)")
        benchmark(args.source, backend=cv2.CAP_V4L2, runs=args.runs)
        print()

    print("Consiglio: usa il backend con il tempo medio minore.")


if __name__ == "__main__":
    main()
