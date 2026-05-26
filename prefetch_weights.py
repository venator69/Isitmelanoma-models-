"""Prefetch default checkpoints into the Ultralytics cache during `docker build`."""

from __future__ import annotations

from ultralytics import YOLO

WEIGHTS = (
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "rtdetr-l.pt",
    "rtdetr-x.pt",
)


def main() -> None:
    for w in WEIGHTS:
        print("Prefetching", w, flush=True)
        YOLO(w)
    print("Prefetch complete.", flush=True)


if __name__ == "__main__":
    main()
