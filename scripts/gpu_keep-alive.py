"""Keep GPU active to prevent SLURM from killing idle jobs.

Runs a small matrix multiply on the GPU every `interval` seconds.
Launch in the background before long CPU-bound tasks (e.g., data prep).

Usage:
    python scripts/gpu_keep-alive.py &           # default: every 60s
    python scripts/gpu_keep-alive.py --interval 30 &
"""

import argparse
import signal
import sys
import time

import torch


def keepalive(interval: int = 60):
    """Periodically run a small GPU op to prevent idle detection.

    Args:
        interval: Seconds between GPU pings
    """
    if not torch.cuda.is_available():
        print("No GPU available, exiting keepalive.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU keepalive started (every {interval}s on {torch.cuda.get_device_name(device)})")

    # Graceful shutdown on SIGTERM/SIGINT
    def _shutdown(signum, frame):
        print("\nGPU keepalive stopped.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        # Tiny matmul — negligible VRAM and compute
        a = torch.randn(64, 64, device=device)
        b = torch.randn(64, 64, device=device)
        _ = a @ b
        del a, b
        torch.cuda.synchronize()
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU keepalive for SLURM jobs")
    parser.add_argument(
        "--interval", type=int, default=60, help="Seconds between GPU pings (default: 60)"
    )
    args = parser.parse_args()
    keepalive(args.interval)
