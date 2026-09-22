"""python -m so1 — run the service with uvloop."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "so1.app:build",
        factory=True,
        host=os.environ.get("SO1_HOST", "0.0.0.0"),
        port=int(os.environ.get("SO1_PORT", "8080")),
        loop="uvloop",
        log_level=os.environ.get("SO1_LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
