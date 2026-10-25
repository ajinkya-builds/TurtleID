"""Download Stack v2 vision weights into HF_HOME and release them.

    python -m scripts.preload_models

Honours TURTLEID_DTYPE, TURTLEID_USE_BIOCAP, and TURTLEID_DETECTOR.
Text towers are dropped during load (see core/stack_v2/runtime.py).
"""

from __future__ import annotations

import logging

import config  # noqa: F401


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from core.stack_v2.runtime import preload_vision_weights

    preload_vision_weights()
    logging.info("Vision weights are cached under HF_HOME")


if __name__ == "__main__":
    main()
