from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rerun_viz.config import build_config, parse_args
from rerun_viz.registry import resolve_adapter
from rerun_viz.core import run_adapter_session


def main():
    args = parse_args()
    config = build_config(args)
    adapter = resolve_adapter(config)
    run_adapter_session(adapter, config)


if __name__ == "__main__":
    main()
