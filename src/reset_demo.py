"""One command to (re)create the demo state: warehouse history + champion v1 + weekly stream files.
Run:  python -m src.reset_demo
Models trained later (v2, v3 ...) are deleted so every demo starts from the same clean state."""
import shutil

from src import abi_config as C
from src.adapt import init_champion
from src.config import MODELS
from src.make_stream import main as make_stream
from src.warehouse import build_warehouse, connect, integrity_report


def main(rebuild_stream: bool = True) -> None:
    for d in MODELS.glob("v*"):
        if d.name != "v1" and d.is_dir():
            shutil.rmtree(d)
    build_warehouse()
    con = connect()
    print("champion:", init_champion(con))
    print(integrity_report(con).query("meaning != 'row count'").to_string(index=False))
    con.close()
    if rebuild_stream or not (C.STREAM_DIR / "manifest.json").exists():
        make_stream()


if __name__ == "__main__":
    main()
