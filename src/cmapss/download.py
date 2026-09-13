"""Download C-MAPSS FD001 files with zip + mirror fallback."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from .config import DATA_RAW, FD001_FILES

NASA_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
# Commit-pinned mirrors of the same public FD001 text files
MIRROR_CANDIDATES = [
    "https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/CMAPSSData/{name}",
    "https://huggingface.co/datasets/SoyVitou/NASA-C-MAPSS-Turbofan-Engine/resolve/main/data/{name}",
    "https://raw.githubusercontent.com/Sage-of-Sparta/NASA-Turbofan-Engine-Degradation/main/CMAPSSData/{name}",
]


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": "cmapss-rul/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _write(dest: Path, content: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    if dest.stat().st_size == 0:
        raise RuntimeError(f"Downloaded empty file: {dest}")


def _from_nasa_zip(raw_dir: Path) -> bool:
    try:
        print(f"Downloading NASA archive: {NASA_ZIP_URL}")
        content = _download_bytes(NASA_ZIP_URL, timeout=180)
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            for needed in FD001_FILES:
                match = next((n for n in names if n.endswith(needed)), None)
                if match is None:
                    print(f"  missing {needed} in zip")
                    return False
                _write(raw_dir / needed, zf.read(match))
                print(f"  extracted {needed}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"NASA zip download failed: {exc}")
        return False


def _from_mirrors(raw_dir: Path) -> bool:
    for name in FD001_FILES:
        dest = raw_dir / name
        ok = False
        for template in MIRROR_CANDIDATES:
            url = template.format(name=name)
            try:
                print(f"Trying mirror for {name}: {url}")
                _write(dest, _download_bytes(url))
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  failed: {exc}")
        if not ok:
            return False
    return True


def download_fd001(raw_dir: Path | None = None, force: bool = False) -> Path:
    """
    Download FD001 train/test/RUL text files into data/raw.

    Prefers the official NASA CMAPSSData.zip, then public mirrors.
    """
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not force and all((raw_dir / n).exists() and (raw_dir / n).stat().st_size > 0 for n in FD001_FILES):
        print(f"FD001 files already present in {raw_dir}")
        return raw_dir

    if _from_nasa_zip(raw_dir):
        print(f"FD001 files ready in {raw_dir}")
        return raw_dir

    if _from_mirrors(raw_dir):
        print(f"FD001 files ready in {raw_dir}")
        return raw_dir

    raise RuntimeError(
        "Could not download FD001. Manually place train_FD001.txt, test_FD001.txt, "
        "and RUL_FD001.txt into data/raw/ from https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    )


if __name__ == "__main__":
    download_fd001()
