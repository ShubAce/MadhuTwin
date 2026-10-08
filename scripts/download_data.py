"""Fetch the open datasets used by MadhuTwin.

Raw third-party data is never committed to the repository (CGMacros is CC BY-NC-SA 4.0,
ShanghaiT2DM is CC BY 4.0). This script reproduces the exact files we use.

CGMacros ships as a single ~657 MB zip, most of which is meal photographs. PhysioNet
serves it slowly, so we read the zip's central directory with HTTP Range requests and pull
only the CSV members we need, in parallel, verifying each member's CRC-32.

    python scripts/download_data.py            # both datasets
    python scripts/download_data.py cgmacros   # one dataset
"""

from __future__ import annotations

import argparse
import hashlib
import io
import struct
import sys
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CGMACROS_ZIP = "https://physionet.org/files/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
CGMACROS_ZIP_SHA256 = "05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d"
CGMACROS_EXTRA = {
    "DataDictionary_Bio.csv": "2d5b4c27873284ddbf218df51172b69ffa9caf6e97a3a122977b7a2761da4c61",
    "DataDictionary_CGMacros-00X.csv": "db7bf114d61b90fc5de43fed5e5cd6ba33af37e900a706e1b5747d7b0ffac3f8",
}
CGMACROS_BASE = "https://physionet.org/files/cgmacros/1.0.0/"

SHANGHAI_ZIP = "https://ndownloader.figshare.com/files/38259264"

LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")  # zip local file header, 30 bytes


class HttpRangeReader(io.RawIOBase):
    """Minimal seekable, read-only view of a remote file via HTTP Range requests.

    Only used to let `zipfile` parse the central directory; member payloads are fetched
    with one request each in `_fetch_member`.
    """

    def __init__(self, url: str, session: requests.Session):
        self.url, self.session, self.pos = url, session, 0
        head = session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = range_get(self.session, self.url, self.pos, self.pos + n - 1)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def range_get(session: requests.Session, url: str, start: int, end: int, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            r.raise_for_status()
            if r.status_code != 206:
                raise RuntimeError(f"server ignored Range header (HTTP {r.status_code})")
            return r.content
        except (requests.RequestException, RuntimeError):
            if attempt == retries - 1:
                raise
    raise AssertionError("unreachable")


def _fetch_member(url: str, info: zipfile.ZipInfo, dest: Path) -> tuple[str, int]:
    session = requests.Session()
    header = range_get(session, url, info.header_offset, info.header_offset + LOCAL_HEADER.size - 1)
    fields = LOCAL_HEADER.unpack(header)
    if fields[0] != b"PK\x03\x04":
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = fields[-2], fields[-1]
    data_start = info.header_offset + LOCAL_HEADER.size + name_len + extra_len
    payload = range_get(session, url, data_start, data_start + info.compress_size - 1)

    if info.compress_type == zipfile.ZIP_STORED:
        raw = payload
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompressobj(-zlib.MAX_WBITS).decompress(payload)
    else:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(raw) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return info.filename, len(raw)


def download_cgmacros(workers: int = 12) -> None:
    out = RAW / "cgmacros"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for name, sha in CGMACROS_EXTRA.items():
        path = out / name
        if not path.exists():
            path.write_bytes(session.get(CGMACROS_BASE + name, timeout=120).content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError(f"checksum mismatch: {name}")

    reader = HttpRangeReader(CGMACROS_ZIP, session)
    # Unbuffered on purpose: zipfile issues a handful of small reads; buffering would fetch MBs.
    with zipfile.ZipFile(reader) as zf:
        wanted = [
            i for i in zf.infolist()
            if i.filename.lower().endswith(".csv") and "/__macosx" not in i.filename.lower()
            and not Path(i.filename).name.startswith("._")
        ]
    todo = []
    for info in wanted:
        dest = out / Path(info.filename).name
        if not (dest.exists() and dest.stat().st_size == info.file_size):
            todo.append((info, dest))
    total = sum(i.compress_size for i, _ in todo)
    print(f"CGMacros: {len(wanted)} CSV members ({len(todo)} to fetch, {total / 1e6:.1f} MB compressed)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_member, CGMACROS_ZIP, info, dest) for info, dest in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            name, size = fut.result()
            print(f"  [{k}/{len(todo)}] {Path(name).name} ({size / 1e6:.2f} MB)")
    print(f"CGMacros ready in {out}  (zip sha256 {CGMACROS_ZIP_SHA256[:12]}..., CC BY-NC-SA 4.0)")


# PhysioNet's AWS open-data mirror is ~35x faster than physionet.org for this dataset (v1.1.2;
# v1.1.3 on physionet.org differs only in documentation).
BIGIDEAS_BASE = "https://physionet-open.s3.amazonaws.com/big-ideas-glycemic-wearable/1.1.2/"
BIGIDEAS_FILES = ("Dexcom", "HR", "IBI", "Food_Log")  # skip ACC/BVP/EDA/TEMP (30+ GB of raw signal)


def _parallel_file(url: str, dest: Path, session: requests.Session, pool: ThreadPoolExecutor, chunk: int = 2 << 20) -> None:
    """Download one file as concurrent byte ranges (PhysioNet throttles each connection)."""
    head = session.head(url, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    if dest.exists() and dest.stat().st_size == size:
        return
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    parts = list(pool.map(lambda r: range_get(requests.Session(), url, r[0], r[1]), ranges))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"".join(parts))


def download_bigideas(workers: int = 32) -> None:
    """BIG IDEAs Lab glycemic + wearable data (PhysioNet, ODC-By 1.0): CGM, heart rate, beat-to-beat
    intervals and food logs for 16 adults; the large raw accelerometer/BVP/EDA/temperature files are skipped."""
    out = RAW / "bigideas"
    session = requests.Session()
    (out / "Demographics.csv").parent.mkdir(parents=True, exist_ok=True)
    (out / "Demographics.csv").write_bytes(session.get(BIGIDEAS_BASE + "Demographics.csv", timeout=120).content)
    jobs = [(f"{pid:03d}", kind) for pid in range(1, 17) for kind in BIGIDEAS_FILES]
    with ThreadPoolExecutor(max_workers=workers) as pool, ThreadPoolExecutor(max_workers=8) as files:
        futures = {files.submit(_parallel_file, f"{BIGIDEAS_BASE}{pid}/{kind}_{pid}.csv", out / pid / f"{kind}_{pid}.csv",
                                session, pool): (pid, kind) for pid, kind in jobs}
        for k, f in enumerate(as_completed(futures), 1):
            f.result()
            if k % 8 == 0 or k == len(jobs):
                print(f"  BIG IDEAs: {k}/{len(jobs)} files")
    print(f"BIG IDEAs ready in {out}  (ODC-By 1.0)")


def download_shanghai() -> None:
    out = RAW / "shanghai"
    target = out / "ShanghaiDM"
    if (target / "Shanghai_T2DM_Summary.xlsx").exists():
        print(f"ShanghaiT2DM already present in {target}")
        return
    out.mkdir(parents=True, exist_ok=True)
    blob = requests.get(SHANGHAI_ZIP, timeout=300).content
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            name = Path(info.filename).name
            if info.is_dir() or name.startswith("~$") or len(parts) < 2:
                continue
            # The archive's top folder has a non-UTF-8 (GBK) name; normalise it.
            dest = target.joinpath(*parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info))
    print(f"ShanghaiT2DM ready in {target}  (CC BY 4.0)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=["cgmacros", "shanghai", "bigideas"], choices=["cgmacros", "shanghai", "bigideas"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if "shanghai" in args.datasets:
        download_shanghai()
    if "cgmacros" in args.datasets:
        download_cgmacros(args.workers)
    if "bigideas" in args.datasets:
        download_bigideas()
    return 0


if __name__ == "__main__":
    sys.exit(main())
