"""
Step 01b (FALLBACK) — Import MIT-BIH from a locally downloaded ZIP or folder.

Use this when step 01 cannot reach physionet.org (DNS/proxy/firewall). Download
the database ONCE on any computer/network that can open PhysioNet, then point this
script at the file — it copies the needed records into data/raw/ for you.

Where to download (any of these, on a working network):
  * Page:  https://physionet.org/content/mitdb/   ("Download the ZIP file" at the bottom)
  * Direct: https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip

Usage:
  python src/01b_import_local.py "C:\\Users\\<you>\\Downloads\\mit-bih-arrhythmia-database-1.0.0.zip"
  python src/01b_import_local.py "C:\\path\\to\\an\\extracted\\folder"
  python src/01b_import_local.py            (auto-search this project, Downloads, Desktop)

Then run:  python src/02_preprocess.py
"""
import os
import sys
import shutil
import zipfile
from pathlib import Path

import config as C

EXTS = (".dat", ".hea", ".atr")
NEEDED = {str(r) for r in C.ALL_RECORDS}


def want(filename: str) -> bool:
    p = Path(filename)
    return p.suffix.lower() in EXTS and p.stem in NEEDED


def from_zip(zpath: Path) -> int:
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if want(Path(m).name):
                with z.open(m) as src, open(C.DATA_RAW / Path(m).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    return n


def from_folder(folder: Path) -> int:
    n = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if want(f):
                shutil.copy2(Path(root) / f, C.DATA_RAW / f)
                n += 1
    return n


def autodiscover():
    home = Path.home()
    cands = []
    for d in [C.PROJECT_ROOT, home / "Downloads", home / "Desktop", home]:
        if d.exists():
            cands += sorted(d.glob("*mit*bih*arrhythmia*.zip"))
            cands += sorted(d.glob("mitdb*.zip"))
    return cands


def main():
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.exists():
            print("Path not found:", target)
            sys.exit(1)
        sources = [target]
    else:
        sources = autodiscover()
        if not sources:
            print("No MIT-BIH zip found automatically.")
            print("Download it (see the header of this file) and run:")
            print('   python src/01b_import_local.py "C:\\path\\to\\the.zip"')
            sys.exit(1)
        print("Found:", sources[0])
        sources = [sources[0]]

    copied = 0
    for s in sources:
        if s.is_dir():
            copied += from_folder(s)
        elif s.suffix.lower() == ".zip":
            copied += from_zip(s)
        else:
            print("Unsupported (need a .zip or a folder):", s)

    have = sum((C.DATA_RAW / f"{r}.dat").exists() for r in C.ALL_RECORDS)
    print(f"Copied {copied} files. Records ready: {have}/{len(C.ALL_RECORDS)} -> {C.DATA_RAW}")
    if have == len(C.ALL_RECORDS):
        print("All needed records present. Next:  python src/02_preprocess.py")
    else:
        print("Still incomplete — make sure you pointed to the FULL MIT-BIH database zip/folder.")


if __name__ == "__main__":
    main()
