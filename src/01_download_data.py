"""
Step 01 — Download the MIT-BIH Arrhythmia Database from PhysioNet.

Downloads only the 44 records used by the inter-patient (DS1/DS2) protocol
into data/raw/. Records already present are skipped, so re-running is cheap.

Run:  python src/01_download_data.py

Requires internet access to physionet.org. If you are behind a proxy/firewall
that blocks it, download 'mitdb' manually from
https://physionet.org/content/mitdb/ and unzip it into data/raw/.
"""
import sys
import wfdb

import config as C


def main():
    records = [str(r) for r in C.ALL_RECORDS]
    print(f"Target: {len(records)} records into {C.DATA_RAW}")

    missing = []
    for r in records:
        if not (C.DATA_RAW / f"{r}.dat").exists():
            missing.append(r)
    if not missing:
        print("All records already present. Nothing to download.")
        return

    print(f"Downloading {len(missing)} records from PhysioNet.", flush=True)
    print("This can take a few minutes; each record is reported below.", flush=True)
    try:
        for i, r in enumerate(missing, 1):
            wfdb.dl_database(C.PN_DIR, dl_dir=str(C.DATA_RAW), records=[r])
            print(f"  [{i:>2}/{len(missing)}] record {r} downloaded", flush=True)
    except Exception as e:
        print("\nDOWNLOAD FAILED:", repr(e), flush=True)
        print("\nNo connection/DNS to physionet.org. Use the OFFLINE import instead:", flush=True)
        print("  1) Download on any working network:", flush=True)
        print("     https://physionet.org/static/published-projects/mitdb/"
              "mit-bih-arrhythmia-database-1.0.0.zip", flush=True)
        print('  2) Run:  python src\\01b_import_local.py "C:\\path\\to\\that.zip"', flush=True)
        sys.exit(1)

    ok = sum((C.DATA_RAW / f"{r}.dat").exists() for r in records)
    print(f"Done. {ok}/{len(records)} records available in {C.DATA_RAW}")


if __name__ == "__main__":
    main()
