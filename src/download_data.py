"""Download the public IBM Telco Customer Churn sample dataset."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset to {DEST} ...")
    try:
        with urlopen(URL, timeout=30) as response, DEST.open("wb") as output:
            output.write(response.read())
    except Exception as exc:
        if DEST.exists():
            DEST.unlink()
        raise SystemExit(f"Download failed: {exc}\nDownload the CSV manually from {URL} and place it at {DEST}") from exc
    print(f"Saved {DEST} ({DEST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
