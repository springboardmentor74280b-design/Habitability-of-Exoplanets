#!/usr/bin/env python3
"""Simple script to call the /predict endpoint and print the response.

Usage:
  python scripts/test_predict.py --url http://127.0.0.1:5000

Requires: requests (`pip install requests`)
"""

import argparse
import sys

import requests

SAMPLE_PAYLOAD = {
    "HSI": 0.5,
    "planet_density": 5.5,
    "pl_eqt": 280,
    "pl_rade": 1.1,
    "pl_bmasse": 1.2,
    "st_teff": 3500,
    "star_luminosity": 0.02,
    "star_type_M": 1,
    "star_type_K": 0,
    "star_type_G": 0,
}


def main():
    p = argparse.ArgumentParser(description="Call the /predict endpoint")
    p.add_argument("--url", default="http://127.0.0.1:5000", help="API base URL")
    p.add_argument("--payload", action="store_true", help="Print sample payload and exit")
    args = p.parse_args()

    if args.payload:
        import json

        print(json.dumps(SAMPLE_PAYLOAD, indent=2))
        return

    try:
        r = requests.post(f"{args.url.rstrip('/')}/predict", json=SAMPLE_PAYLOAD, timeout=10)
        print(f"Status: {r.status_code}")
        try:
            print(r.json())
        except Exception:
            print(r.text)
            sys.exit(1)

        if r.status_code != 200:
            sys.exit(2)

    except requests.exceptions.RequestException as exc:
        print(f"Request failed: {exc}")
        sys.exit(3)


if __name__ == "__main__":
    main()
