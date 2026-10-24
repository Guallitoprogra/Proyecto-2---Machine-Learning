"""Descarga el archivo original de Dry Bean desde UCI."""

import argparse
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
SOURCE = "https://archive.ics.uci.edu/dataset/602/dry"
MEMBER = "DryBeanDataset/Dry_Bean_Dataset.xlsx"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="ZIP original descargado manualmente desde UCI")
    args = parser.parse_args()
    if args.zip:
        archive = args.zip.read_bytes()
    else:
        print("Descargando Dry Bean desde UCI...")
        with urlopen(URL, timeout=60) as response:
            archive = response.read()

    # Solo usamos el Excel original; no cambiamos sus filas ni sus columnas.
    with ZipFile(BytesIO(archive)) as zipped:
        workbook = zipped.read(MEMBER)

    data = pd.read_excel(BytesIO(workbook), engine="openpyxl")
    if data.shape != (13611, 17) or "Class" not in data.columns:
        raise ValueError("El archivo no coincide con la estructura esperada de Dry Bean.")
    if data["Class"].nunique() != 7:
        raise ValueError("Se esperaban siete variedades de frijol.")

    destination = ROOT / "data" / "raw" / "Dry_Bean_Dataset.xlsx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(workbook)

    # Guardamos la fecha real y las huellas para reconocer la descarga utilizada.
    metadata = {
        "dataset": "Dry Bean",
        "source": SOURCE,
        "download_url": URL,
        "acquisition": "ZIP descargado manualmente de UCI" if args.zip else "Descarga directa de UCI",
        "doi": "https://doi.org/10.24432/C50S4B",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "obtained_at": datetime.now(ZoneInfo("America/Guatemala")).isoformat(timespec="seconds"),
        "file": destination.relative_to(ROOT).as_posix(),
        "archive_sha256": sha256(archive).hexdigest(),
        "file_sha256": sha256(workbook).hexdigest(),
        "rows": len(data),
        "predictors": len(data.columns) - 1,
        "target": "Class",
        "classes": sorted(data["Class"].unique().tolist()),
        "transformations": "Ninguna. Se conserva el Excel original.",
    }
    metadata_path = ROOT / "data" / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Archivo guardado en {destination.relative_to(ROOT)}")
    print(f"Verificados {len(data):,} registros, 16 predictores y siete clases.")


if __name__ == "__main__":
    main()
