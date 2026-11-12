"""
RF8 (Could): asigna segmento por lote a clientes nuevos de la tarjeta con el modelo guardado.

Uso:
    python src/asignar_nuevos.py --entrada nuevos.csv --salida nuevos_segmentados.csv
El archivo de entrada necesita las columnas CustomerID, Annual Income (k$) y Spending Score (1-100).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

import segmentacion as seg

RAIZ = Path(__file__).resolve().parents[1]


def asignar(nuevos: pd.DataFrame) -> pd.DataFrame:
    paquete = joblib.load(RAIZ / "models/kmeans_k5.joblib")
    df = nuevos.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=seg.RENOMBRAR)
    df.insert(0, "id_anon", df["customer_id"].map(seg.anonimizar_id))
    X = paquete["scaler"].transform(df[paquete["variables"]].to_numpy(dtype=float))
    df["segmento"] = pd.Series(paquete["modelo"].predict(X)).map(paquete["letras"]).to_numpy()
    df = seg.aplicar_regla_credito(df)
    return df[["id_anon", "segmento", "ingreso_anual_k", "spending_score", "apto_promo_credito"]]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", required=True)
    p.add_argument("--salida", required=True)
    a = p.parse_args()
    res = asignar(seg.cargar_datos(a.entrada))
    res.to_csv(a.salida, index=False)
    print(res["segmento"].value_counts().sort_index().to_string())
