"""
Pipeline end-to-end: datos crudos -> dataset preparado -> modelo -> listados y fichas.

Uso (desde la raíz del repositorio):
    python src/pipeline.py --entrada data/raw/Mall_Customers.xlsx --k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import silhouette_score

import segmentacion as seg

RAIZ = Path(__file__).resolve().parents[1]


def main(entrada: str, k: int) -> dict:
    # Fase 3 - Preparación (G2)
    crudo = seg.cargar_datos(entrada)
    df = seg.preparar_datos(crudo)
    (RAIZ / "data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(RAIZ / "data/processed/clientes_preparados.csv", index=False)

    # Fase 4 - Modelado (G3)
    X, scaler = seg.matriz_modelo(df)
    curvas = seg.evaluar_k(X)
    modelo = seg.entrenar_modelo(X, k)
    estabilidad = seg.probar_estabilidad(X, k, modelo.labels_)
    centroides = scaler.inverse_transform(modelo.cluster_centers_)
    letras = seg.nombrar_segmentos(centroides)
    df["segmento"] = pd.Series(modelo.labels_).map(letras).to_numpy()

    # Fase 5 - Evaluación (G4)
    df = seg.aplicar_regla_credito(df)
    perfil = seg.perfilar_segmentos(df)
    sil = silhouette_score(X, modelo.labels_)
    metricas = {
        "k": k,
        "silhouette": round(float(sil), 3),
        "segmento_mas_chico_pct": float(perfil["pct_clientes"].min()),
        "ari_min_20_semillas": round(float(estabilidad["ARI"].min()), 3),
        "cumple_RNF1": bool(sil >= 0.50 and perfil["pct_clientes"].min() >= 5),
        "cumple_RNF2": bool(estabilidad["ARI"].min() == 1.0),
        "clientes_excluidos_credito": int((~df["apto_promo_credito"]).sum()),
        "variables_fit": seg.VARIABLES_MODELO,
    }

    # Fase 6 - Despliegue: modelo y listados (RF4, RF5, RF8)
    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump({"modelo": modelo, "scaler": scaler, "letras": letras,
                 "variables": seg.VARIABLES_MODELO}, RAIZ / "models/kmeans_k5.joblib")
    out = RAIZ / "outputs/listados"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id_anon", "segmento", "ingreso_anual_k", "spending_score", "apto_promo_credito"]
    for letra, grupo in df.groupby("segmento"):
        grupo[cols].to_csv(out / f"listado_segmento_{letra}.csv", index=False)
        grupo.loc[grupo["apto_promo_credito"], cols].to_csv(
            out / f"listado_credito_segmento_{letra}.csv", index=False)
    perfil.to_csv(RAIZ / "outputs/fichas_segmentos.csv")
    curvas.to_csv(RAIZ / "outputs/curvas_k.csv", index=False)
    df.to_csv(RAIZ / "outputs/clientes_segmentados.csv", index=False)
    with open(RAIZ / "outputs/metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    return metricas


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", default=str(RAIZ / "data/raw/Mall_Customers.xlsx"))
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    print(json.dumps(main(a.entrada, a.k), ensure_ascii=False, indent=2))
