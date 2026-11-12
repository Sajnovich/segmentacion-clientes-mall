"""
Pruebas de aceptación de las reglas éticas y de calidad (puertas G3, G4 y G5).
Ejecutar desde la raíz:  python tests/test_reglas.py   (o pytest tests/)
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
import segmentacion as seg  # noqa: E402

LISTADOS = RAIZ / "outputs/listados"


def test_rnf1_calidad():
    m = json.loads((RAIZ / "outputs/metricas.json").read_text(encoding="utf-8"))
    assert m["silhouette"] >= 0.50 and m["segmento_mas_chico_pct"] >= 5


def test_rnf2_estabilidad():
    m = json.loads((RAIZ / "outputs/metricas.json").read_text(encoding="utf-8"))
    assert m["ari_min_20_semillas"] == 1.0


def test_rnf4_sin_demograficas_en_fit():
    assert set(seg.VARIABLES_MODELO).isdisjoint(seg.VARIABLES_AUDITORIA)


def test_rf5_ningun_cliente_bajo_30k_en_listados_de_credito():
    for f in LISTADOS.glob("listado_credito_*.csv"):
        assert (pd.read_csv(f)["ingreso_anual_k"] >= seg.UMBRAL_CREDITO_K).all(), f.name


def test_rf5_30_casos_excluidos():
    m = json.loads((RAIZ / "outputs/metricas.json").read_text(encoding="utf-8"))
    assert m["clientes_excluidos_credito"] == 30


def test_rnf3_sin_identificadores_originales():
    for f in list(LISTADOS.glob("*.csv")) + [RAIZ / "data/processed/clientes_preparados.csv"]:
        cols = pd.read_csv(f, nrows=1).columns
        assert "customer_id" not in cols and "CustomerID" not in cols, f.name


def test_rf1_todos_los_clientes_con_segmento():
    df = pd.read_csv(RAIZ / "outputs/clientes_segmentados.csv")
    assert len(df) == 200 and df["segmento"].notna().all()


if __name__ == "__main__":
    pruebas = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in pruebas:
        t()
        print(f"OK  {t.__name__}")
    print(f"{len(pruebas)} pruebas aprobadas")
