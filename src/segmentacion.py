"""
Segmentación de clientes de un centro comercial (Caso 3) - Grupo 3, SCY1102.

Funciones del pipeline CRISP-DM:
  preparar_datos()      -> Fase 3: normaliza columnas, anonimiza CustomerID (SHA-256), Unknown explícito
  evaluar_k()           -> Fase 4: codo (inercia) y Silhouette para k = 2..10
  entrenar_modelo()     -> Fase 4: K-Means final (solo ingreso y gasto, RF1 / RNF4)
  probar_estabilidad()  -> Fase 4: misma asignación con 20 semillas (RNF2)
  nombrar_segmentos()   -> etiqueta A-E según la posición de los centroides
  perfilar_segmentos()  -> Fase 5: fichas (RF3) y auditoría demográfica (RF6)
  aplicar_regla_credito -> Fase 5/6: excluye ingreso < 30 k$ de promociones de crédito (RF5)
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

VARIABLES_MODELO = ["ingreso_anual_k", "spending_score"]          # RF1: únicas variables del fit
VARIABLES_AUDITORIA = ["genero", "edad", "educacion", "estado_civil"]  # RNF4: solo perfilar/auditar
UMBRAL_CREDITO_K = 30                                               # RF5
RANDOM_STATE = 42                                                   # RNF2
N_INIT = 10

RENOMBRAR = {
    "CustomerID": "customer_id",
    "Gender": "genero",
    "Age": "edad",
    "Education": "educacion",
    "Marital Status": "estado_civil",
    "Annual Income (k$)": "ingreso_anual_k",
    "Spending Score (1-100)": "spending_score",
}

SEGMENTOS = {
    "A": "Equilibrados",
    "B": "Premium",
    "C": "Jóvenes de alto gasto",
    "D": "Alto ingreso cauteloso",
    "E": "Ahorradores",
}


def _sal() -> str:
    """La sal del hash se lee de una variable de entorno; nunca se versiona el valor real."""
    return os.environ.get("SEG_SALT", "grupo3-scy1102-demo")


def anonimizar_id(customer_id, sal: str | None = None) -> str:
    """RNF3: SHA-256 con sal del CustomerID original (12 primeros caracteres hex)."""
    sal = _sal() if sal is None else sal
    return hashlib.sha256(f"{sal}{customer_id}".encode("utf-8")).hexdigest()[:12]


def cargar_datos(ruta: str) -> pd.DataFrame:
    if ruta.endswith(".xlsx"):
        return pd.read_excel(ruta)
    return pd.read_csv(ruta)


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Fase 3. Devuelve un dataset sin el CustomerID original (G2)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]  # 'Education ' trae un espacio al final
    df = df.rename(columns=RENOMBRAR)
    assert df.isna().sum().sum() == 0, "Hay valores nulos: aplicar contingencia de datos"
    assert not df["customer_id"].duplicated().any(), "CustomerID duplicados"
    for c in ["educacion", "estado_civil"]:
        df[c] = df[c].astype(str).str.strip().replace({"": "Unknown"})  # RNF7: sin imputar
    df.insert(0, "id_anon", df["customer_id"].map(anonimizar_id))
    assert not df["id_anon"].duplicated().any(), "Colisión de hash"
    return df.drop(columns=["customer_id"])


def matriz_modelo(df: pd.DataFrame, scaler: StandardScaler | None = None):
    X = df[VARIABLES_MODELO].to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def evaluar_k(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X)
        filas.append({"k": k, "inercia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(filas)


def entrenar_modelo(X: np.ndarray, k: int = 5) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X)


def probar_estabilidad(X: np.ndarray, k: int, etiquetas_base, n_semillas: int = 20) -> pd.DataFrame:
    """RNF2: compara la asignación con 20 semillas distintas (ARI = 1 => idéntica)."""
    filas = []
    for s in range(n_semillas):
        lab = KMeans(n_clusters=k, random_state=s, n_init=N_INIT).fit_predict(X)
        filas.append({"semilla": s, "ARI": adjusted_rand_score(etiquetas_base, lab)})
    return pd.DataFrame(filas)


def nombrar_segmentos(centroides_originales: np.ndarray) -> dict[int, str]:
    """Asigna letras A-E según ingreso/gasto del centroide (independiente del orden interno de K-Means)."""
    nombres = {}
    for i, (ing, gas) in enumerate(centroides_originales):
        alto_ing, bajo_ing = ing >= 70, ing < 40
        alto_gas, bajo_gas = gas >= 65, gas < 35
        if alto_ing and alto_gas:
            nombres[i] = "B"
        elif bajo_ing and alto_gas:
            nombres[i] = "C"
        elif alto_ing and bajo_gas:
            nombres[i] = "D"
        elif bajo_ing and bajo_gas:
            nombres[i] = "E"
        else:
            nombres[i] = "A"
    assert len(set(nombres.values())) == len(nombres), "Dos centroides con la misma letra: revisar k"
    return nombres


def perfilar_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """RF3: datos de la ficha por segmento."""
    g = df.groupby("segmento")
    perfil = pd.DataFrame({
        "nombre": [SEGMENTOS[s] for s in g.size().index],
        "clientes": g.size(),
        "pct_clientes": (g.size() / len(df) * 100).round(1),
        "ingreso_prom_k": g["ingreso_anual_k"].mean().round(1),
        "gasto_prom": g["spending_score"].mean().round(1),
        "edad_prom": g["edad"].mean().round(1),
        "edad_min": g["edad"].min(),
        "edad_max": g["edad"].max(),
        "pct_mujeres": (g["genero"].apply(lambda s: (s == "F").mean()) * 100).round(1),
        "ingreso_menor_30k": g["ingreso_anual_k"].apply(lambda s: (s < UMBRAL_CREDITO_K).sum()),
        "pct_educ_unknown": (g["educacion"].apply(lambda s: (s == "Unknown").mean()) * 100).round(1),
    })
    return perfil


def auditoria(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """RF6: composición (%) de cada segmento por variable demográfica. Solo para auditar."""
    df = df.assign(tramo_edad=pd.cut(df["edad"], [17, 29, 44, 59, 120],
                                     labels=["18-29", "30-44", "45-59", "60+"]))
    salida = {}
    for var in ["genero", "tramo_edad", "educacion", "estado_civil"]:
        salida[var] = (pd.crosstab(df["segmento"], df[var], normalize="index") * 100).round(1)
    return salida


def aplicar_regla_credito(df: pd.DataFrame) -> pd.DataFrame:
    """RF5: marca a quién NO se le puede ofrecer crédito o endeudamiento."""
    df = df.copy()
    df["apto_promo_credito"] = df["ingreso_anual_k"] >= UMBRAL_CREDITO_K
    return df
