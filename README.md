# Segmentación de clientes de un centro comercial (Caso 3)

Grupo 3 · SCY1102 Gestión de Proyectos de Datos · Sección 002V · Evaluación Parcial N.º 5
Alejandro Cáceres (Jefe de proyecto) · Cristian Chicaisa (Data Engineer) · Eduardo Sajnovich (Data Scientist) · Stanley Joseph (Analista BI y despliegue)

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sajnovich/segmentacion-clientes-mall/blob/main/notebooks/EP5_Segmentacion_Clientes.ipynb)
[![Pipeline y pruebas](https://github.com/Sajnovich/segmentacion-clientes-mall/actions/workflows/pipeline.yml/badge.svg)](https://github.com/Sajnovich/segmentacion-clientes-mall/actions/workflows/pipeline.yml)

Segmentación con K-Means (k = 5) de 200 clientes usando **solo** ingreso anual y Spending Score.
Las variables demográficas se usan únicamente para perfilar y auditar.

## Resultados
| Indicador | Meta | Resultado |
|---|---|---|
| Silhouette (RNF1) | ≥ 0,50 | **0,555** |
| Segmento más chico (RNF1) | ≥ 5 % | **11 %** (C, 22 clientes) |
| Estabilidad con 20 semillas (RNF2) | asignación idéntica | **ARI = 1,000** |
| Validación 80/20 (160 entrenamiento, 40 prueba) | mismo segmento | **40 de 40** |
| Clientes < 30 k$ fuera de listados de crédito (RF5) | 30 | **30** (0 incumplimientos) |
| CustomerID original en datos procesados y listados (RNF3) | 0 | **0** |

| Segmento | Clientes | Ingreso prom. (k$) | Gasto prom. |
|---|---|---|---|
| A · Equilibrados | 81 | 55,3 | 49,5 |
| B · Premium | 39 | 86,5 | 82,1 |
| C · Jóvenes de alto gasto | 22 | 25,7 | 79,4 |
| D · Alto ingreso cauteloso | 35 | 88,2 | 17,1 |
| E · Ahorradores | 23 | 26,3 | 20,9 |

## Estructura
```
data/raw/                 dataset original del caso (Mall_Customers.xlsx)
data/processed/           dataset preparado y anonimizado (SHA-256 con sal)  -> E3 / G2
notebooks/                EP5_Segmentacion_Clientes.ipynb (fases 2 a 6, ejecutado)
src/segmentacion.py       funciones del pipeline
src/pipeline.py           ejecución completa                                  -> E4
src/asignar_nuevos.py     asignación por lote de clientes nuevos (RF8)
tests/test_reglas.py      7 pruebas de aceptación (RNF1-4, RF1, RF5)
models/kmeans_k5.joblib   modelo + escalador guardados
outputs/                  fichas, métricas, curvas, listados por segmento, figuras -> E5 / E6
dashboard/dashboard.html  dashboard interactivo de segmentos (RF7)
gestion/                  tablero de control: backlog, Kanban, Gantt, valor ganado, reportes, desviaciones
.github/workflows/        automatización: pipeline y pruebas en cada cambio
```

## Cómo ejecutar
En Google Colab: abrir el notebook con el botón de arriba y elegir **Entorno de ejecución > Ejecutar todas**.

En un computador:
```bash
pip install -r requirements.txt
python src/pipeline.py                # regenera modelo, listados y métricas
python tests/test_reglas.py           # pruebas de aceptación
python src/asignar_nuevos.py --entrada nuevos.csv --salida nuevos_segmentados.csv
```
El dashboard se abre directamente en el navegador (`dashboard/dashboard.html`), sin servidor.
La sal del hash se define con la variable de entorno `SEG_SALT` (el valor de producción no se versiona).
