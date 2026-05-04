from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.logger_config import setup_logger
from src.utils_inference import enforce_column_types

logger = setup_logger("inference.log")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


########################################################################################################################
#                                                   CONFIGURACIÓN                                                      #
########################################################################################################################

MODEL_CONFIG = {
    "coste": {
        "model_filename": "modelo_costes.pkl",
        "target_name": "Precio_USD",
    }
}

# ⚠️ DEBE SER IGUAL QUE EN TRAINING
COLUMN_TYPES = {
    "Marca": "category",
    "Tipo": "category",
    "Disipador": "category",
    "Capacidad_GB": "Int64",
    "Velocidad_MHz": "Int64",
    "Latencia_CAS": "Int64",
    "Modulos": "Int64",
    "Voltaje": "float64",
    "Precio_USD": "float64",
    "RGB": "boolean",
}


########################################################################################################################
#                                                   CARGA DE MODELOS                                                   #
########################################################################################################################

def _load_pickle(path: Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {path}")

    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)

        logger.info(f"Modelo cargado correctamente: {path}")
        return obj

    except Exception:
        logger.exception(f"Error al cargar el modelo: {path}")
        raise


def load_models() -> dict[str, Any]:
    logger.info("Cargando modelos...")

    models = {}

    for model_type, config in MODEL_CONFIG.items():
        model_path = MODELS_DIR / config["model_filename"]
        models[model_type] = _load_pickle(model_path)

    logger.info(f"Modelos cargados: {list(models.keys())}")
    return models


########################################################################################################################
#                                                      HELPERS                                                         #
########################################################################################################################

def _normalize_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = pd.Index(df.columns).map(lambda col: str(col).strip())
    return df


def _get_expected_columns_from_pipeline(pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessing"]

    columns = []

    for _, transformer, cols in preprocessor.transformers_:
        if transformer == "drop" or cols is None:
            continue

        if isinstance(cols, (str, int)):
            columns.append(str(cols))
        else:
            columns.extend([str(c) for c in list(cols)])

    return list(dict.fromkeys(columns))


def _prepare_input(raw_data: dict[str, Any], pipeline) -> pd.DataFrame:

    if not isinstance(raw_data, dict):
        raise TypeError("raw_data debe ser un diccionario.")

    # 1. Crear DataFrame
    df = pd.DataFrame([raw_data])
    df = _normalize_columns_to_str(df)

    # 2. Columnas esperadas
    expected_cols = _get_expected_columns_from_pipeline(pipeline)

    # 3. Añadir faltantes
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 4. Filtrar orden correcto
    df = df[expected_cols].copy()

    # 5. Tipado
    column_types = {
        col: dtype for col, dtype in COLUMN_TYPES.items()
        if col in df.columns
    }

    df = enforce_column_types(df, column_types, logger=logger)

    logger.info(
        f"Input preparado | shape={df.shape} | columnas={len(expected_cols)}"
    )

    return df


########################################################################################################################
#                                                     PREDICCIONES                                                     #
########################################################################################################################

def predict_coste(raw_data: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    pipeline = models["coste"]

    df = _prepare_input(raw_data, pipeline)

    pred = pipeline.predict(df)[0]
    pred = float(pred)

    if pd.isna(pred):
        raise ValueError("Predicción nula")

    return {"prediccion_coste": round(pred, 2)}


def predict_complejidad(raw_data: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    pipeline = models["complejidad"]

    df = _prepare_input(raw_data, pipeline)

    pred_class = pipeline.predict(df)[0]

    if pd.isna(pred_class):
        raise ValueError("Predicción nula")

    result = {
        "prediccion_complejidad": int(pred_class)
    }

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df)[0]

        result["probabilidades"] = {
            str(i): round(float(p), 4)
            for i, p in enumerate(probs)
        }

    return result


def predict_both(raw_data: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    return {
        **predict_coste(raw_data, models),
        **predict_complejidad(raw_data, models),
    }