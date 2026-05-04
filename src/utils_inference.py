import pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype
from pathlib import Path



########################################################################################################################
#                                                       LIMPIEZA                                                       #
########################################################################################################################


def drop_unnecessary_columns(df, columns_to_drop, logger=None):
    """
    Elimina del DataFrame las columnas indicadas en columns_to_drop
    que realmente existan en df.
    """
    df = df.copy()
    initial_shape = df.shape

    # Normalizar entrada por seguridad
    columns_to_drop = [str(col) for col in columns_to_drop]

    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]

    df_result = df.drop(columns=existing_cols)

    if logger:
        logger.info(
            f"Eliminadas {len(existing_cols)} columnas. "
            f"Shape antes: {initial_shape} | Shape después: {df_result.shape}"
        )

        if existing_cols:
            logger.info(f"Columnas eliminadas: {existing_cols}")

        if missing_cols:
            logger.info(f"Columnas no encontradas y omitidas: {missing_cols}")

    return df_result


def drop_null_rows_by_column(df, column_name, logger=None):
    """
    Elimina las filas con valores nulos en una columna concreta.

    Pensada especialmente para entrenamiento, cuando se necesita asegurar
    que la variable objetivo no contiene nulos.
    """
    df = df.copy()
    column_name = str(column_name)

    if column_name not in df.columns:
        message = f"La columna '{column_name}' no existe en el DataFrame."
        if logger:
            logger.error(message)
        raise ValueError(message)

    initial_rows = len(df)
    null_count = df[column_name].isna().sum()

    df_result = df.dropna(subset=[column_name]).copy()
    final_rows = len(df_result)
    removed_rows = initial_rows - final_rows

    if logger:
        logger.info(
            f"Eliminadas {removed_rows} filas con null en '{column_name}'. "
            f"Filas antes: {initial_rows} | Filas después: {final_rows}"
        )

        if null_count != removed_rows:
            logger.warning(
                f"Se detectaron {null_count} nulos en '{column_name}', "
                f"pero se eliminaron {removed_rows} filas."
            )

    return df_result


def enforce_column_types(df, column_types_dict, logger=None):
    """
    Convierte las columnas del DataFrame a los tipos especificados en column_types_dict.
    """
    df = df.copy()

    for col, expected_type in column_types_dict.items():
        if col not in df.columns:
            if logger:
                logger.warning(f"La columna '{col}' no existe en el DataFrame. Se omite.")
            continue

        expected_type = str(expected_type).strip()
        current_type = str(df[col].dtype)

        try:
            # Datetime
            if expected_type == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")

            # Enteros (respetando tipo exacto)
            elif expected_type in ("Int64", "int64", "Int32", "int32"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].astype(expected_type)

            # Floats
            elif expected_type in ("float64", "float32", "float"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].astype("float32" if expected_type == "float32" else "float64")

            # Booleanos (nullable)
            elif expected_type == "boolean":
                df[col] = df[col].astype("boolean")

            # Categóricos
            elif expected_type == "category":
                df[col] = df[col].astype("category")

            # Strings
            elif expected_type in ("string", "object"):
                df[col] = df[col].astype("string")

            # Otros tipos
            else:
                df[col] = df[col].astype(expected_type)

            if logger:
                logger.info(
                    f"Columna '{col}' convertida de '{current_type}' a '{expected_type}'"
                )

        except Exception as e:
            if logger:
                logger.exception(
                    f"Error al convertir la columna '{col}' a '{expected_type}'"
                )
            raise TypeError(
                f"Error al convertir la columna '{col}' a '{expected_type}'. "
                f"Valores de ejemplo: {df[col].head(5).tolist()}"
            ) from e

    return df
# def fill_nulls_in_boolean_columns(df, logger=None):
#     """
#     Detecta columnas booleanas y reemplaza sus nulos por False,
#     manteniendo el tipo booleano.
#     """
#     df = df.copy()

#     boolean_cols = [col for col in df.columns if is_bool_dtype(df[col])]

#     if logger:
#         logger.info(f"Se han detectado {len(boolean_cols)} columnas booleanas.")

#     for col in boolean_cols:
#         null_count = df[col].isna().sum()
#         original_dtype = str(df[col].dtype)

#         df[col] = df[col].fillna(False).astype("boolean")

#         if logger:
#             logger.info(
#                 f"Columna booleana '{col}': {null_count} nulos reemplazados por False. "
#                 f"Tipo final: {df[col].dtype} (antes: {original_dtype})"
#             )

#     return df

# def fill_empty_categorical_with_missing(df, logger=None):
#     """
#     Rellena con 'MISSING' los valores vacíos o nulos de las columnas categóricas.
#     """
#     df = df.copy()

#     cat_cols = df.select_dtypes(include=["category"]).columns.tolist()

#     if logger:
#         logger.info(f"Se han detectado {len(cat_cols)} columnas categóricas.")

#     for col in cat_cols:
#         null_count = df[col].isna().sum()

#         # Convertimos vacíos tipo "" en NaN
#         df[col] = df[col].replace("", pd.NA)

#         # Añadimos "MISSING" como categoría si no existe
#         if "MISSING" not in df[col].cat.categories:
#             df[col] = df[col].cat.add_categories(["MISSING"])

#         # Rellenamos nulos
#         df[col] = df[col].fillna("MISSING")

#         if logger:
#             final_null_count = df[col].isna().sum()
#             logger.info(
#                 f"Columna categórica '{col}': "
#                 f"{null_count} nulos detectados, {final_null_count} nulos restantes."
#             )

#     return df

# def fill_all_null_int64_with_zero(df, logger=None):
#     """
#     Rellena con 0 únicamente las columnas Int64 que estén completamente a null.
#     Las columnas Int64 con datos parciales se dejan tal cual para que el pipeline
#     pueda imputarlas después con la mediana.
#     """
#     df = df.copy()

#     int_cols = df.select_dtypes(include=["Int64"]).columns.tolist()

#     if logger:
#         logger.info(f"Se han detectado {len(int_cols)} columnas Int64.")

#     all_null_cols = []
#     partial_null_cols = []

#     for col in int_cols:
#         null_count = df[col].isna().sum()

#         if null_count == len(df):
#             df[col] = df[col].fillna(0).astype("Int64")
#             all_null_cols.append(col)

#             if logger:
#                 logger.info(
#                     f"Columna Int64 '{col}' completamente nula. "
#                     f"Se ha rellenado con 0."
#                 )
#         else:
#             partial_null_cols.append(col)

#             if logger:
#                 logger.info(
#                     f"Columna Int64 '{col}' con {null_count} nulos. "
#                     f"Se deja para imputación posterior."
#                 )

#     return df, all_null_cols, partial_null_cols

# from pandas.api.types import is_categorical_dtype


# def unify_text_case(df, columns, case="lower", logger=None):
#     """
#     Unifica mayúsculas/minúsculas en una o varias columnas de texto.
#     """
#     df = df.copy()

#     allowed_cases = {"lower", "upper", "title"}
#     case = str(case).strip().lower()

#     if case not in allowed_cases:
#         raise ValueError("El parámetro 'case' debe ser 'lower', 'upper' o 'title'.")

#     if isinstance(columns, str):
#         columns = [columns]

#     columns = [str(col) for col in columns]

#     for col in columns:
#         if col not in df.columns:
#             if logger:
#                 logger.warning(f"La columna '{col}' no existe en el DataFrame. Se omite.")
#             continue

#         original_dtype = df[col].dtype
#         null_count = df[col].isna().sum()

#         # Pasamos temporalmente a string para poder usar .str
#         series = df[col].astype("string")

#         # Limpiar espacios laterales
#         series = series.str.strip()

#         if case == "lower":
#             series = series.str.lower()
#         elif case == "upper":
#             series = series.str.upper()
#         elif case == "title":
#             series = series.str.title()

#         # Si la columna original era categórica, la devolvemos a category
#         if is_categorical_dtype(original_dtype):
#             df[col] = series.astype("category")
#         else:
#             df[col] = series

#         if logger:
#             logger.info(
#                 f"Columna '{col}' normalizada a formato '{case}'. "
#                 f"Tipo original: '{original_dtype}'. Nulos: {null_count}."
#             )

#     return df

def split_features_target(df, target_column, logger=None):
    """
    Separa el DataFrame en variables predictoras (X) y variable objetivo (y).
    """
    target_column = str(target_column)

    if target_column not in df.columns:
        message = f"La columna target '{target_column}' no existe en el DataFrame."
        if logger:
            logger.error(message)
        raise ValueError(message)

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    if logger:
        logger.info(
            f"Separación realizada correctamente. "
            f"Target: '{target_column}' | X shape: {X.shape} | y shape: {y.shape}"
        )

    return X, y


def wape(y_true, y_pred):
    """
    Calcula el WAPE (Weighted Absolute Percentage Error) en porcentaje.

    Fórmula:
        WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true y y_pred deben tener la misma forma. "
            f"Recibido: {y_true.shape} vs {y_pred.shape}"
        )

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        raise ValueError(
            "No se puede calcular WAPE porque la suma de |y_true| es 0."
        )

    return np.sum(np.abs(y_true - y_pred)) / denominator * 100

########################################################################################################################
#                                                    GUARDADO DEL MODELO                                               #
########################################################################################################################

def save_model_as_pkl(model, file_name, logger=None):
    """
    Guarda un modelo serializado en la carpeta /models del proyecto.
    """
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    file_name = str(file_name).strip()

    if not file_name.endswith(".pkl"):
        file_name += ".pkl"

    model_path = models_dir / file_name

    try:
        with open(model_path, "wb") as file:
            pickle.dump(model, file)

        if logger:
            logger.info(f"Modelo guardado correctamente en: {model_path}")

        return str(model_path)

    except Exception:
        if logger:
            logger.exception(f"Error al guardar el modelo en: {model_path}")
        raise