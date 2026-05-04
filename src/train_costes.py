from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from logger_config import setup_logger
from utils_db import get_table_as_dataframe
from utils_inference import (
    drop_unnecessary_columns,
    drop_null_rows_by_column,
    enforce_column_types,
    split_features_target,
    wape,
    save_model_as_pkl,
)

TARGET_COL = "Precio_USD"
TABLE_NAME = "dbo.ram_price_dataset_2000"
COLUMN_TYPES = {"Marca": "category","Tipo": "category", 
                "Disipador": "category","Capacidad_GB": "Int64",
                "Velocidad_MHz": "Int64","Latencia_CAS": "Int64",
                "Modulos": "Int64","Voltaje": "float64",
                "Precio_USD": "float64","RGB": "boolean"}
RANDOM_STATE = 42


def evaluate_regression(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    wape_value = wape(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "wape": float(wape_value),
    }

def log_and_print_metrics(title: str, metrics: dict, logger) -> None:
    print(f"\n{title}")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"WAPE: {metrics['wape']:.2f}%")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R²:   {metrics['r2']:.3f}")

    logger.info(
        f"{title} | "
        f"MAE={metrics['mae']:.2f} | "
        f"WAPE={metrics['wape']:.2f}% | "
        f"RMSE={metrics['rmse']:.2f} | "
        f"R2={metrics['r2']:.3f}"
    )


def build_preprocessor(X: pd.DataFrame, logger):
    cat_vars = X.select_dtypes(include=["category"]).columns.tolist()
    bool_vars = X.select_dtypes(include=["boolean"]).columns.tolist()
    num_vars = X.select_dtypes(include=["float64", "float32"]).columns.tolist()
    int_vars = X.select_dtypes(include=["Int64", "int64", "Int32", "int32"]).columns.tolist()

    if logger:
        logger.info(
            f"Variables detectadas | "
            f"numéricas: {len(num_vars)} | "
            f"enteras: {len(int_vars)} | "
            f"booleanas: {len(bool_vars)} | "
            f"categóricas: {len(cat_vars)}"
        )

    transformers = []

    # Numéricas float
    if num_vars:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean"))
        ])
        transformers.append(("num", num_pipeline, num_vars))

    # Enteras → tratadas igual que numéricas (pero con mediana)
    if int_vars:
        int_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("int", int_pipeline, int_vars))

    # Booleanas
    if bool_vars:
        bool_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])
        transformers.append(("bin", bool_pipeline, bool_vars))

    # Categóricas
    if cat_vars:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", cat_pipeline, cat_vars))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_pipeline(X: pd.DataFrame, logger) -> Pipeline:
    preprocessor = build_preprocessor(X, logger)

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_estimators=3000,
        max_depth=7,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=2,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessor),
        ("xgb", model),
    ])


def save_feature_importances(pipeline: Pipeline, output_path: Path, logger) -> None:
    feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    importances = pipeline.named_steps["xgb"].feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Longitud distinta: feature_names={len(feature_names)}, "
            f"importances={len(importances)}"
        )

    importancias = (
        pd.DataFrame({
            "variable": feature_names,
            "importancia": importances
        })
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importancias.to_excel(output_path, index=False)

    logger.info(f"Importancias guardadas en: {output_path}")
    print(f"Archivo de importancias guardado en: {output_path}")


def save_scatter_plot(y_true, y_pred, output_path: Path, logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title("Predicción vs Real")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Gráfico de dispersión guardado en: {output_path}")


def main():
    logger = setup_logger("train_costes.log")
    logger.info("Inicio del entrenamiento del modelo de costes")

    try:
        project_root = Path(__file__).resolve().parent.parent
        # data_dir = project_root / "data"
        output_dir = project_root / "outputs"

########################################################################################################################
#                                                    CARGA DE DATOS                                                    #
########################################################################################################################
        df = get_table_as_dataframe(TABLE_NAME, logger=logger)

########################################################################################################################
#                                               PREPARACIÓN DE LOS DATOS                                               #
########################################################################################################################
        columns_to_drop = ["Id"]
        df = drop_unnecessary_columns(df, columns_to_drop, logger=logger)
        df = drop_null_rows_by_column(df, TARGET_COL, logger=logger)

        df = enforce_column_types(df, COLUMN_TYPES, logger=logger)
        # df = unify_text_case(df, "MARCA_C_C_MOLDE", case="upper", logger=logger)
        # df = fill_nulls_in_boolean_columns(df, logger=logger)
        # df = fill_empty_categorical_with_missing(df, logger=logger)
        # df, int_all_null_cols, int_partial_cols = fill_all_null_int64_with_zero(df, logger=logger)

        X, y = split_features_target(df, TARGET_COL, logger=logger)

        # Blindaje adicional
        X.columns = X.columns.astype(str)
        # y = pd.to_numeric(y, errors="coerce")

        valid_idx = y.notna()
        X = X.loc[valid_idx].copy()
        y = y.loc[valid_idx].astype(float)

        logger.info(f"Dataset final para entrenamiento | X: {X.shape} | y: {y.shape}")

########################################################################################################################
#                                                   TRAIN / VALIDACIÓN / TEST                                          #
########################################################################################################################
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE
        )

        X_train, X_eval, y_train, y_eval = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.25,
            random_state=RANDOM_STATE
        )

        logger.info(
            f"Particiones | train: {X_train.shape} | eval: {X_eval.shape} | test: {X_test.shape}"
        )

        # Modelo provisional para validación
        eval_pipeline = build_pipeline(X_train, logger)
        eval_pipeline.fit(X_train, y_train)

        y_eval_pred = eval_pipeline.predict(X_eval)
        eval_metrics = evaluate_regression(y_eval, y_eval_pred)
        log_and_print_metrics("VALIDACIÓN", eval_metrics, logger)

        # Modelo final reentrenado con train+eval (opcional, lo puedes dejar comentado)
        # final_pipeline = build_pipeline(X_train_val, logger)
        # final_pipeline.fit(X_train_val, y_train_val)

        # y_test_pred = final_pipeline.predict(X_test)
        # test_metrics = evaluate_regression(y_test, y_test_pred)
        # log_and_print_metrics("TEST", test_metrics, logger)

########################################################################################################################
#                                                           SALIDAS                                                    #
########################################################################################################################
        # save_feature_importances(
        #     final_pipeline,
        #     output_dir / "importancia_variables_coste.xlsx",
        #     logger
        # )

        # save_scatter_plot(
        #     y_test,
        #     y_test_pred,
        #     output_dir / "prediccion_vs_real_coste_test.png",
        #     logger
        # )

        # save_model_as_pkl(final_pipeline, "modelo_costes.pkl", logger=logger)

        logger.info("Entrenamiento del modelo de costes finalizado correctamente.")

    except Exception:
        logger.exception("Se produjo un error no controlado en train_costes.py")
        raise


if __name__ == "__main__":
    main()