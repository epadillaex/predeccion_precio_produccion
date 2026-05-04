from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import (
    load_models,
    predict_coste,
)
from src.logger_config import setup_logger

logger = setup_logger("api.log")


########################################################################################################################
#                                            ESQUEMAS DE ENTRADA / SALIDA                                              #
########################################################################################################################

class PredictionRequest(BaseModel):
    datos: dict[str, Any] = Field(
        ...,
        description="Diccionario con las variables del producto"
    )


class PredictionResponse(BaseModel):
    ok: bool
    resultado: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


########################################################################################################################
#                                                 CICLO DE VIDA DE LA APP                                               #
########################################################################################################################

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Arrancando API y cargando modelo de costes...")

    try:
        models = load_models()

        if "coste" not in models:
            raise RuntimeError("No se encontró el modelo de costes.")

        app.state.model = models["coste"]

        logger.info("Modelo de costes cargado correctamente.")
        yield

    except Exception:
        logger.exception("Error al arrancar la API.")
        raise

    finally:
        logger.info("Cerrando API.")
        app.state.model = None


app = FastAPI(
    title="API de predicción de costes",
    description="Servicio para predecir el coste",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en desarrollo está bien
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


########################################################################################################################
#                                                      ENDPOINTS                                                       #
########################################################################################################################

@app.get("/")
def root():
    return {"message": "API de predicción de costes operativa"}


@app.get("/health", response_model=HealthResponse)
def health():
    model_loaded = getattr(app.state, "model", None) is not None

    return HealthResponse(
        status="ok",
        models_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Endpoint para predecir el coste
    """
    try:
        model = getattr(app.state, "model", None)

        if model is None:
            raise HTTPException(
                status_code=500,
                detail="El modelo no está cargado en memoria."
            )

        resultado = predict_coste(request.datos, {"coste": model})

        return PredictionResponse(
            ok=True,
            resultado=resultado
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Error no controlado en /predict")
        raise HTTPException(
            status_code=500,
            detail="Se produjo un error interno al generar la predicción."
        )