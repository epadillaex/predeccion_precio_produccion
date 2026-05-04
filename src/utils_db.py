import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from dotenv import load_dotenv

########################################################################################################################
#                                                    CONEXIÓN BBDD                                                     #
########################################################################################################################

def _get_db_config() -> dict:
    """
    Carga la configuración de conexión desde variables de entorno.
    """
    load_dotenv()

    config = {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "server": os.getenv("DB_SERVER"),
        "database": os.getenv("DB_NAME"),
        "driver": os.getenv("DB_DRIVER"),
    }

    missing = [key for key, value in config.items() if not value]
    if missing:
        raise ValueError(
            f"Faltan variables de entorno para la conexión a SQL Server: {missing}"
        )

    return config


def get_sql_server_engine(logger=None) -> Engine:
    """
    Crea y devuelve un engine de SQLAlchemy para SQL Server.
    """
    config = _get_db_config()

    if logger:
        logger.info(
            f"Creando conexión a la base de datos '{config['database']}' "
            f"en el servidor '{config['server']}'"
        )

    connection_url = URL.create(
        "mssql+pyodbc",
        username=config["user"],
        password=config["password"],
        host=config["server"],
        database=config["database"],
        query={
            "driver": config["driver"],
            "TrustServerCertificate": "yes"
        }
    )

    engine = create_engine(connection_url)
    return engine

########################################################################################################################
#                                                 LECTURA DE TABLAS                                                    #
########################################################################################################################

def get_table_as_dataframe(table_name: str, schema: str | None = None, logger=None) -> pd.DataFrame:
    """
    Conecta a SQL Server y devuelve el contenido de una tabla como DataFrame.

    Parámetros:
    - table_name: nombre de la tabla
    - schema: esquema opcional, por ejemplo 'dbo'
    """
    engine = None

    try:
        engine = get_sql_server_engine(logger=logger)

        full_table_name = f"{schema}.{table_name}" if schema else table_name

        if logger:
            logger.info(f"Leyendo tabla '{full_table_name}'")

        query = text(f"SELECT * FROM {full_table_name}")

        with engine.connect() as connection:
            df = pd.read_sql(query, connection)

        if logger:
            logger.info(
                f"Tabla '{full_table_name}' cargada correctamente. "
                f"Shape: {df.shape}"
            )

        return df

    except Exception as e:
        if logger:
            logger.exception(f"Error al obtener la tabla '{table_name}'")
        raise RuntimeError(f"Error al obtener la tabla '{table_name}': {e}") from e

    finally:
        if engine is not None:
            engine.dispose()
