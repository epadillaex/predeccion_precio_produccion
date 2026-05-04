import os
import logging

def setup_logger(log_file_name: str) -> logging.Logger:
    """
    Configura y devuelve un logger que escribe en consola y en archivo.
    Los archivos .log se guardan en la carpeta /logs del proyecto.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))   # .../src
    project_root = os.path.dirname(current_dir)                # .../GPML
    logs_dir = os.path.join(project_root, "logs")

    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, log_file_name)

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
