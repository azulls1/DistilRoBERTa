"""Configuración leída del entorno. El DSN de la base viene de un secreto de Docker Swarm."""
import os
from functools import lru_cache
from pathlib import Path


def _leer_secreto(variable: str, archivo_por_defecto: str) -> str:
    """Devuelve la variable de entorno o, si no existe, el contenido del archivo de secreto."""
    if valor := os.getenv(variable):
        return valor
    ruta = Path(os.getenv(f"{variable}_FILE", archivo_por_defecto))
    return ruta.read_text().strip() if ruta.exists() else ""


class Config:
    def __init__(self) -> None:
        self.database_url = _leer_secreto("DATABASE_URL", "/run/secrets/distilroberta_db_dsn")
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.redis_resultados = os.getenv("REDIS_RESULT_URL", "redis://redis:6379/1")
        self.ruta_modelo = os.getenv("MODELO_DIR", "/app/modelo")
        self.cors = [o.strip() for o in os.getenv(
            "CORS_ORIGINS", "https://distilroberta.iagentek.com.mx,http://localhost:4200").split(",") if o.strip()]
        self.max_caracteres = 512


@lru_cache
def config() -> Config:
    return Config()
