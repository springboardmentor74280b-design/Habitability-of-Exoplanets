import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "exohabit-secret")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///exoplanets.db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
