from fastapi import FastAPI
from predict import router as predict_router

app = FastAPI(
    title="Exoplanet Habitability Prediction API",
    version="1.0.0"
)


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",  "http://localhost:5173"],  # during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Exoplanet Habitability Prediction API"
    }

# Include ML prediction routes
app.include_router(predict_router, prefix="/api")

