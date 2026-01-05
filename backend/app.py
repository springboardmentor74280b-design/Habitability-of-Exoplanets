# backend/app.py
from flask import Flask
from flask_cors import CORS
from config import Config
from database import db
from routes.predict import predict_bp
from routes.ranking import ranking_bp   # ✅ already correct

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Enable CORS for all routes (allow frontend requests)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize database
    db.init_app(app)

    # Register Blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(ranking_bp)   # ✅ ADD THIS LINE

    return app

app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)
