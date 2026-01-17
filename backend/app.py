from flask import Flask
from flask_cors import CORS
from config import Config
from database import db
from routes.predict import predict_bp
from routes.ranking import ranking_bp 


def create_app():
    app = Flask(
        __name__,
        static_folder="../frontend",
        static_url_path=""
    )

    app.config.from_object(Config)

    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize DB
    db.init_app(app)

    # Register Blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(ranking_bp)

    # Serve frontend
    @app.route("/")
    def serve_frontend():
        return app.send_static_file("index.html")

    return app


app = create_app()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5000)
