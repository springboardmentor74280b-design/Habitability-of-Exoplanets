from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    
    # Config
    basedir = os.path.abspath(os.path.dirname(__file__))
    # Database at ../instance/project.db
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, '../instance/exoplanets.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    # Register Blueprints / Routes
    from .routes import main
    app.register_blueprint(main)
    
    # Ensure instance folder exists
    try:
        os.makedirs(os.path.join(app.instance_path), exist_ok=True)
        # Also ensure the ../instance path relative to here works as expected by other parts
        os.makedirs(os.path.join(basedir, '../instance'), exist_ok=True)
    except OSError:
        pass
    
    return app
