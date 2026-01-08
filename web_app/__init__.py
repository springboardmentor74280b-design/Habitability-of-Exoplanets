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
    
    return app
