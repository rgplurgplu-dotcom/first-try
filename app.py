import openssl
from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
app.config
app.config

app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'['CORS_HEADERS'] = 'Content-Type'                                                                               
CORS(app)


            