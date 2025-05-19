"""
ElderCare API Server.
Ce script expose les fonctionnalités d'ElderCare via une API REST.
"""

import os
import sys
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from dotenv import load_dotenv

# Ajouter le chemin parent au sys.path pour trouver les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Importer la version complète depuis le dossier chatbot
from chatbot.eldercare_complete import ElderCareComplete

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Créer l'application Flask
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Permettre les requêtes cross-origin pour le développement

# Initialiser l'assistant ElderCare
elder_care = ElderCareComplete()

# Variable pour indiquer si l'assistant est initialisé
is_initialized = False

def async_to_sync(async_func):
    """Convertit une fonction asynchrone en fonction synchrone pour Flask."""
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    wrapper.__name__ = async_func.__name__
    return wrapper

@app.route("/api/health", methods=["GET"])
def health_check():
    """Vérifier si l'API est en fonctionnement."""
    return jsonify({"status": "ok", "initialized": is_initialized})


@app.route("/api/initialize", methods=["POST"])
@async_to_sync
async def initialize_assistant():
    """Initialiser l'assistant ElderCare."""
    global is_initialized
    
    try:
        success = await elder_care.initialize()
        is_initialized = success
        
        if success:
            return jsonify({"status": "success", "message": "Assistant initialized"})
        else:
            return jsonify({"status": "error", "message": "Failed to initialize assistant"}), 500
    except Exception as e:
        logger.error(f"Error initializing assistant: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500


@app.route("/api/message", methods=["POST"])
@async_to_sync
async def process_message():
    """Traiter un message de l'utilisateur."""
    global is_initialized
    
    if not is_initialized:
        # Auto-initialiser si nécessaire
        try:
            success = await elder_care.initialize()
            is_initialized = success
            if not success:
                return jsonify({"status": "error", "message": "Failed to auto-initialize assistant"}), 500
        except Exception as e:
            logger.error(f"Error auto-initializing assistant: {str(e)}")
            return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500
    
    data = request.json
    user_message = data.get("message")
    
    if not user_message:
        return jsonify({"status": "error", "message": "No message provided"}), 400
    
    try:
        # Traiter le message utilisateur
        response = await elder_care.process_message(user_message)
        return jsonify({
            "status": "success", 
            "response": response,
            # Optionnellement retourner l'historique de conversation
            "conversation": [
                {"role": item["role"], "content": item["content"]} 
                for item in elder_care.conversation_history[-10:]
            ]
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500


@app.route("/api/user_profile", methods=["GET"])
def get_user_profile():
    """Obtenir le profil utilisateur."""
    global is_initialized
    
    if not is_initialized:
        try:
            # Initialiser globalement si nécessaire
            asyncio.run(elder_care.initialize())
            is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing for profile: {str(e)}")
            return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500
    
    try:
        return jsonify({
            "status": "success",
            "profile": elder_care.user_profile
        })
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500


@app.route("/api/user_profile", methods=["PUT"])
def update_user_profile():
    """Mettre à jour le profil utilisateur."""
    if not is_initialized:
        return jsonify({"status": "error", "message": "Assistant not initialized"}), 400
    
    data = request.json
    profile_updates = data.get("profile", {})
    
    try:
        # Mettre à jour seulement les champs fournis
        for key, value in profile_updates.items():
            if key in elder_care.user_profile:
                elder_care.user_profile[key] = value
        
        return jsonify({
            "status": "success",
            "profile": elder_care.user_profile
        })
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500


def run_api(host="0.0.0.0", port=5000):
    """Exécuter le serveur API."""
    global is_initialized
    
    # Initialiser l'assistant au démarrage du serveur
    try:
        asyncio.run(elder_care.initialize())
        is_initialized = True
        logger.info("Assistant initialized successfully at startup")
    except Exception as e:
        logger.error(f"Failed to initialize assistant at startup: {str(e)}")
    
    # Démarrer le serveur Flask
    print(f"API server running at http://{host}:{port}")
    app.run(host=host, port=port, debug=True, threaded=True)


if __name__ == "__main__":
    print("Starting ElderCare API server...")
    run_api() 