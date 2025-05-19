"""
Script de démarrage pour ElderCare.
Lance à la fois le serveur backend API et le frontend React.
"""

import os
import sys
import subprocess
import time
import logging
import platform
import signal
import webbrowser
from threading import Thread

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ajout des répertoires au PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# S'assurer que le module chatbot est accessible
CHATBOT_DIR = os.path.join(ROOT_DIR, "chatbot")
if os.path.exists(CHATBOT_DIR) and CHATBOT_DIR not in sys.path:
    sys.path.append(CHATBOT_DIR)

# Chemins pour les processus
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BACKEND_DIR), "frontend")
API_SCRIPT = os.path.join(BACKEND_DIR, "api.py")

# Processus globaux
frontend_process = None
backend_process = None

def is_windows():
    return platform.system().lower() == "windows"

def check_prerequisites():
    """Vérifier que toutes les dépendances sont installées."""
    # Vérifier Python packages
    try:
        import flask
        import flask_cors
        import dotenv
        logger.info("Les dépendances Python sont installées.")
    except ImportError as e:
        logger.error(f"Dépendance Python manquante: {e}")
        print("\nCertaines dépendances Python sont manquantes.")
        print("Installez-les avec: pip install -r eldercare/requirements.txt")
        return False
    
    # Vérifier que le module chatbot existe
    if not os.path.exists(CHATBOT_DIR):
        logger.error(f"Le répertoire chatbot n'existe pas: {CHATBOT_DIR}")
        print(f"\nLe répertoire contenant le module ElderCare complet n'existe pas: {CHATBOT_DIR}")
        print("Assurez-vous que le répertoire 'chatbot' avec eldercare_complete.py existe.")
        return False
        
    # Vérifier que eldercare_complete.py existe
    eldercare_complete_path = os.path.join(CHATBOT_DIR, "eldercare_complete.py")
    if not os.path.exists(eldercare_complete_path):
        logger.error(f"Le fichier eldercare_complete.py n'existe pas: {eldercare_complete_path}")
        print(f"\nLe fichier ElderCare complet n'existe pas: {eldercare_complete_path}")
        print("Assurez-vous que le fichier 'eldercare_complete.py' existe dans le répertoire 'chatbot'.")
        return False
    
    # Vérifier Node.js/npm
    if not os.path.exists(FRONTEND_DIR):
        logger.error(f"Le répertoire frontend n'existe pas: {FRONTEND_DIR}")
        return False
    
    # Vérifier node_modules
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(node_modules):
        logger.warning("Les dépendances Node.js ne sont pas installées.")
        print("\nLes dépendances Node.js ne sont pas installées.")
        choice = input("Voulez-vous installer les dépendances maintenant? (o/n): ")
        if choice.lower() in ['o', 'oui', 'y', 'yes']:
            print("Installation des dépendances Node.js...")
            cmd = "npm.cmd" if is_windows() else "npm"
            try:
                subprocess.run([cmd, "install"], cwd=FRONTEND_DIR, check=True)
            except subprocess.CalledProcessError:
                logger.error("Erreur lors de l'installation des dépendances Node.js")
                print("Erreur lors de l'installation des dépendances Node.js")
                return False
        else:
            return False
    
    return True

def start_backend():
    """Démarrer le serveur API Flask."""
    logger.info("Démarrage du serveur backend API Flask...")
    
    # Configuration de l'environnement
    env = os.environ.copy()
    # Ajout des répertoires au PYTHONPATH
    python_path = [ROOT_DIR, CHATBOT_DIR]
    if "PYTHONPATH" in env:
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    
    # Exécuter le script API
    python_executable = sys.executable
    if is_windows():
        return subprocess.Popen([python_executable, API_SCRIPT], 
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                               env=env)
    else:
        return subprocess.Popen([python_executable, API_SCRIPT], 
                               preexec_fn=os.setpgrp,
                               env=env)

def start_frontend():
    """Démarrer le serveur de développement React."""
    logger.info("Démarrage du frontend React...")
    
    if is_windows():
        cmd = "npm.cmd"
    else:
        cmd = "npm"
    
    # Exécuter npm start dans le répertoire frontend
    if is_windows():
        return subprocess.Popen([cmd, "start"], 
                               cwd=FRONTEND_DIR,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        return subprocess.Popen([cmd, "start"], 
                               cwd=FRONTEND_DIR,
                               preexec_fn=os.setpgrp)

def open_browser():
    """Ouvrir le navigateur après un délai."""
    time.sleep(4)  # Attendre que les serveurs démarrent
    url = "http://localhost:3000"
    logger.info(f"Ouverture de l'application dans le navigateur: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"Erreur lors de l'ouverture du navigateur: {str(e)}")
        print(f"L'application est disponible à l'adresse: {url}")

def cleanup(signum=None, frame=None):
    """Nettoyer les processus avant de quitter."""
    print("\nArrêt des serveurs...")
    
    # Arrêter le frontend
    if frontend_process:
        logger.info("Arrêt du serveur frontend...")
        if is_windows():
            frontend_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            frontend_process.terminate()
        try:
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            frontend_process.kill()
    
    # Arrêter le backend
    if backend_process:
        logger.info("Arrêt du serveur API backend...")
        if is_windows():
            backend_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
    
    logger.info("Tous les serveurs ont été arrêtés.")
    sys.exit(0)

def main():
    """Fonction principale."""
    global frontend_process, backend_process
    
    print("\n=== ElderCare - Démarrage de l'application ===")
    
    # Vérifier les prérequis
    if not check_prerequisites():
        print("Impossible de démarrer l'application car certains prérequis ne sont pas satisfaits.")
        sys.exit(1)
    
    # Enregistrer le handler pour le signal d'interruption
    signal.signal(signal.SIGINT, cleanup)
    if is_windows():
        signal.signal(signal.SIGBREAK, cleanup)
    else:
        signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Démarrer backend
        backend_process = start_backend()
        if backend_process.poll() is not None:
            logger.error("Erreur lors du démarrage du serveur backend API.")
            print("Erreur lors du démarrage du serveur backend API.")
            sys.exit(1)
        logger.info("Serveur API backend démarré.")
        
        # Démarrer frontend
        frontend_process = start_frontend()
        if frontend_process.poll() is not None:
            logger.error("Erreur lors du démarrage du serveur frontend React.")
            print("Erreur lors du démarrage du serveur frontend React.")
            cleanup()
            sys.exit(1)
        logger.info("Serveur frontend démarré.")
        
        # Ouvrir navigateur dans un thread séparé
        browser_thread = Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("\n=== ElderCare est en cours d'exécution ===")
        print("API Backend: http://localhost:5000")
        print("Frontend:    http://localhost:3000")
        print("\nAppuyez sur Ctrl+C pour quitter.")
        
        # Attendre que les processus se terminent
        while True:
            if backend_process.poll() is not None:
                logger.error("Le serveur backend s'est arrêté de manière inattendue.")
                cleanup()
                break
            
            if frontend_process.poll() is not None:
                logger.error("Le serveur frontend s'est arrêté de manière inattendue.")
                cleanup()
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur...")
        cleanup()
    except Exception as e:
        logger.error(f"Erreur dans la boucle principale: {str(e)}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main() 