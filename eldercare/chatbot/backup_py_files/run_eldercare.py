"""
Script pour lancer la version complète de l'ElderCare Assistant.
Ce script tente d'abord d'utiliser Azure LLM, puis se rabat sur d'autres options si nécessaire.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Lance l'ElderCare Assistant complet."""
    print("\n=== Bienvenue à ElderCare Assistant (Version Complète) ===\n")
    print("Vérification de la configuration...\n")
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Vérifier quelles options sont disponibles
    has_github_token = bool(os.environ.get("GITHUB_TOKEN"))
    has_gemini_key = bool(os.environ.get("GEMINI_API_KEY"))
    
    print("Options disponibles :")
    print(f"- Azure LLM : {'Oui' if has_github_token else 'Non'}")
    print(f"- Gemini    : {'Oui' if has_gemini_key else 'Non'}")
    print(f"- Local     : Oui (toujours disponible)\n")
    
    # Choix du modèle à utiliser
    if not has_github_token and not has_gemini_key:
        print("Aucune clé API trouvée. Utilisation du mode local uniquement.")
        print("Lancement de l'assistant en mode local...")
        os.system("python run_local_only.py")
        return
    
    # Demander à l'utilisateur quel modèle utiliser
    print("Quel modèle souhaitez-vous utiliser ?")
    if has_github_token:
        print("1. Azure LLM (recommandé)")
    if has_gemini_key:  
        print("2. Gemini")
    print("3. Local (sans IA avancée)")
    print("4. Auto (essaie Azure, puis Gemini, puis Local)")
    
    choice = input("\nVotre choix (1-4) : ").strip()
    
    if choice == "1" and has_github_token:
        print("\nLancement de l'assistant avec Azure LLM...")
        os.system("python run_azure_only.py")
    elif choice == "2" and has_gemini_key:
        print("\nLancement de l'assistant avec Gemini...")
        # Importation directe pour éviter les conflits de dépendances
        from improved_chatbot import main as run_gemini
        await run_gemini()
    elif choice == "3":
        print("\nLancement de l'assistant en mode local...")
        os.system("python run_local_only.py")
    elif choice == "4":
        print("\nLancement de l'assistant en mode auto (essaie différents modèles)...")
        # Utiliser eldercare_assistant.py qui implémente déjà cette logique
        from eldercare_assistant import main as run_auto
        await run_auto()
    else:
        print("\nChoix invalide ou modèle non disponible.")
        print("Lancement de l'assistant en mode local par défaut...")
        os.system("python run_local_only.py")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nAu revoir! Passez une bonne journée!")
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        print(f"\nErreur lors du lancement: {str(e)}")
        print("Essayez de lancer directement 'python run_local_only.py' comme alternative.") 