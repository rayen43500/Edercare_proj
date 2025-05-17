#!/usr/bin/env python3
"""
Script de démarrage direct pour ElderCareComplete.
Ce script lance directement l'assistant ElderCareComplete sans passer par
les autres gestionnaires comme run_eldercare ou run_eldercare_assistant.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

# Import depuis eldercare_complete.py
from eldercare_complete import ElderCareComplete

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Exécuter directement l'assistant ElderCareComplete."""
    try:
        print("\n=== Bienvenue à ElderCare Assistant (Version Complète - Direct) ===\n")
        print("Initialisation de l'assistant...")
        
        # Charger les variables d'environnement
        load_dotenv()
        load_dotenv('azure_config.env', override=True)
        
        # Configurer le token GitHub pour Azure LLM
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            github_token = input("Entrez votre token GitHub pour utiliser Azure LLM: ")
            os.environ["GITHUB_TOKEN"] = github_token
        
        # Créer et initialiser l'assistant
        assistant = ElderCareComplete()
        initialized = await assistant.initialize()
        
        if not initialized:
            print("ERREUR: Impossible d'initialiser Azure LLM.")
            print("Assurez-vous que votre token GitHub est valide et a les permissions 'models:read'")
            return
        
        print("Assistant initialisé avec succès!")
        
        # Afficher message de bienvenue en français
        print("\nJe suis votre assistant ElderCare complet. Je peux vous aider avec:")
        print("1. Conversation générale et assistance")
        print("2. Exécution de commandes (ouvrir applications, dossiers, etc.)")
        print("3. Recherche web et informations")
        print("4. Gestion de vos notes et rappels")
        print("5. Contrôle système (son, arrêt, etc.)")
        
        # Afficher quelques exemples de commandes
        print("\nExemples de ce que vous pouvez me demander:")
        print("- \"Ouvre Chrome\"")
        print("- \"Quelle heure est-il ?\"")
        print("- \"Recherche les bienfaits du yoga\"")
        print("- \"Prends une note : acheter du pain\"")
        print("- \"Quels sont les symptômes de l'arthrite ?\"")
        
        print("\nTapez 'aide' pour voir plus d'exemples ou 'quitter' pour sortir.")
        
        # Boucle de discussion principale
        while True:
            try:
                user_input = input("\nVous: ").strip()
                
                # Gestion des commandes spéciales
                if user_input.lower() in ['quitter', 'sortir', 'bye', 'quit', 'exit']:
                    print("\nAu revoir! Passez une bonne journée!")
                    break
                elif user_input.lower() in ['aide', 'help']:
                    print("\nVoici ce que je peux faire pour vous :")
                    
                    # Afficher les intentions disponibles avec des exemples
                    all_intents = assistant.intent_detector.get_all_intents()
                    for intent in all_intents:
                        examples = assistant.intent_detector.get_intent_examples(intent)
                        if examples:
                            intent_name = intent.replace("_", " ").capitalize()
                            print(f"\n{intent_name}:")
                            print(f"  Exemple: \"{examples[0]}\"")
                    
                    continue
                
                # Traiter le message utilisateur
                print("\nTraitement en cours...")
                response = await assistant.process_message(user_input)
                print(f"\nElderCare: {response}")
                
            except KeyboardInterrupt:
                print("\n\nAu revoir! Passez une bonne journée!")
                break
            except Exception as e:
                print(f"\nErreur: {str(e)}")
                logger.error(f"Erreur dans la boucle principale: {str(e)}")
    
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        print(f"\nErreur lors du lancement: {str(e)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nAu revoir! Passez une bonne journée!")
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        print(f"\nErreur lors du lancement: {str(e)}")
        print("Assurez-vous que toutes les dépendances sont installées.") 