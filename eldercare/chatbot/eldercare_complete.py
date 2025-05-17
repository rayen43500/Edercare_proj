"""
ElderCare Assistant Complet.
Ce script intègre toutes les fonctionnalités : détection d'intentions, exécution d'actions et LLM Azure.
"""

import os
import sys
import asyncio
import logging
import datetime
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

from intent_detector import IntentDetector
from action_handler import ActionHandler
from azure_llm_integration import AzureLLMProvider

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()


class ElderCareComplete:
    """
    ElderCare Assistant complet intégrant détection d'intentions, actions et IA conversationnelle.
    """
    
    def __init__(self):
        """Initialiser l'assistant complet."""
        self.intent_detector = IntentDetector()
        self.action_handler = ActionHandler()
        self.azure_llm = AzureLLMProvider()
        self.conversation_history = []
        
        # Profil utilisateur par défaut
        self.user_profile = {
            "id": "user1",
            "name": "Utilisateur",
            "age": 65,
            "interests": ["lecture", "musique", "jardinage"],
            "health_conditions": ["arthrite"],
            "tech_comfort": "débutant"
        }
    
    async def initialize(self) -> bool:
        """Initialiser les composants de l'assistant."""
        return self.azure_llm.initialize()
    
    async def process_message(self, user_message: str) -> str:
        """
        Traiter un message utilisateur en détectant l'intention et en exécutant l'action appropriée.
        
        Args:
            user_message: Message de l'utilisateur
            
        Returns:
            str: Texte de réponse
        """
        # Ajouter le message à l'historique de conversation
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Détecter l'intention
        intent, params = self.intent_detector.detect_intent(user_message)
        
        if intent:
            logger.info(f"Intention détectée : {intent}, Paramètres : {params}")
            
            # Exécuter l'action correspondante
            success, message, command = self.action_handler.handle_action(intent, params)
            
            # Ajouter la réponse à l'historique
            self.conversation_history.append({"role": "assistant", "content": message})
            
            # Si une commande a été exécutée, inclure cette information dans la réponse
            if command:
                return f"{message}\nCommande exécutée: {command}"
            return message
        else:
            # Aucune intention détectée, utiliser l'IA pour générer une réponse
            return await self._generate_ai_response(user_message)
    
    async def _generate_ai_response(self, user_message: str) -> str:
        """Générer une réponse avec l'IA Azure LLM."""
        # Créer prompt système avec infos profil utilisateur
        system_prompt = self._get_personalized_system_prompt()
        
        # Créer messages pour LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Ajouter historique de conversation (jusqu'aux 10 derniers échanges)
        messages.extend(self.conversation_history[-10:])
        
        # Générer réponse
        response, model = await self.azure_llm.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        # Ajouter réponse à l'historique de conversation
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Réduire l'historique s'il devient trop long
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _get_personalized_system_prompt(self) -> str:
        """Créer un prompt système personnalisé basé sur le profil utilisateur."""
        name = self.user_profile.get("name", "utilisateur")
        age = self.user_profile.get("age", "senior")
        interests = ", ".join(self.user_profile.get("interests", []))
        health = ", ".join(self.user_profile.get("health_conditions", []))
        tech_comfort = self.user_profile.get("tech_comfort", "débutant")
        
        return f"""Vous êtes un assistant intelligent appelé ElderCare Assistant, conçu pour aider les personnes âgées.

Information sur l'utilisateur:
- Nom: {name}
- Âge: {age} ans
- Intérêts: {interests}
- Conditions de santé: {health}
- Niveau de confort avec la technologie: {tech_comfort}

Vos directives:
1. Soyez respectueux, patient et empathique. Parlez clairement et simplement.
2. Donnez des instructions technologiques pas à pas et détaillées.
3. Répondez toujours en français. Utilisez un langage simple.
4. Pour les questions de santé, fournissez des informations générales mais conseillez de consulter un médecin pour des avis médicaux.
5. Soyez bref, concis. Évitez les textes trop longs.
6. Adaptez vos réponses au profil et aux besoins de l'utilisateur.
7. Si l'utilisateur a des difficultés techniques, proposez des solutions simples.
8. Soyez informatif, utile et amical - comme un aidant bienveillant."""


async def main():
    """Exécuter l'assistant ElderCare complet."""
    try:
        print("\n=== Bienvenue à ElderCare Assistant (Version Complète) ===\n")
        print("Initialisation de l'assistant...")
        
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