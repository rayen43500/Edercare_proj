"""
Script d'entraînement pour ElderCare Assistant.
Ce script démontre comment utiliser l'assistant et ajouter de nouvelles commandes.
"""

import os
import sys
import asyncio
import logging
import json
from typing import Dict, List, Optional

from intent_detector import IntentDetector
from action_handler import ActionHandler

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommandTrainer:
    """Démontre et entraîne l'utilisation de l'ElderCare Assistant."""
    
    def __init__(self):
        """Initialisation du trainer."""
        self.intent_detector = IntentDetector()
        self.action_handler = ActionHandler()
        self.training_examples = []
        self.custom_commands = {}
        
        # Charger les commandes personnalisées si le fichier existe
        if os.path.exists("custom_commands.json"):
            try:
                with open("custom_commands.json", "r", encoding="utf-8") as f:
                    self.custom_commands = json.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement des commandes: {str(e)}")
    
    def save_custom_commands(self):
        """Enregistre les commandes personnalisées dans un fichier."""
        try:
            with open("custom_commands.json", "w", encoding="utf-8") as f:
                json.dump(self.custom_commands, f, ensure_ascii=False, indent=2)
            print("Commandes personnalisées enregistrées avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des commandes: {str(e)}")
            print(f"Erreur: {str(e)}")
    
    def add_custom_command(self, command_name: str, ssh_command: str, response_message: str):
        """Ajoute une commande personnalisée."""
        self.custom_commands[command_name.lower()] = {
            "ssh_command": ssh_command,
            "message": response_message
        }
        print(f"Commande '{command_name}' ajoutée.")
    
    def process_message(self, user_message: str) -> str:
        """
        Traite un message utilisateur, détecte l'intention et exécute l'action correspondante.
        
        Args:
            user_message: Le message de l'utilisateur
            
        Returns:
            La réponse de l'assistant
        """
        # Détecter l'intention
        intent, params = self.intent_detector.detect_intent(user_message)
        
        if intent == "ouvrir_application" and params:
            app_name = params[0].lower()
            
            # Vérifier si c'est une commande personnalisée
            if app_name in self.custom_commands:
                cmd = self.custom_commands[app_name]
                return f"{cmd['message']}\n💻 (Commande SSH exécutée) : {cmd['ssh_command']}"
        
        if intent:
            # Exécuter l'action correspondante
            success, message, command = self.action_handler.handle_action(intent, params)
            return message
        else:
            # Aucune intention détectée
            return "Je ne comprends pas cette commande. Tapez 'aide' pour voir les commandes disponibles."
    
    def run_training_session(self):
        """Exécute une session d'entraînement interactive."""
        print("\n=== Session d'entraînement ElderCare Assistant ===\n")
        print("Vous pouvez:")
        print("1. Tester les commandes existantes (ex: 'ouvre chrome')")
        print("2. Ajouter des commandes personnalisées (ex: 'ajouter commande')")
        print("3. Voir des exemples (ex: 'exemples')")
        print("4. Quitter (ex: 'quitter')")
        
        while True:
            try:
                user_input = input("\nVous: ").strip()
                
                if user_input.lower() in ["quitter", "exit", "sortir", "bye", "q"]:
                    print("Au revoir! Session d'entraînement terminée.")
                    break
                elif user_input.lower() in ["aide", "help"]:
                    print("\nCommandes disponibles:")
                    print("- 'ouvre/lance [application]': Ouvre une application")
                    print("- 'ajouter commande': Ajoute une nouvelle commande personnalisée")
                    print("- 'voir commandes': Affiche les commandes personnalisées")
                    print("- 'exemples': Montre des exemples d'utilisation")
                    print("- 'quitter': Termine la session")
                elif user_input.lower() == "exemples":
                    self._show_examples()
                elif user_input.lower() == "voir commandes":
                    self._show_custom_commands()
                elif user_input.lower() == "ajouter commande":
                    self._add_command_interactive()
                else:
                    response = self.process_message(user_input)
                    print(f"\nElderCare: {response}")
                    
                    # Enregistrer l'exemple d'entraînement
                    self.training_examples.append({
                        "user": user_input,
                        "assistant": response
                    })
            
            except KeyboardInterrupt:
                print("\nSession interrompue.")
                break
            except Exception as e:
                print(f"\nErreur: {str(e)}")
    
    def _show_examples(self):
        """Affiche des exemples d'utilisation."""
        print("\n=== Exemples d'utilisation ===")
        examples = [
            {"user": "Salut", "assistant": "Salut ! Que puis-je faire pour toi aujourd'hui ?"},
            {"user": "Tu peux ouvrir Chrome ?", "assistant": "Bien sûr, j'ouvre Google Chrome maintenant.\n💻 (Commande SSH exécutée) : google-chrome &"},
            {"user": "Lance VS Code", "assistant": "D'accord, Visual Studio Code est en cours de lancement.\n💻 (Commande SSH exécutée) : code &"},
            {"user": "Ouvre la calculatrice", "assistant": "Voici votre calculatrice.\n💻 (Commande SSH exécutée) : gnome-calculator &"},
            {"user": "Je veux écouter de la musique", "assistant": "J'ouvre Spotify pour vous.\n💻 (Commande SSH exécutée) : spotify &"},
            {"user": "Et si tu me lances un terminal ?", "assistant": "Terminal lancé.\n💻 (Commande SSH exécutée) : gnome-terminal &"},
            {"user": "Peux-tu ouvrir le gestionnaire de fichiers ?", "assistant": "Bien sûr, gestionnaire de fichiers en cours d'ouverture.\n💻 (Commande SSH exécutée) : nautilus &"},
            {"user": "Tu peux lancer GIMP ?", "assistant": "GIMP est prêt pour l'édition d'images.\n💻 (Commande SSH exécutée) : gimp &"}
        ]
        
        for example in examples:
            print(f"\nUtilisateur : {example['user']}")
            print(f"Assistant : {example['assistant']}")
    
    def _show_custom_commands(self):
        """Affiche les commandes personnalisées."""
        if not self.custom_commands:
            print("\nAucune commande personnalisée n'a été ajoutée.")
            return
        
        print("\n=== Commandes personnalisées ===")
        for command_name, command_data in self.custom_commands.items():
            print(f"\nCommande: {command_name}")
            print(f"Message: {command_data['message']}")
            print(f"Commande SSH: {command_data['ssh_command']}")
    
    def _add_command_interactive(self):
        """Ajoute une commande de manière interactive."""
        print("\n=== Ajouter une commande personnalisée ===")
        
        command_name = input("Nom de la commande (ex: 'firefox'): ").strip()
        if not command_name:
            print("Opération annulée.")
            return
        
        ssh_command = input("Commande SSH (ex: 'firefox &'): ").strip()
        if not ssh_command:
            print("Opération annulée.")
            return
        
        response_message = input("Message de réponse (ex: 'Firefox est prêt.'): ").strip()
        if not response_message:
            print("Opération annulée.")
            return
        
        self.add_custom_command(command_name, ssh_command, response_message)
        self.save_custom_commands()


def main():
    """Fonction principale."""
    try:
        trainer = CommandTrainer()
        trainer.run_training_session()
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        print(f"\nErreur: {str(e)}")


if __name__ == "__main__":
    main() 