import os
import asyncio
import logging
import re
from typing import Dict, Any, Optional, List

# Import modules
from system_commands import SystemCommandExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleElderCareAssistant:
    """
    Version simplifiée de l'ElderCare Assistant qui fonctionne sans dépendances externes.
    Cette version peut exécuter des commandes système et fournir des réponses basiques.
    """
    
    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        """Initialize the simple assistant."""
        self.system_executor = SystemCommandExecutor()
        self.conversation_history = []
        
        # Default user profile
        self.user_profile = user_profile or {
            "id": "user1",
            "name": "User",
            "age": 65,
            "interests": ["reading", "music", "gardening"],
            "health_conditions": ["arthritis"],
            "tech_comfort": "beginner"
        }
    
    async def initialize(self) -> bool:
        """Initialize the assistant components."""
        logger.info("Simple ElderCare Assistant initialized")
        return True
    
    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and generate a response or execute a command.
        
        Args:
            user_message: User's input message
            
        Returns:
            str: Response text
        """
        # Check if it's a command to open an application
        app_name = self._extract_app_name(user_message.lower())
        
        if app_name:
            # Execute the command and generate a response
            success, message = self.system_executor.execute_command(app_name)
            if success:
                return f"J'ai ouvert {app_name} pour vous.\n{message}"
            else:
                # If command execution failed, return the error message
                available_apps = self.system_executor.get_available_applications()
                available_msg = "Voici les applications disponibles: " + ", ".join(available_apps[:5])
                if len(available_apps) > 5:
                    available_msg += ", et plus..."
                return f"{message} {available_msg}"
        
        # If not a command, process as a regular message
        return self._generate_simple_response(user_message)
    
    def _generate_simple_response(self, user_message: str) -> str:
        """
        Generate a simple response based on keywords in the message.
        
        Args:
            user_message: User's message
            
        Returns:
            str: Generated response
        """
        user_message_lower = user_message.lower()
        
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Simple keyword-based responses
        if any(word in user_message_lower for word in ["bonjour", "salut", "hello", "hi"]):
            response = "Bonjour ! Comment puis-je vous aider aujourd'hui ?"
        
        elif any(word in user_message_lower for word in ["santé", "douleur", "mal", "medicament", "médicament"]):
            response = "Pour les questions de santé, je vous recommande de consulter un professionnel de la santé. Je peux vous aider à trouver des informations générales, mais un médecin pourra vous donner des conseils personnalisés."
        
        elif any(word in user_message_lower for word in ["internet", "email", "ordinateur", "téléphone", "smartphone", "whatsapp"]):
            response = "Les technologies peuvent parfois être complexes. Pouvez-vous me dire précisément ce que vous essayez de faire ? Je peux vous guider étape par étape."
        
        elif any(word in user_message_lower for word in ["film", "musique", "livre", "lire", "regarder", "écouter"]):
            response = "Les activités de loisir sont importantes pour le bien-être. Quels sont vos genres préférés ? Je pourrais vous faire quelques suggestions adaptées."
        
        elif "merci" in user_message_lower:
            response = "Je vous en prie ! N'hésitez pas si vous avez d'autres questions."
            
        elif any(word in user_message_lower for word in ["aide", "comment", "besoin"]):
            response = "Je suis là pour vous aider. N'hésitez pas à me poser des questions sur la technologie, la santé ou les loisirs. Vous pouvez aussi me demander d'ouvrir des applications en disant par exemple 'ouvre Chrome'."
            
        else:
            response = "Je comprends. Pour mieux vous aider, pourriez-vous me donner plus de détails sur ce que vous recherchez ?"
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Trim conversation history if it gets too long
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _extract_app_name(self, message: str) -> Optional[str]:
        """Extract application name from message if it's a command."""
        # Simple command pattern
        command_patterns = [
            "ouvre ", "lance ", "démarre ", "exécute ",
            "open ", "launch ", "start ", "run "
        ]
        
        for pattern in command_patterns:
            if pattern in message:
                try:
                    app_name = message.split(pattern)[1].split()[0]
                    return app_name
                except IndexError:
                    return None
                
        return None


async def main():
    """Run the simple ElderCare Assistant."""
    try:
        print("\n=== Bienvenue à ElderCare Assistant (Version simple) ===\n")
        print("Initialisation de l'assistant...")
        
        # Create and initialize the assistant
        assistant = SimpleElderCareAssistant()
        await assistant.initialize()
        
        print("Assistant initialisé avec succès!")
        
        # Display welcome message in French
        print("\nJe suis votre assistant ElderCare. Je peux vous aider avec:")
        print("1. Assistance technologique")
        print("2. Conseils de santé et informations")
        print("3. Recommandations de divertissement")
        print("4. Conversation générale et soutien")
        print("5. Ouvrir des applications pour vous")
        
        print("\nTapez 'aide' pour plus d'informations ou 'quitter' pour sortir.")
        
        # Main chat loop
        while True:
            try:
                user_input = input("\nVous: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quitter', 'sortir', 'bye', 'quit']:
                    print("\nAu revoir! Passez une bonne journée!")
                    break
                elif user_input.lower() in ['aide', 'help']:
                    print("\nCommandes disponibles:")
                    print("- aide: Afficher ce message d'aide")
                    print("- quitter: Quitter le programme")
                    print("\nVous pouvez aussi me demander de:")
                    print("- Ouvrir des applications (ex: 'ouvre Chrome')")
                    print("- Vous aider avec la technologie")
                    print("- Donner des conseils de santé")
                    print("- Recommander des divertissements")
                    continue
                
                # Process user message
                response = await assistant.process_message(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nAu revoir! Passez une bonne journée!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print("\nJe m'excuse, mais j'ai rencontré une erreur. Veuillez réessayer.")
                continue
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print("\nJe m'excuse, mais j'ai rencontré une erreur grave. Veuillez redémarrer l'application.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nAu revoir! Passez une bonne journée!")
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        print("\nErreur fatale. Veuillez redémarrer l'application.") 