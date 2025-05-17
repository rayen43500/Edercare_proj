import os
import asyncio
import logging
import re
from typing import Dict, Any, Optional

# Import your existing modules
from improved_chatbot import ElderCareAssistant
from system_commands import SystemCommandExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedElderCareAssistant:
    """
    Enhanced ElderCare Assistant with system command execution capabilities.
    This assistant can answer questions and execute system commands like opening applications.
    """
    
    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced assistant.
        
        Args:
            user_profile: Optional user profile information
        """
        self.assistant = ElderCareAssistant(
            user_profile=user_profile,
            knowledge_dir="knowledge_base",
            cache_dir="response_cache",
            max_history=10,
            use_local_model=True
        )
        self.system_executor = SystemCommandExecutor()
        self.command_patterns = [
            r"(?:can you |please |)(?:open|launch|start|run) ([\w\s]+)",
            r"(?:i want to use|start up|get|show me) ([\w\s]+)",
            r"(?:open|launch|start|run) ([\w\s]+?)(?: for me| please| now|)",
            r"(?:peux-tu |pouvez-vous |s'il vous plaît |)(?:ouvrir|lancer|démarrer|exécuter) ([\w\s]+)",
            r"(?:je veux utiliser|démarre|montre-moi) ([\w\s]+)",
            r"(?:ouvre|lance|démarre|exécute) ([\w\s]+?)(?: pour moi| s'il te plaît| maintenant|)"
        ]
        
    async def initialize(self):
        """Initialize the assistant and its components."""
        await self.assistant.initialize()
        logger.info("Enhanced ElderCare Assistant initialized")
        
    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and generate a response or execute a command.
        
        Args:
            user_message: The user's input message
            
        Returns:
            Response text or command execution result
        """
        # Check if it's a command to open an application
        app_name = self._extract_app_name(user_message)
        
        if app_name:
            # Execute the command and generate a response
            success, message = self.system_executor.execute_command(app_name)
            if success:
                return f"J'ai ouvert {app_name} pour vous."
            else:
                # If command execution failed, return the error message
                # and list available applications
                available_apps = self.system_executor.get_available_applications()
                available_msg = "Voici les applications disponibles: " + ", ".join(available_apps[:5])
                if len(available_apps) > 5:
                    available_msg += ", et plus..."
                return f"{message} {available_msg}"
        
        # If not a command, process as a regular message
        return await self.assistant.process_message(user_message)
    
    def _extract_app_name(self, message: str) -> Optional[str]:
        """
        Extract application name from a message if it's a command.
        
        Args:
            message: User message
            
        Returns:
            Application name or None if not a command
        """
        message = message.lower()
        
        # Check each pattern for a match
        for pattern in self.command_patterns:
            match = re.search(pattern, message)
            if match:
                app_name = match.group(1).strip()
                return app_name
                
        # Check if any of the allowed applications are mentioned directly
        available_apps = self.system_executor.get_available_applications()
        for app in available_apps:
            if app in message:
                return app
                
        return None


async def main():
    """Run the enhanced ElderCare Assistant."""
    try:
        # Initialize with a sample user profile
        user_profile = {
            "id": "user1",
            "name": "User",
            "age": 65,
            "interests": ["reading", "music", "gardening"],
            "health_conditions": ["arthritis"],
            "tech_comfort": "beginner"
        }
        
        print("\n=== Bienvenue à ElderCare Assistant ===\n")
        print("Initialisation de l'assistant...")
        
        # Create and initialize the assistant
        assistant = EnhancedElderCareAssistant(user_profile=user_profile)
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