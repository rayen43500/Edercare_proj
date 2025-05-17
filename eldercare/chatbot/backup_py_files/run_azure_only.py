import os
import asyncio
import logging
from dotenv import load_dotenv
from azure_llm_integration import AzureLLMProvider
from system_commands import SystemCommandExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AzureOnlyAssistant:
    """
    ElderCare Assistant configured to use only Azure LLM for responses.
    """
    
    def __init__(self):
        """Initialize the Azure-only assistant."""
        self.azure_llm = AzureLLMProvider()
        self.system_executor = SystemCommandExecutor()
        self.conversation_history = []
        
        # Default user profile
        self.user_profile = {
            "id": "user1",
            "name": "User",
            "age": 65,
            "interests": ["reading", "music", "gardening"],
            "health_conditions": ["arthritis"],
            "tech_comfort": "beginner"
        }
    
    async def initialize(self) -> bool:
        """Initialize the assistant components."""
        return self.azure_llm.initialize()
    
    async def process_message(self, user_message: str) -> str:
        """
        Process a user message using only Azure LLM.
        
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
        
        # If not a command, process with Azure LLM
        return await self._generate_azure_response(user_message)
    
    async def _generate_azure_response(self, user_message: str) -> str:
        """Generate a response using Azure LLM."""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Create system prompt with user profile info
        system_prompt = self._get_personalized_system_prompt()
        
        # Create messages for LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history (up to last 5 exchanges)
        messages.extend(self.conversation_history[-10:])
        
        # Generate response
        response, model = await self.azure_llm.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Trim conversation history if it gets too long
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def _get_personalized_system_prompt(self) -> str:
        """Create a personalized system prompt based on user profile."""
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
    
    def _extract_app_name(self, message: str) -> str:
        """Extract application name from message if it's a command."""
        # Simple command pattern
        command_patterns = [
            "ouvre ", "lance ", "démarre ", "exécute ",
            "open ", "launch ", "start ", "run "
        ]
        
        for pattern in command_patterns:
            if pattern in message:
                app_name = message.split(pattern)[1].split()[0]
                return app_name
                
        return None


async def main():
    """Run the Azure-only ElderCare Assistant."""
    try:
        print("\n=== Bienvenue à ElderCare Assistant (Azure uniquement) ===\n")
        print("Initialisation de l'assistant...")
        
        # Set up GitHub token for Azure LLM
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            github_token = input("Entrez votre token GitHub pour utiliser Azure LLM: ")
            os.environ["GITHUB_TOKEN"] = github_token
        
        # Create and initialize the assistant
        assistant = AzureOnlyAssistant()
        initialized = await assistant.initialize()
        
        if not initialized:
            print("ERREUR: Impossible d'initialiser Azure LLM.")
            print("Assurez-vous que votre token GitHub est valide et a les permissions 'models:read'")
            return
        
        print("Assistant initialisé avec succès!")
        
        # Display welcome message in French
        print("\nJe suis votre assistant ElderCare (Azure uniquement). Je peux vous aider avec:")
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