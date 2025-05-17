import os
import asyncio
import logging
import re
from typing import Dict, Any, Optional, List

# Import your existing modules
from improved_chatbot import ElderCareAssistant
from system_commands import SystemCommandExecutor
from azure_llm_integration import AzureLLMProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedElderCareAssistantAzure:
    """
    Enhanced ElderCare Assistant using Azure LLM with system command execution capabilities.
    This assistant can answer questions, execute system commands, and uses the advanced
    Azure AI Inference SDK for more natural and helpful responses.
    """
    
    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced assistant with Azure LLM.
        
        Args:
            user_profile: Optional user profile information
        """
        self.user_profile = user_profile or {
            "id": "user1",
            "name": "User",
            "age": 65,
            "interests": ["reading", "music", "gardening"],
            "health_conditions": ["arthritis"],
            "tech_comfort": "beginner"
        }
        
        # Initialize components
        self.knowledge_dir = "knowledge_base"
        self.system_executor = SystemCommandExecutor()
        self.azure_llm = AzureLLMProvider()
        self.conversation_history = []
        
        # Command patterns for application detection
        self.command_patterns = [
            r"(?:can you |please |)(?:open|launch|start|run) ([\w\s]+)",
            r"(?:i want to use|start up|get|show me) ([\w\s]+)",
            r"(?:open|launch|start|run) ([\w\s]+?)(?: for me| please| now|)",
            r"(?:peux-tu |pouvez-vous |s'il vous plaît |)(?:ouvrir|lancer|démarrer|exécuter) ([\w\s]+)",
            r"(?:je veux utiliser|démarre|montre-moi) ([\w\s]+)",
            r"(?:ouvre|lance|démarre|exécute) ([\w\s]+?)(?: pour moi| s'il te plaît| maintenant|)"
        ]
        
    async def initialize(self) -> bool:
        """
        Initialize the assistant and its components.
        
        Returns:
            bool: Success status
        """
        # Initialize the Azure LLM component
        llm_initialized = self.azure_llm.initialize()
        
        logger.info(f"Azure LLM initialization: {'success' if llm_initialized else 'failed'}")
        logger.info("Enhanced ElderCare Assistant with Azure LLM initialized")
        
        return llm_initialized
        
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
        
        # If not a command, process as a regular message using Azure LLM
        return await self._generate_llm_response(user_message)
    
    async def _generate_llm_response(self, user_message: str) -> str:
        """
        Generate a response using the Azure LLM with context from knowledge base.
        
        Args:
            user_message: User's message
            
        Returns:
            str: Generated response
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Create system prompt with user profile info
        system_prompt = self._get_personalized_system_prompt()
        
        # Get context from knowledge base files based on user message
        context = await self._retrieve_relevant_context(user_message)
        
        # Create messages for LLM
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context if available
        if context:
            messages.append({
                "role": "system",
                "content": f"INFORMATIONS PERTINENTES:\n{context}"
            })
        
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
        """
        Create a personalized system prompt based on user profile.
        
        Returns:
            str: System prompt
        """
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
    
    async def _retrieve_relevant_context(self, query: str) -> str:
        """
        Retrieve relevant information from knowledge base files.
        
        Args:
            query: User query
            
        Returns:
            str: Relevant context from knowledge base
        """
        # Simple keyword-based retrieval for now
        # You could implement more sophisticated retrieval here
        context_parts = []
        
        # Keywords to categories mapping
        category_keywords = {
            "health": ["santé", "douleur", "médical", "maladie", "arthrite", "arthrose", "sommeil", 
                    "médicament", "symptôme", "exercice", "thérapie", "douleurs"],
            "technology": ["téléphone", "smartphone", "ordinateur", "internet", "email", "application", 
                        "appli", "technologie", "wifi", "bluetooth", "whatsapp", "facebook",
                        "tablette", "appareil", "photo", "message", "sms"],
            "entertainment": ["livre", "film", "musique", "lecture", "loisir", "hobby", "jardin", 
                          "jardinage", "divertissement", "jeu", "television", "tv", "radio"]
        }
        
        # Determine relevant categories
        query_lower = query.lower()
        relevant_categories = []
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    relevant_categories.append(category)
                    break
        
        # If no relevant categories found, include all
        if not relevant_categories:
            relevant_categories = list(category_keywords.keys())
        
        # Read relevant knowledge base files
        for category in relevant_categories:
            context_parts.extend(await self._read_category_files(category))
        
        # Return combined context (limit to reasonable size)
        return "\n\n".join(context_parts[:3])  # Limit to 3 context parts
    
    async def _read_category_files(self, category: str) -> List[str]:
        """
        Read files from a specific knowledge base category.
        
        Args:
            category: Category name
            
        Returns:
            List[str]: Content of relevant files
        """
        category_dir = os.path.join(self.knowledge_dir, category)
        results = []
        
        if not os.path.exists(category_dir):
            return results
            
        try:
            for filename in os.listdir(category_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(category_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        results.append(content)
        except Exception as e:
            logger.error(f"Error reading files from {category}: {str(e)}")
        
        return results
    
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
    """Run the enhanced ElderCare Assistant with Azure LLM."""
    try:
        print("\n=== Bienvenue à ElderCare Assistant (Version Azure) ===\n")
        print("Initialisation de l'assistant...")
        
        # Create and initialize the assistant
        assistant = EnhancedElderCareAssistantAzure()
        initialized = await assistant.initialize()
        
        if not initialized:
            print("NOTE: L'assistant fonctionne, mais sans Azure LLM. Pour utiliser Azure LLM:")
            print("1. Assurez-vous d'avoir la variable d'environnement GITHUB_TOKEN définie")
            print("2. Votre token doit avoir les permissions 'models:read'")
        
        print("Assistant initialisé avec succès!")
        
        # Display welcome message in French
        print("\nJe suis votre assistant ElderCare (Version Azure). Je peux vous aider avec:")
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