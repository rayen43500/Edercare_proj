"""
Script unifié pour l'ElderCare Assistant intégrant à la fois la conversation IA et l'exécution de commandes système.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from azure_llm_integration import AzureLLMProvider
from commande_service import CommandeService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ElderCareUnified:
    """
    ElderCare Assistant unifié qui combine conversation IA et exécution de commandes.
    """
    
    def __init__(self):
        """Initialiser l'assistant unifié."""
        self.azure_llm = AzureLLMProvider()
        self.command_service = CommandeService()
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
        Traiter un message utilisateur avec Azure LLM ou exécuter une commande système.
        
        Args:
            user_message: Message de l'utilisateur
            
        Returns:
            str: Texte de réponse
        """
        # Vérifier si c'est une commande d'ouverture d'application
        app_name = self._extract_app_name(user_message.lower())
        
        if app_name:
            # Exécuter la commande via le service de commandes
            success, message, commande = self.command_service.executer_commande(app_name)
            
            if success:
                # Ajouter l'interaction à l'historique de conversation
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": message})
                
                return f"{message}\nCommande exécutée: {commande}"
            else:
                # Si l'exécution a échoué, retourner le message d'erreur
                available_apps = self.command_service.obtenir_applications_disponibles()
                available_msg = "Voici les applications disponibles: " + ", ".join(available_apps[:5])
                if len(available_apps) > 5:
                    available_msg += ", et plus..."
                
                return f"{message} {available_msg}"
        
        # Si ce n'est pas une commande, traiter avec Azure LLM
        return await self._generate_azure_response(user_message)
    
    async def _generate_azure_response(self, user_message: str) -> str:
        """Générer une réponse en utilisant Azure LLM."""
        # Ajouter le message utilisateur à l'historique de conversation
        self.conversation_history.append({"role": "user", "content": user_message})
        
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
    
    def _extract_app_name(self, message: str) -> str:
        """Extraire le nom de l'application depuis le message s'il s'agit d'une commande."""
        # Motifs de commande simples
        command_patterns = [
            "ouvre ", "ouvrir ", "lance ", "lancer ", "démarre ", "démarrer ", "exécute ", "exécuter ",
            "open ", "launch ", "start ", "run "
        ]
        
        for pattern in command_patterns:
            if pattern in message:
                app_name = message.split(pattern)[1].strip()
                return app_name
                
        return None


async def main():
    """Exécuter l'assistant ElderCare unifié."""
    try:
        print("\n=== Bienvenue à ElderCare Assistant (Version Unifiée) ===\n")
        print("Initialisation de l'assistant...")
        
        # Configurer le token GitHub pour Azure LLM
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            github_token = input("Entrez votre token GitHub pour utiliser Azure LLM: ")
            os.environ["GITHUB_TOKEN"] = github_token
        
        # Créer et initialiser l'assistant
        assistant = ElderCareUnified()
        initialized = await assistant.initialize()
        
        if not initialized:
            print("ERREUR: Impossible d'initialiser Azure LLM.")
            print("Assurez-vous que votre token GitHub est valide et a les permissions 'models:read'")
            return
        
        print("Assistant initialisé avec succès!")
        
        # Afficher message de bienvenue en français
        print("\nJe suis votre assistant ElderCare unifié. Je peux vous aider avec:")
        print("1. Assistance technologique et conversation générale")
        print("2. Conseils de santé et informations")
        print("3. Ouvrir des applications pour vous")
        
        # Afficher les applications disponibles
        apps = assistant.command_service.obtenir_applications_disponibles()
        print("\nApplications disponibles :")
        for i in range(0, len(apps), 3):
            ligne = []
            for j in range(3):
                if i+j < len(apps):
                    ligne.append(f"- {apps[i+j]}")
            print("  ".join(ligne))
        
        print("\nTapez 'aide' pour plus d'informations ou 'quitter' pour sortir.")
        
        # Boucle de discussion principale
        while True:
            try:
                user_input = input("\nVous: ").strip()
                
                # Gestion des commandes spéciales
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
                    print("- Répondre à des questions générales")
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