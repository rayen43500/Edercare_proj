"""
Client multi-fournisseurs pour l'assistant ElderCare
Permet de basculer automatiquement entre Azure AI, GitHub AI et le mode local
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Vérifier si les SDKs sont disponibles
try:
    from azure.ai.inference import AIClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    logger.warning("Azure AI SDK n'est pas installé. Utilisez: pip install azure-ai-inference")

try:
    from github_ai_integration import GitHubAIClient
    GITHUB_AI_AVAILABLE = True
except ImportError:
    GITHUB_AI_AVAILABLE = False
    logger.warning("Module GitHub AI non trouvé. Assurez-vous que github_ai_integration.py existe.")

class SimpleLocalLLM:
    """
    Modèle local simple pour répondre aux questions basiques.
    Utilisé comme fallback de dernier recours.
    """
    
    def __init__(self):
        """Initialise le modèle local simple."""
        self.responses = {
            "salutation": [
                "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
                "Bonjour ! Que puis-je faire pour vous ?",
                "Salut ! Je suis votre assistant ElderCare. Comment puis-je vous aider ?"
            ],
            "aide": [
                "Je peux vous aider à ouvrir des applications, chercher des informations, "
                "et répondre à vos questions. Que souhaitez-vous faire ?",
                "Voici comment je peux vous aider : ouvrir des applications, fermer des programmes, "
                "répondre à vos questions sur divers sujets."
            ],
            "remerciement": [
                "De rien ! Je suis là pour vous aider.",
                "Avec plaisir ! N'hésitez pas si vous avez besoin d'autre chose."
            ],
            "adieu": [
                "Au revoir ! Passez une bonne journée.",
                "À bientôt ! N'hésitez pas à revenir si vous avez besoin d'aide."
            ],
            "default": [
                "Je suis désolé, je fonctionne actuellement en mode local limité. "
                "Je peux vous aider avec des tâches simples comme ouvrir des applications.",
                "Je suis en mode simplifié pour le moment. Je peux vous aider pour les "
                "commandes de base. Pour des réponses plus détaillées, veuillez réessayer plus tard."
            ]
        }
        
        # Mots-clés pour la détection d'intention simple
        self.keywords = {
            "salutation": ["bonjour", "salut", "hello", "hi", "hey", "coucou"],
            "aide": ["aide", "help", "comment", "quoi", "que", "peux-tu"],
            "remerciement": ["merci", "thanks", "thank", "gratitude", "appreciation"],
            "adieu": ["au revoir", "adieu", "bye", "ciao", "à plus", "à bientôt"]
        }
    
    def detect_intent(self, message: str) -> str:
        """
        Détecte l'intention de l'utilisateur basée sur des mots-clés.
        
        Args:
            message: Message de l'utilisateur
            
        Returns:
            str: Intention détectée
        """
        message_lower = message.lower()
        
        # Vérifier les mots-clés pour chaque intention
        for intent, words in self.keywords.items():
            if any(word in message_lower for word in words):
                return intent
                
        # Vérifier si c'est une commande d'application
        if any(x in message_lower for x in ["ouvrir", "ouvre", "lancer", "lance", "démarrer", "démarre"]):
            return "ouvrir_application"
            
        if any(x in message_lower for x in ["fermer", "ferme", "quitter", "quitte", "arrêter", "arrête"]):
            return "fermer_application"
            
        return "default"
    
    def generate_response(self, message: str) -> str:
        """
        Génère une réponse simple basée sur le message de l'utilisateur.
        
        Args:
            message: Message de l'utilisateur
            
        Returns:
            str: Réponse générée
        """
        import random
        
        intent = self.detect_intent(message)
        responses = self.responses.get(intent, self.responses["default"])
        
        return random.choice(responses)


class MultiProviderLLM:
    """
    Client qui gère plusieurs fournisseurs d'IA: Azure, GitHub et local.
    Bascule automatiquement entre les fournisseurs selon la disponibilité.
    """
    
    def __init__(self):
        """Initialise le client multi-fournisseurs."""
        # Variables de configuration
        self.azure_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_INFERENCE_API_KEY")
        self.azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
        self.github_token = os.getenv("GITHUB_TOKEN")
        
        # Mode de fonctionnement (azure, github, local, auto)
        self.mode = os.getenv("ASSISTANT_MODE", "auto").lower()
        
        # Clients pour chaque fournisseur
        self.azure_client = None
        self.github_client = None
        self.local_client = SimpleLocalLLM()
        
        # État d'Azure
        self.azure_status = {
            "available": False,
            "rate_limited": False,
            "reset_time": None,
            "last_error": None
        }
        
        # Charger le statut sauvegardé
        self._load_azure_status()
        
        # Initialiser les clients disponibles
        self._init_clients()
        
        # Afficher le status des intégrations
        self._log_status()
    
    def _init_clients(self):
        """Initialise les clients disponibles."""
        # Initialiser GitHub AI si disponible
        if GITHUB_AI_AVAILABLE and self.github_token:
            try:
                self.github_client = GitHubAIClient()
                logger.info("Client GitHub AI initialisé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du client GitHub AI: {str(e)}")
        
        # Initialiser Azure AI seulement si en mode azure ou auto, et s'il n'est pas en limite de requêtes
        if (self.mode in ["azure", "auto"]) and AZURE_SDK_AVAILABLE and not self.azure_status["rate_limited"]:
            if self.azure_endpoint and self.azure_api_key:
                try:
                    self.azure_client = AIClient(
                        endpoint=self.azure_endpoint,
                        credential=AzureKeyCredential(self.azure_api_key)
                    )
                    self.azure_status["available"] = True
                    logger.info("Client Azure AI initialisé avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors de l'initialisation du client Azure: {str(e)}")
                    self.azure_status["available"] = False
                    self.azure_status["last_error"] = str(e)
    
    def _log_status(self):
        """Affiche le statut des intégrations."""
        logger.info(f"Assistant initialisé en mode {self.mode}")
        
        # Statut Azure
        if self.azure_status["rate_limited"]:
            logger.warning(f"Azure AI en limite de requêtes - Réinitialisation: {self.azure_status['reset_time']}")
        elif self.azure_client:
            logger.info("Azure AI disponible et configuré")
        else:
            logger.info("Azure AI non disponible")
            
        # Statut GitHub
        if self.github_client and self.github_client.is_available():
            logger.info("GitHub AI disponible et configuré")
        else:
            logger.info("GitHub AI non disponible")
            
        # Mode local toujours disponible
        logger.info("Mode local disponible (fallback)")
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> Tuple[str, str]:
        """
        Génère une réponse en utilisant le fournisseur approprié selon le mode et la disponibilité.
        
        Args:
            messages: Liste de messages pour la conversation
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            
        Returns:
            Tuple[str, str]: (réponse, modèle utilisé)
        """
        # Si on est en mode spécifique, on utilise uniquement ce fournisseur
        if self.mode == "azure":
            return await self._use_azure(messages, temperature, max_tokens)
        elif self.mode == "github":
            return await self._use_github(messages, temperature, max_tokens)
        elif self.mode == "local":
            return await self._use_local(messages), "local"
        
        # Mode auto: essayer les fournisseurs dans l'ordre de préférence
        
        # 1. Essayer Azure AI (s'il est disponible et non limité)
        if self.azure_client and self.azure_status["available"] and not self.azure_status["rate_limited"]:
            try:
                return await self._use_azure(messages, temperature, max_tokens)
            except HttpResponseError as e:
                # Gérer la limite de requêtes
                if e.status_code == 429:
                    logger.warning("Limite de requêtes Azure AI atteinte! Passage à GitHub AI")
                    self._handle_azure_rate_limit(e)
                else:
                    logger.error(f"Erreur HTTP lors de l'appel à Azure: {e.status_code} - {e.message}")
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à Azure: {str(e)}")
        
        # 2. Essayer GitHub AI si disponible
        if self.github_client and self.github_client.is_available():
            try:
                return await self._use_github(messages, temperature, max_tokens)
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à GitHub AI: {str(e)}")
        
        # 3. Fallback vers le mode local
        logger.warning("Aucun fournisseur d'IA n'est disponible. Utilisation du mode local.")
        return await self._use_local(messages), "local (fallback)"
    
    def _handle_azure_rate_limit(self, error: HttpResponseError):
        """Gère les erreurs de limite de requêtes Azure."""
        self.azure_status["rate_limited"] = True
        retry_after = error.headers.get("Retry-After")
        
        if retry_after:
            try:
                seconds = int(retry_after)
                reset_time = datetime.now() + timedelta(seconds=seconds)
                self.azure_status["reset_time"] = reset_time.isoformat()
                logger.info(f"Reset de la limite Azure prévu à {reset_time}")
            except:
                self.azure_status["reset_time"] = retry_after  # Stocker tel quel
        
        self.azure_status["last_error"] = f"Rate limit reached: {error.message}"
        self._save_azure_status()
    
    async def _use_azure(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, str]:
        """Utilise Azure AI pour générer une réponse."""
        if not self.azure_client:
            raise Exception("Client Azure AI non disponible")
            
        # Formatage correct pour azure-ai-inference
        chat_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            chat_messages.append({"role": role, "content": content})
            
        # Appel à Azure OpenAI
        response = await asyncio.to_thread(
            self.azure_client.chat.completions.create,
            model=self.azure_deployment,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content, f"azure/{self.azure_deployment}"
    
    async def _use_github(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float,
        max_tokens: int
    ) -> Tuple[str, str]:
        """Utilise GitHub AI pour générer une réponse."""
        if not self.github_client or not self.github_client.is_available():
            raise Exception("Client GitHub AI non disponible")
            
        return await self.github_client.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def _use_local(self, messages: List[Dict[str, str]]) -> str:
        """Utilise le modèle local pour générer une réponse."""
        # Extraire le dernier message de l'utilisateur
        user_message = ""
        for message in reversed(messages):
            if message["role"] == "user":
                user_message = message["content"]
                break
                
        return self.local_client.generate_response(user_message)
    
    def _load_azure_status(self) -> None:
        """Charge le statut Azure depuis le fichier de configuration."""
        config_dir = Path.home() / ".eldercare_assistant"
        config_file = config_dir / "azure_status.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                    self.azure_status["available"] = status.get("azure_available", False)
                    self.azure_status["rate_limited"] = status.get("use_fallback", False)
                    
                    rate_limit_info = status.get("rate_limit_info", {})
                    self.azure_status["reset_time"] = rate_limit_info.get("reset_time")
                    self.azure_status["last_error"] = rate_limit_info.get("last_error")
                    
                    # Vérifier si le délai de réinitialisation est passé
                    if self.azure_status["reset_time"] and self.azure_status["rate_limited"]:
                        try:
                            reset_time = datetime.fromisoformat(self.azure_status["reset_time"])
                            if datetime.now() > reset_time:
                                logger.info("Délai de limite d'API expiré. Réinitialisation du statut.")
                                self.azure_status["rate_limited"] = False
                                self.azure_status["reset_time"] = None
                        except:
                            # Si c'est un nombre de secondes plutôt qu'une date
                            try:
                                seconds = int(self.azure_status["reset_time"])
                                last_checked = datetime.fromisoformat(status.get("last_checked"))
                                reset_time = last_checked + timedelta(seconds=seconds)
                                if datetime.now() > reset_time:
                                    logger.info("Délai de limite d'API expiré. Réinitialisation du statut.")
                                    self.azure_status["rate_limited"] = False
                                    self.azure_status["reset_time"] = None
                            except:
                                pass
                    
                    logger.info(f"Statut Azure chargé: Disponible={self.azure_status['available']}, "
                             f"Limité={self.azure_status['rate_limited']}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du statut Azure: {str(e)}")
    
    def _save_azure_status(self) -> None:
        """Sauvegarde le statut Azure dans le fichier de configuration."""
        config_dir = Path.home() / ".eldercare_assistant"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / "azure_status.json"
        
        try:
            status = {
                "last_checked": datetime.now().isoformat(),
                "azure_available": self.azure_status["available"],
                "use_fallback": self.azure_status["rate_limited"],
                "rate_limit_info": {
                    "reset_time": self.azure_status["reset_time"],
                    "last_error": self.azure_status["last_error"]
                }
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Statut Azure sauvegardé dans {config_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du statut Azure: {str(e)}")


# Test simple
async def main():
    """Fonction de test pour le module multi-fournisseurs."""
    print("=== Test du client multi-fournisseurs ===")
    
    # Créer le client
    client = MultiProviderLLM()
    
    # Test de génération de réponse
    messages = [
        {"role": "system", "content": "Tu es un assistant ElderCare."},
        {"role": "user", "content": "Bonjour, comment vas-tu ?"}
    ]
    
    print("\nEnvoi de la requête...")
    response, model = await client.generate_response(messages)
    
    print(f"\nQuestion: Bonjour, comment vas-tu ?")
    print(f"Réponse: {response}")
    print(f"Modèle utilisé: {model}")
    
    print("\nStatut des fournisseurs:")
    print(f"- Mode: {client.mode}")
    print(f"- Azure disponible: {client.azure_client is not None}")
    print(f"- GitHub AI disponible: {client.github_client is not None if client.github_client else False}")
    print(f"- Azure en limite: {client.azure_status['rate_limited']}")
    if client.azure_status["reset_time"]:
        print(f"- Reset Azure prévu: {client.azure_status['reset_time']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur. Au revoir!")
    except Exception as e:
        print(f"Erreur: {str(e)}") 