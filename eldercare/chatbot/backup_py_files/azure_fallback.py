"""
Module de fallback pour Azure AI
Permet de basculer automatiquement en mode local lorsque la limite d'API est atteinte
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

# Vérifier si Azure AI SDK est disponible
try:
    from azure.ai.inference import AIClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure AI SDK n'est pas installé. Mode local activé par défaut.")

class SimpleLocalLLM:
    """
    Modèle local simple pour répondre aux questions basiques.
    Utilisé comme fallback quand Azure n'est pas disponible.
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


class AzureFallbackLLM:
    """
    Wrapper pour Azure AI avec fallback vers un modèle local.
    Gère automatiquement les limites d'API et les erreurs.
    """
    
    def __init__(self):
        """Initialise le wrapper Azure avec fallback."""
        self.azure_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_INFERENCE_API_KEY")
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
        self.azure_client = None
        self.local_model = SimpleLocalLLM()
        
        # Mode de fonctionnement (azure, local, auto)
        self.mode = os.getenv("ASSISTANT_MODE", "auto").lower()
        
        # État d'Azure
        self.azure_status = {
            "available": False,
            "rate_limited": False,
            "reset_time": None,
            "last_error": None
        }
        
        # Initialiser et vérifier les statuts
        self._load_azure_status()
        
        logger.info(f"Assistant initialisé en mode {self.mode}")
        if self.mode == "local" or not AZURE_AVAILABLE:
            logger.info("Mode local activé - Azure AI ne sera pas utilisé")
        elif self.azure_status["rate_limited"]:
            reset_time = self.azure_status["reset_time"]
            logger.warning(f"Azure AI en limite de requêtes - Mode local activé temporairement (Reset: {reset_time})")
    
    async def initialize(self) -> bool:
        """
        Initialise le client Azure AI si disponible et autorisé.
        
        Returns:
            bool: True si Azure est disponible, False sinon
        """
        # Si on est en mode local forcé ou que la SDK n'est pas disponible
        if self.mode == "local" or not AZURE_AVAILABLE:
            self.azure_status["available"] = False
            return False
            
        # Si les informations de connexion sont manquantes
        if not (self.azure_endpoint and self.azure_api_key):
            logger.warning("Informations de connexion Azure AI manquantes")
            self.azure_status["available"] = False
            self.azure_status["last_error"] = "Configuration manquante"
            return False
            
        # Vérifier si on est en limite de requêtes
        if self.azure_status["rate_limited"]:
            # Vérifier si la période de reset est passée
            if self.azure_status["reset_time"]:
                try:
                    reset_time = datetime.fromisoformat(self.azure_status["reset_time"])
                    if datetime.now() < reset_time:
                        logger.warning(f"Azure AI toujours en limite de requêtes jusqu'à {reset_time}")
                        return False
                except:
                    # Si le format de date est incorrect, on réinitialise
                    pass
        
        # Tenter d'initialiser le client Azure
        try:
            self.azure_client = AIClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_api_key)
            )
            logger.info("Client Azure AI initialisé avec succès")
            self.azure_status["available"] = True
            self.azure_status["rate_limited"] = False
            self.azure_status["reset_time"] = None
            self._save_azure_status()
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client Azure: {str(e)}")
            self.azure_status["available"] = False
            self.azure_status["last_error"] = str(e)
            self._save_azure_status()
            return False
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> Tuple[str, str]:
        """
        Génère une réponse en utilisant Azure AI ou le modèle local selon la disponibilité.
        
        Args:
            messages: Liste de messages pour la conversation
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            
        Returns:
            Tuple[str, str]: (réponse, modèle utilisé)
        """
        # Si on est en mode local forcé
        if self.mode == "local":
            return await self._use_local_model(messages), "local"
            
        # Si on est en mode auto et que Azure est disponible, on tente de l'utiliser
        if self.mode == "auto" and self.azure_status["available"] and not self.azure_status["rate_limited"]:
            try:
                response = await self._use_azure_model(messages, temperature, max_tokens)
                return response, f"azure/{self.deployment_name}"
            except HttpResponseError as e:
                # Si on atteint la limite de requêtes
                if e.status_code == 429:
                    logger.warning("Limite de requêtes Azure AI atteinte!")
                    self.azure_status["rate_limited"] = True
                    
                    # Calculer le temps de reset
                    retry_after = e.headers.get("Retry-After")
                    if retry_after:
                        try:
                            seconds = int(retry_after)
                            reset_time = datetime.now() + timedelta(seconds=seconds)
                            self.azure_status["reset_time"] = reset_time.isoformat()
                            logger.info(f"Reset prévu à {reset_time}")
                        except:
                            pass
                            
                    self._save_azure_status()
                    
                    # Fallback vers le modèle local
                    return await self._use_local_model(messages), "local (fallback - rate limit)"
                else:
                    logger.error(f"Erreur HTTP lors de l'appel à Azure: {e.status_code} - {e.message}")
                    return await self._use_local_model(messages), "local (fallback - error)"
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à Azure: {str(e)}")
                return await self._use_local_model(messages), "local (fallback - error)"
        
        # Par défaut, on utilise le modèle local
        return await self._use_local_model(messages), "local"
    
    async def _use_azure_model(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Utilise Azure AI pour générer une réponse.
        
        Args:
            messages: Liste de messages pour la conversation
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            
        Returns:
            str: Réponse générée
        """
        if not self.azure_client:
            success = await self.initialize()
            if not success:
                raise Exception("Client Azure AI non disponible")
                
        # Appel à Azure OpenAI
        response = await asyncio.to_thread(
            self.azure_client.chat.completions.create,
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def _use_local_model(self, messages: List[Dict[str, str]]) -> str:
        """
        Utilise le modèle local pour générer une réponse.
        
        Args:
            messages: Liste de messages pour la conversation
            
        Returns:
            str: Réponse générée
        """
        # Extraire le dernier message de l'utilisateur
        user_message = ""
        for message in reversed(messages):
            if message["role"] == "user":
                user_message = message["content"]
                break
                
        return self.local_model.generate_response(user_message)
    
    def _load_azure_status(self) -> None:
        """Charge le statut d'Azure depuis le fichier de configuration."""
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
                    
                    logger.info(f"Statut Azure chargé: Disponible={self.azure_status['available']}, "
                             f"Limité={self.azure_status['rate_limited']}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du statut Azure: {str(e)}")
    
    def _save_azure_status(self) -> None:
        """Sauvegarde le statut d'Azure dans le fichier de configuration."""
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
    """Fonction de test pour le module de fallback."""
    print("=== Test du module de fallback Azure ===")
    
    fallback_llm = AzureFallbackLLM()
    await fallback_llm.initialize()
    
    # Test de génération de réponse
    messages = [
        {"role": "system", "content": "Tu es un assistant ElderCare."},
        {"role": "user", "content": "Bonjour, comment vas-tu ?"}
    ]
    
    response, model = await fallback_llm.generate_response(messages)
    
    print(f"\nQuestion: Bonjour, comment vas-tu ?")
    print(f"Réponse: {response}")
    print(f"Modèle utilisé: {model}")
    
    print("\nStatut Azure:")
    for key, value in fallback_llm.azure_status.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur. Au revoir!")
    except Exception as e:
        print(f"Erreur: {str(e)}") 