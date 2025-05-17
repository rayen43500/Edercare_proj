"""
Script de test pour la connexion Azure AI
Vérifie la validité des clés API et les limites de requêtes
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Essayer d'importer Azure AI SDK
try:
    from azure.ai.inference import AIClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError, ServiceResponseError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️ Azure AI SDK n'est pas installé. Exécutez: pip install azure-ai-inference")

# Chargement des variables d'environnement
load_dotenv('.env')
load_dotenv('azure_config.env')  # Priorité aux variables de azure_config.env

class AzureConnectionTester:
    """Classe pour tester la connexion à Azure AI."""
    
    def __init__(self):
        """Initialise le testeur de connexion Azure."""
        self.azure_endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_INFERENCE_API_KEY")
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
        self.client = None
        self.rate_limit_info = {
            "remaining_tokens": None,
            "remaining_requests": None,
            "reset_time": None,
            "last_error": None
        }
        
        # Vérification de la configuration
        if not (self.azure_endpoint and self.azure_api_key):
            print("❌ Configuration Azure manquante!")
            print("   La variable d'environnement AZURE_INFERENCE_ENDPOINT ou AZURE_API_KEY n'est pas définie.")
            print("\nCréez un fichier .env ou azure_config.env avec le contenu suivant:")
            print("AZURE_INFERENCE_ENDPOINT=https://votre-endpoint.region.inference.ai.azure.com")
            print("AZURE_INFERENCE_API_KEY=votre_clé_api")
            print("AZURE_DEPLOYMENT_NAME=gpt-4")  # Optionnel
    
    async def initialize_client(self):
        """Initialise le client Azure AI."""
        if not AZURE_AVAILABLE:
            print("❌ Azure AI SDK non disponible. Installez-le avec: pip install azure-ai-inference")
            return False
            
        if not (self.azure_endpoint and self.azure_api_key):
            print("❌ Clés API Azure manquantes")
            return False
            
        try:
            self.client = AIClient(
                endpoint=self.azure_endpoint,
                credential=AzureKeyCredential(self.azure_api_key)
            )
            print("✅ Client Azure AI initialisé avec succès")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation du client Azure: {str(e)}")
            return False
    
    async def test_connection(self):
        """Teste la connexion à Azure en envoyant une requête simple."""
        if not self.client:
            success = await self.initialize_client()
            if not success:
                return False
                
        try:
            print("\n🔄 Test de connexion à Azure AI en cours...")
            # Requête simple pour tester
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "Tu es un assistant de test."},
                    {"role": "user", "content": "Dis simplement 'Connexion réussie'"}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            # Extraction des informations de limitation d'API des en-têtes
            self._extract_rate_limit_info(response)
            
            content = response.choices[0].message.content
            print(f"✅ Connexion à Azure AI réussie!")
            print(f"📝 Réponse: {content}")
            print(f"💡 Modèle utilisé: {response.model}")
            
            self._display_rate_limit_info()
            return True
            
        except HttpResponseError as e:
            if e.status_code == 429:
                print("❌ Limite de requêtes Azure AI atteinte!")
                retry_after = e.headers.get('Retry-After', 'inconnu')
                self.rate_limit_info["last_error"] = f"Rate limit reached. Retry after: {retry_after} seconds"
                self.rate_limit_info["reset_time"] = retry_after
                
                print(f"⏰ Attendez {retry_after} secondes avant de réessayer")
                print("💡 Conseil: Utilisez le mode local en attendant que la limite soit réinitialisée")
            else:
                print(f"❌ Erreur HTTP lors du test de connexion: {e.status_code} - {e.message}")
                self.rate_limit_info["last_error"] = f"HTTP Error {e.status_code}: {e.message}"
            
            self._display_rate_limit_info()
            return False
        except Exception as e:
            print(f"❌ Erreur lors du test de connexion: {str(e)}")
            self.rate_limit_info["last_error"] = str(e)
            return False
    
    def _extract_rate_limit_info(self, response):
        """Extrait les informations de limitation d'API des en-têtes de réponse."""
        if hasattr(response, 'headers'):
            headers = response.headers
            self.rate_limit_info["remaining_tokens"] = headers.get('x-ratelimit-remaining-tokens')
            self.rate_limit_info["remaining_requests"] = headers.get('x-ratelimit-remaining-requests')
            # Si d'autres informations sont disponibles dans les en-têtes
    
    def _display_rate_limit_info(self):
        """Affiche les informations de limitation d'API."""
        print("\n📊 Informations sur les limites d'API Azure:")
        if self.rate_limit_info["remaining_tokens"]:
            print(f"- Tokens restants: {self.rate_limit_info['remaining_tokens']}")
        if self.rate_limit_info["remaining_requests"]:
            print(f"- Requêtes restantes: {self.rate_limit_info['remaining_requests']}")
        if self.rate_limit_info["reset_time"]:
            print(f"- Temps avant réinitialisation: {self.rate_limit_info['reset_time']} secondes")
        if self.rate_limit_info["last_error"]:
            print(f"- Dernière erreur: {self.rate_limit_info['last_error']}")
    
    def create_fallback_config(self):
        """Crée un fichier de configuration pour le mode dégradé (fallback)."""
        config_dir = Path.home() / ".eldercare_assistant"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / "azure_status.json"
        
        status = {
            "last_checked": datetime.now().isoformat(),
            "azure_available": self.client is not None,
            "rate_limit_info": self.rate_limit_info,
            "use_fallback": self.rate_limit_info["last_error"] is not None
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
            
        print(f"\n💾 Configuration de fallback créée: {config_file}")
        print("📋 Cette configuration indique à l'assistant d'utiliser le mode local")
        print("   si Azure n'est pas disponible ou si la limite de requêtes est atteinte.")


async def main():
    """Fonction principale."""
    print("🔍 Test de connexion à Azure AI")
    print("===============================")
    
    # Vérifier les variables d'environnement
    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    api_key = os.getenv("AZURE_INFERENCE_API_KEY")
    
    print("📋 Configuration actuelle:")
    print(f"- AZURE_INFERENCE_ENDPOINT: {'✅ Défini' if endpoint else '❌ Non défini'}")
    print(f"- AZURE_INFERENCE_API_KEY: {'✅ Défini' if api_key else '❌ Non défini'}")
    print(f"- AZURE_DEPLOYMENT_NAME: {os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-4')} (par défaut: gpt-4)")
    
    # Tester la connexion
    tester = AzureConnectionTester()
    await tester.test_connection()
    
    # Créer la configuration pour le mode dégradé
    tester.create_fallback_config()
    
    print("\n💡 Astuces pour résoudre les problèmes:")
    print("1. Vérifiez vos clés API et endpoint Azure")
    print("2. Si vous avez atteint la limite de requêtes, attendez la réinitialisation")
    print("3. Utilisez un abonnement Azure avec des limites plus élevées")
    print("4. Activez le mode local pour continuer à utiliser l'assistant sans Azure")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur. Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}") 