"""
Script de configuration du token GitHub pour l'assistant ElderCare
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path

def get_token_from_args():
    """Obtient le token GitHub à partir des arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Configuration du token GitHub pour l'assistant ElderCare")
    parser.add_argument('--token', help="Token GitHub (avec autorisation models:read)")
    parser.add_argument('--save', action='store_true', help="Enregistrer le token de manière permanente")
    parser.add_argument('--test', action='store_true', help="Tester la connexion à GitHub AI")
    parser.add_argument('--update-env', action='store_true', help="Mettre à jour le fichier azure_config.env")
    
    args = parser.parse_args()
    return args

def set_token_env(token):
    """Définit le token GitHub comme variable d'environnement."""
    os.environ["GITHUB_TOKEN"] = token
    return token

def save_token_to_config(token):
    """Enregistre le token GitHub dans le fichier de configuration."""
    config_dir = Path.home() / ".eldercare_assistant"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "github_config.json"
    
    config = {
        "github_token": token
    }
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"✅ Token GitHub enregistré dans {config_file}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement du token: {str(e)}")
        return False

def update_azure_config_env(mode="github"):
    """Met à jour le fichier azure_config.env pour utiliser GitHub AI."""
    try:
        env_file = Path("azure_config.env")
        if not env_file.exists():
            print("❌ Fichier azure_config.env non trouvé.")
            return False
            
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remplacer le mode
        content = content.replace("ASSISTANT_MODE=azure", f"ASSISTANT_MODE={mode}")
        content = content.replace("ASSISTANT_MODE=local", f"ASSISTANT_MODE={mode}")
        content = content.replace("ASSISTANT_MODE=auto", f"ASSISTANT_MODE={mode}")
        
        # Ajouter le token GitHub s'il n'existe pas déjà
        if "GITHUB_TOKEN" not in content:
            content += f"\n# Token GitHub pour l'API GitHub AI\n"
            content += f"GITHUB_TOKEN={os.environ.get('GITHUB_TOKEN', '')}\n"
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Mode assistant mis à jour: {mode}")
        print(f"✅ Configuration GitHub ajoutée à {env_file}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour du fichier de configuration: {str(e)}")
        return False

async def test_github_ai():
    """Teste la connexion à GitHub AI."""
    try:
        # Import du module GitHub AI
        try:
            from github_ai_integration import GitHubAIClient
        except ImportError:
            print("❌ Module github_ai_integration.py non trouvé.")
            return False
        
        print("🔄 Test de connexion à GitHub AI...")
        
        # Créer le client
        github_client = GitHubAIClient()
        
        if not github_client.is_available():
            print("❌ Client GitHub AI non disponible.")
            print("   Vérifiez que le token est correctement configuré et que le SDK Azure AI est installé.")
            return False
        
        # Tester une requête simple
        messages = [
            {"role": "system", "content": "Tu es un assistant de test."},
            {"role": "user", "content": "Dis simplement 'Connexion GitHub AI réussie'"}
        ]
        
        response, model = await github_client.generate_response(messages, temperature=0.1, max_tokens=20)
        
        print(f"✅ Connexion à GitHub AI réussie!")
        print(f"📝 Réponse: {response}")
        print(f"💡 Modèle utilisé: {model}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de GitHub AI: {str(e)}")
        return False

def print_setup_instructions():
    """Affiche les instructions de configuration."""
    print("\n📋 Instructions de configuration du token GitHub:")
    print("1. Allez sur https://github.com/settings/tokens")
    print("2. Cliquez sur 'Generate new token (classic)'")
    print("3. Donnez un nom à votre token (ex: 'ElderCare Assistant')")
    print("4. Accordez l'autorisation 'models:read'")
    print("5. Cliquez sur 'Generate token'")
    print("6. Copiez le token généré et utilisez-le dans ce script:")
    print("   python setup_github_token.py --token=votre_token_ici --save --test --update-env")
    print("\nAlternativement, définissez la variable d'environnement directement:")
    print("- Windows PowerShell: $Env:GITHUB_TOKEN = \"votre_token_ici\"")
    print("- Windows CMD: set GITHUB_TOKEN=votre_token_ici")
    print("- Linux/Mac: export GITHUB_TOKEN=\"votre_token_ici\"")

async def main():
    """Fonction principale."""
    print("=== Configuration du token GitHub pour ElderCare Assistant ===")
    
    args = get_token_from_args()
    
    # Si pas de token spécifié, vérifier s'il existe dans l'environnement
    token = args.token or os.environ.get("GITHUB_TOKEN")
    
    if not token:
        print("❌ Aucun token GitHub spécifié.")
        print_setup_instructions()
        return
    
    # Définir le token comme variable d'environnement
    set_token_env(token)
    
    # Sauvegarder le token si demandé
    if args.save:
        save_token_to_config(token)
    
    # Mettre à jour le fichier de configuration si demandé
    if args.update_env:
        update_azure_config_env()
    
    # Tester la connexion si demandé
    if args.test:
        await test_github_ai()
    
    print("\n✅ Configuration terminée!")
    print("Vous pouvez maintenant utiliser l'assistant ElderCare avec GitHub AI.")
    print("Pour lancer l'assistant, exécutez: python run_eldercare_assistant.py")
    print("Le système utilisera GitHub AI au lieu d'Azure si la configuration est correcte.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterruption par l'utilisateur. Au revoir!")
    except Exception as e:
        print(f"Erreur: {str(e)}") 