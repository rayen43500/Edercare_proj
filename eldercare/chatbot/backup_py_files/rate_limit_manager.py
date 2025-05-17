"""
Gestionnaire de limites de requêtes Azure
Ce script permet de vérifier et gérer les limites d'API Azure
"""

import os
import json
import datetime
import argparse
from pathlib import Path

def seconds_to_human_readable(seconds):
    """Convertit un nombre de secondes en format lisible par un humain."""
    if seconds < 60:
        return f"{seconds} secondes"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes} minutes et {remaining_seconds} secondes"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{hours} heures, {minutes} minutes et {seconds} secondes"

def get_config_path():
    """Retourne le chemin vers le fichier de configuration Azure."""
    return Path(os.path.expanduser("~")) / ".eldercare_assistant" / "azure_status.json"

def read_status():
    """Lit le statut actuel d'Azure depuis le fichier de configuration."""
    config_path = get_config_path()
    if not config_path.exists():
        print("❌ Aucun fichier de statut trouvé.")
        return None
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier de statut: {str(e)}")
        return None

def update_status(status=None, mode="local"):
    """Met à jour le statut Azure et le mode de fonctionnement."""
    if status is None:
        # Créer un nouveau statut par défaut
        status = {
            "last_checked": datetime.datetime.now().isoformat(),
            "azure_available": False,
            "use_fallback": True,
            "rate_limit_info": {
                "remaining_tokens": "0",
                "remaining_requests": "0",
                "reset_time": None,
                "last_error": "Manually set to use fallback"
            }
        }
    
    # Enregistrer le statut
    config_path = get_config_path()
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
        print(f"✅ Statut Azure mis à jour: {config_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour du statut: {str(e)}")
    
    # Mettre à jour le mode dans le fichier de configuration
    try:
        env_file = Path("azure_config.env")
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remplacer le mode
            content = content.replace("ASSISTANT_MODE=azure", f"ASSISTANT_MODE={mode}")
            content = content.replace("ASSISTANT_MODE=local", f"ASSISTANT_MODE={mode}")
            content = content.replace("ASSISTANT_MODE=auto", f"ASSISTANT_MODE={mode}")
            
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Mode assistant mis à jour: {mode}")
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour du mode: {str(e)}")

def check_rate_limit_status():
    """Vérifie si la limite d'API Azure a été atteinte et affiche le temps restant."""
    status = read_status()
    if not status:
        print("❌ Impossible de déterminer le statut de la limite d'API.")
        return
    
    print("\n📊 Statut de la connexion Azure:")
    print(f"- Dernière vérification: {status.get('last_checked', 'Inconnue')}")
    print(f"- Azure disponible: {'✅' if status.get('azure_available', False) else '❌'}")
    print(f"- Mode dégradé activé: {'✅' if status.get('use_fallback', False) else '❌'}")
    
    rate_limit_info = status.get('rate_limit_info', {})
    last_error = rate_limit_info.get('last_error')
    reset_time = rate_limit_info.get('reset_time')
    
    if last_error:
        print(f"- Dernière erreur: {last_error}")
    
    if reset_time:
        try:
            # Essayer de traiter comme un nombre de secondes
            seconds = int(reset_time)
            print(f"- Temps avant réinitialisation: {seconds_to_human_readable(seconds)}")
            
            # Calculer la date/heure de réinitialisation
            try:
                last_checked = datetime.datetime.fromisoformat(status.get('last_checked'))
                reset_datetime = last_checked + datetime.timedelta(seconds=seconds)
                print(f"- Réinitialisation prévue le: {reset_datetime.strftime('%Y-%m-%d à %H:%M:%S')}")
            except:
                pass
        except:
            # Si ce n'est pas un nombre, afficher tel quel
            print(f"- Information de réinitialisation: {reset_time}")

def reset_status():
    """Réinitialise le statut Azure pour forcer une nouvelle vérification."""
    status = {
        "last_checked": datetime.datetime.now().isoformat(),
        "azure_available": True,
        "use_fallback": False,
        "rate_limit_info": {
            "remaining_tokens": None,
            "remaining_requests": None,
            "reset_time": None,
            "last_error": None
        }
    }
    
    update_status(status, mode="auto")
    print("✅ Statut Azure réinitialisé avec succès!")
    print("ℹ️ L'assistant tentera d'utiliser Azure lors du prochain démarrage.")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Gestion des limites d'API Azure")
    parser.add_argument('--check', action='store_true', help="Vérifier le statut actuel")
    parser.add_argument('--reset', action='store_true', help="Réinitialiser le statut (forcer l'utilisation d'Azure)")
    parser.add_argument('--force-local', action='store_true', help="Forcer l'utilisation du mode local")
    parser.add_argument('--force-azure', action='store_true', help="Forcer l'utilisation du mode Azure")
    parser.add_argument('--auto', action='store_true', help="Utiliser le mode automatique")
    
    args = parser.parse_args()
    
    if args.check:
        check_rate_limit_status()
    elif args.reset:
        reset_status()
    elif args.force_local:
        update_status(mode="local")
    elif args.force_azure:
        update_status(mode="azure")
    elif args.auto:
        update_status(mode="auto")
    else:
        # Par défaut, vérifier le statut
        check_rate_limit_status()
        
        print("\n💡 Options disponibles:")
        print("  --check       : Vérifier le statut actuel")
        print("  --reset       : Réinitialiser pour utiliser Azure")
        print("  --force-local : Forcer le mode local")
        print("  --force-azure : Forcer le mode Azure")
        print("  --auto        : Utiliser le mode automatique")

if __name__ == "__main__":
    main() 