#!/usr/bin/env python3
"""
Script de nettoyage qui supprime tous les fichiers Python inutiles
et garde uniquement ceux nécessaires pour exécuter eldercare_complete.py
"""

import os
import shutil
import glob

# Les fichiers essentiels à garder
ESSENTIAL_FILES = [
    "eldercare_complete.py",      # Script principal
    "intent_detector.py",         # Détection d'intentions
    "action_handler.py",          # Gestionnaire d'actions
    "azure_llm_integration.py",   # Intégration Azure
    "commande_service_mini.py",   # Service de commandes simplifié utilisé par action_handler
    "eldercare_run.py",           # Script de lancement simplifié
    "system_commands.py",         # Commandes système utilisées par action_handler
    "cleanup_project.py"          # Ce script de nettoyage
]

# Dossiers à conserver
ESSENTIAL_DIRS = [
    "knowledge_base",             # Base de connaissances (si utilisée)
    "faiss_index"                 # Index FAISS (si utilisé)
]

# Fichiers de configuration importants (non Python)
ESSENTIAL_CONFIG = [
    "azure_config.env",
    ".env",
    "requirements.txt"
]

def main():
    """Fonction principale qui nettoie le projet."""
    print("\n=== Nettoyage du projet - ElderCare Assistant ===")
    print("Ce script va supprimer tous les fichiers Python non essentiels.")
    print(f"Seuls ces fichiers seront conservés: {', '.join(ESSENTIAL_FILES)}")
    
    # Demander confirmation
    response = input("\nÊtes-vous sûr de vouloir continuer? (o/n): ").strip().lower()
    if response != 'o':
        print("Opération annulée.")
        return
    
    # Lister tous les fichiers Python
    all_py_files = glob.glob("*.py")
    
    # Identifier les fichiers à supprimer
    files_to_delete = [f for f in all_py_files if f not in ESSENTIAL_FILES]
    
    # Créer un dossier de sauvegarde
    backup_dir = "backup_py_files"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Déplacer les fichiers vers la sauvegarde et supprimer
    for file in files_to_delete:
        try:
            print(f"Sauvegarde et suppression de: {file}")
            shutil.copy2(file, os.path.join(backup_dir, file))
            os.remove(file)
        except Exception as e:
            print(f"Erreur lors de la suppression de {file}: {e}")
    
    # Afficher les résultats
    print("\n=== Nettoyage terminé ===")
    print(f"Fichiers sauvegardés dans: {backup_dir}")
    print(f"Fichiers supprimés: {len(files_to_delete)}")
    print("Fichiers conservés:")
    for file in ESSENTIAL_FILES:
        if os.path.exists(file):
            print(f"  - {file}")
    
    # Vérifier les fichiers de configuration
    print("\nFichiers de configuration:")
    for config in ESSENTIAL_CONFIG:
        if os.path.exists(config):
            print(f"  - {config} (présent)")
        else:
            print(f"  - {config} (manquant)")
    
    print("\nLe projet a été nettoyé. Vous pouvez maintenant exécuter:")
    print("python eldercare_run.py")

if __name__ == "__main__":
    main() 