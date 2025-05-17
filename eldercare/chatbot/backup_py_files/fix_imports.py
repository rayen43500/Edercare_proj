"""
Script pour corriger les imports obsolètes dans le projet ElderCare Assistant.
Ce script met à jour les imports déprécés de langchain vers langchain_community.
"""

import os
import re
import glob

def fix_imports_in_file(file_path):
    """
    Corrige les imports obsolètes dans un fichier.
    
    Args:
        file_path: Chemin du fichier à corriger
    
    Returns:
        bool: True si des modifications ont été effectuées, False sinon
    """
    print(f"Traitement du fichier: {file_path}")
    
    # Mapping des imports obsolètes vers les nouveaux imports
    import_mapping = {
        r"from langchain\.embeddings import (.*?)": r"from langchain_community.embeddings import \1",
        r"from langchain\.vectorstores import (.*?)": r"from langchain_community.vectorstores import \1",
        r"from langchain\.document_loaders import (.*?)": r"from langchain_community.document_loaders import \1",
        r"from langchain\.llms import (.*?)": r"from langchain_community.llms import \1",
        r"from langchain\.chat_models import (.*?)": r"from langchain_community.chat_models import \1",
        r"from langchain\.retrievers import (.*?)": r"from langchain_community.retrievers import \1",
        r"from langchain\.chains import (.*?)": r"from langchain_community.chains import \1",
    }
    
    # Lire le contenu du fichier
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            # Essayer avec un autre encodage
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"  Erreur lors de la lecture du fichier: {e}")
            return False
    
    # Faire une copie du contenu original pour détecter les modifications
    original_content = content
    
    # Appliquer les remplacements
    for pattern, replacement in import_mapping.items():
        content = re.sub(pattern, replacement, content)
    
    # Vérifier si des modifications ont été effectuées
    if content != original_content:
        # Sauvegarder le fichier avec les modifications
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Imports corrigés dans {file_path}")
            return True
        except Exception as e:
            print(f"  Erreur lors de l'écriture du fichier: {e}")
            return False
    else:
        print(f"  Aucune correction nécessaire dans {file_path}")
        return False

def main():
    """Corrige les imports dans tous les fichiers Python du projet."""
    print("=== Correction des imports obsolètes dans le projet ElderCare Assistant ===")
    
    # Récupérer tous les fichiers Python
    python_files = glob.glob("*.py")
    
    # Compter le nombre de fichiers modifiés
    modified_files = 0
    
    # Traiter chaque fichier
    for file_path in python_files:
        if fix_imports_in_file(file_path):
            modified_files += 1
    
    print(f"\nTraitement terminé. {modified_files} fichier(s) modifié(s).")
    print("Pour lancer le projet principal, exécutez:")
    print("python eldercare_assistant.py")

if __name__ == "__main__":
    main() 