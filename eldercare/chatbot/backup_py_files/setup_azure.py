import os

def main():
    """Configure l'assistant pour utiliser Azure"""
    print("=== Configuration d'Azure pour ElderCare Assistant ===")
    
    # Demander le token GitHub
    print("\nEntrez votre token GitHub avec permission 'models:read':")
    token = input("> ").strip()
    
    if not token:
        print("Aucun token fourni. Configuration annulée.")
        return
    
    # Écrire le token dans un fichier .env avec encodage UTF-8
    with open(".env", "w", encoding="utf-8") as f:
        f.write(f"GITHUB_TOKEN={token}\n")
    
    print("\nConfiguration terminée!")
    print("Pour lancer l'assistant avec Azure, exécutez:")
    print("python run_azure_only.py")

if __name__ == "__main__":
    main() 