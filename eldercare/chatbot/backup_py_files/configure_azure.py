import os
import sys

def main():
    """Configure l'environnement pour utiliser Azure LLM uniquement."""
    print("=== Configuration de ElderCare Assistant pour Azure LLM ===")
    print("\nCe script va vous aider à configurer l'assistant pour utiliser uniquement Azure LLM.")
    
    # Vérifier si le fichier .env existe déjà
    env_file = ".env"
    if os.path.exists(env_file):
        print("\nUn fichier .env existe déjà. Voulez-vous le modifier? (o/n)")
        response = input("> ").strip().lower()
        if response != "o":
            print("Configuration annulée.")
            return
    
    # Demander le token GitHub
    print("\nPour utiliser Azure LLM, vous avez besoin d'un token GitHub avec permission 'models:read'.")
    print("Allez sur https://github.com/settings/tokens pour créer un token.")
    github_token = input("\nEntrez votre token GitHub: ").strip()
    
    if not github_token:
        print("Aucun token fourni. Configuration annulée.")
        return
    
    # Créer ou mettre à jour le fichier .env
    with open(env_file, "w") as f:
        f.write(f"# Token GitHub avec permission models:read pour Azure LLM\n")
        f.write(f"GITHUB_TOKEN={github_token}\n\n")
        f.write(f"# Autres clés API (non nécessaires pour le mode Azure uniquement)\n")
        f.write(f"# GEMINI_API_KEY=votre_cle_gemini_ici\n")
        f.write(f"# OPENAI_API_KEY=votre_cle_openai_ici\n")
    
    print(f"\nConfiguration terminée! Le fichier {env_file} a été créé/mis à jour.")
    print("\nPour lancer l'assistant en mode Azure uniquement, exécutez:")
    print("python run_azure_only.py")

if __name__ == "__main__":
    main() 