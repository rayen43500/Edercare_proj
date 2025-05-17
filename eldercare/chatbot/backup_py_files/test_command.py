"""
Script de test pour l'exécution des commandes système.
"""

import asyncio
from system_commands import SystemCommandExecutor

async def main():
    """Fonction principale pour tester l'exécution des commandes."""
    executor = SystemCommandExecutor()
    
    print("\n=== Test d'exécution de commandes système ===\n")
    print("Applications disponibles :")
    for app in executor.get_available_applications()[:10]:  # Limiter à 10 applications
        print(f"- {app}")
    
    print("\nEntrez le nom de l'application à ouvrir (ou 'quit' pour quitter) :")
    
    while True:
        user_input = input("\nCommande > ").strip().lower()
        
        if user_input in ['quit', 'exit', 'q']:
            break
            
        # Extraire le nom de l'application
        app_name = None
        command_patterns = ["ouvre ", "lance ", "démarre ", "exécute ", "open ", "start "]
        
        for pattern in command_patterns:
            if pattern in user_input:
                app_name = user_input.split(pattern)[1].strip()
                break
        
        # Si aucun motif n'est trouvé, considérer le texte entier comme nom d'application
        if not app_name:
            app_name = user_input
        
        # Exécuter la commande
        success, message = executor.execute_command(app_name)
        
        # Afficher le résultat
        if success:
            print(f"\n✅ Succès : {message}")
        else:
            print(f"\n❌ Erreur : {message}")
    
    print("\nAu revoir !")

if __name__ == "__main__":
    asyncio.run(main()) 