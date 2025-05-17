"""
Version simplifiée du service d'exécution de commandes pour ElderCare Assistant.
"""

import os
import sys
import subprocess
import logging
from typing import Dict, List, Tuple, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommandeService:
    """Service simplifié pour exécuter des commandes système avec messages de confirmation."""
    
    def __init__(self, lang: str = "fr"):
        """
        Initialisation du service de commandes.
        
        Args:
            lang: Code de langue pour l'interface (fr ou en)
        """
        self.os_type = sys.platform
        self.lang = lang
        self.commandes = self._initialiser_commandes_par_defaut()
        logger.info(f"Service de commandes initialisé pour la plateforme: {self.os_type}")
    
    def _initialiser_commandes_par_defaut(self) -> Dict[str, str]:
        """Initialise la liste des commandes disponibles selon le système d'exploitation."""
        if self.os_type == 'win32':  # Windows
            return {
                "chrome": "start chrome",
                "firefox": "start firefox",
                "edge": "start msedge",
                "notepad": "start notepad",
                "bloc-notes": "start notepad",
                "calculatrice": "start calc",
                "calculator": "start calc",
                "explorateur": "start explorer",
                "explorer": "start explorer",
                "météo": "start ms-weather:",
                "weather": "start ms-weather:",
                "word": "start winword",
                "excel": "start excel",
                "paint": "start mspaint",
                "terminal": "start wt",
                "powershell": "start powershell",
                "cmd": "start cmd"
            }
        elif self.os_type == 'darwin':  # macOS
            return {
                "chrome": "open -a 'Google Chrome'",
                "firefox": "open -a Firefox",
                "safari": "open -a Safari",
                "calculatrice": "open -a Calculator",
                "calculator": "open -a Calculator",
                "terminal": "open -a Terminal",
                "finder": "open -a Finder",
                "explorateur": "open -a Finder",
                "explorer": "open -a Finder"
            }
        else:  # Linux
            return {
                "chrome": "google-chrome",
                "firefox": "firefox",
                "calculatrice": "gnome-calculator",
                "calculator": "gnome-calculator",
                "explorateur": "nautilus",
                "explorer": "nautilus",
                "terminal": "gnome-terminal"
            }
    
    def executer_commande(self, app_name: str) -> Tuple[bool, str, str]:
        """
        Exécute une commande pour l'application spécifiée.
        
        Args:
            app_name: Nom de l'application à ouvrir/fermer
            
        Returns:
            Tuple de (succès: bool, message: str, commande: str)
        """
        # Normaliser le nom de l'application
        app_name_lower = app_name.lower().strip()
        
        # Vérifier si c'est une commande pour fermer une application
        fermer = False
        fermer_patterns = ["fermer", "ferme", "close", "kill", "stop", "arrête", "quitte", "terminer"]
        
        for pattern in fermer_patterns:
            if pattern in app_name_lower:
                fermer = True
                break
        
        # Vérifier si l'application est dans la liste des applications autorisées
        found = False
        for app in self.commandes.keys():
            if app in app_name_lower:
                app_name_lower = app
                found = True
                break
                
        if not found:
            return False, f"Application '{app_name_lower}' non trouvée.", ""
        
        # Obtenir la commande pour l'application
        if fermer:
            if self.os_type == 'win32':
                commande = f"taskkill /F /IM {self.commandes[app_name_lower].split()[-1]}.exe"
            elif self.os_type == 'darwin':
                app_mac = self.commandes[app_name_lower].replace('open -a ', '').strip("'")
                commande = f"pkill -x '{app_mac}'"
            else:
                commande = f"pkill -f {self.commandes[app_name_lower].split()[0]}"
        else:
            commande = self.commandes[app_name_lower]
        
        try:
            # Exécuter la commande
            if self.os_type == 'win32':
                subprocess.Popen(commande, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(
                    commande, 
                    shell=True, 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            logger.info(f"Commande exécutée avec succès: {commande}")
            
            if fermer:
                return True, f"J'ai fermé {app_name_lower} pour vous.", commande
            else:
                return True, f"J'ai ouvert {app_name_lower} pour vous.", commande
            
        except Exception as e:
            erreur = f"Erreur lors de l'exécution de la commande pour '{app_name_lower}': {str(e)}"
            logger.error(erreur)
            return False, erreur, commande 