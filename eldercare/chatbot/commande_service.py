"""
Service d'exécution de commandes avec interface utilisateur améliorée et fonctionnalités étendues.
Version 2.0
"""

import os
import sys
import subprocess
import asyncio
import logging
import json
import re
import locale
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("commandeservice.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Définition des constantes
CONFIG_DIR = Path.home() / ".commandeservice"
CONFIG_FILE = CONFIG_DIR / "config.json"
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m"
}

# Détection de la langue par défaut du système
SYSTEM_LANG = locale.getdefaultlocale()[0]
DEFAULT_LANG = "fr" if SYSTEM_LANG and SYSTEM_LANG.startswith("fr") else "en"

# Traductions
TRANSLATIONS = {
    "fr": {
        "service_initialized": "Service de commandes initialisé pour la plateforme: {}",
        "welcome": "=== Assistant ElderCare - Service de Commandes ===",
        "available_apps": "Applications disponibles :",
        "command_examples": "Exemples de commandes :",
        "open_example": "- ouvrir {}",
        "close_example": "- fermer {}",
        "exit_example": "- quitter (pour sortir)",
        "search_example": "- rechercher [terme] (pour chercher une application)",
        "command_prompt": "Votre commande > ",
        "goodbye": "Merci d'avoir utilisé le service de commandes. Au revoir !",
        "success_open": "Bien sûr, à votre service. J'ai ouvert {} pour vous.",
        "success_close": "Bien sûr, à votre service. J'ai fermé {} pour vous.",
        "command_executed": "Commande exécutée: {}",
        "app_not_found": "Application '{}' non trouvée.",
        "app_suggestion": "Vouliez-vous dire {} ?",
        "app_not_authorized": "Application '{}' non trouvée ou non autorisée.",
        "error_executing": "Erreur lors de l'exécution de la commande pour '{}': {}",
        "searching_for": "Recherche d'applications contenant '{}':",
        "no_results": "Aucun résultat trouvé pour '{}'",
        "config_saved": "Configuration enregistrée avec succès.",
        "config_loaded": "Configuration chargée avec succès.",
        "custom_command_added": "Commande personnalisée '{}' ajoutée avec succès.",
        "help_title": "Aide du Service de Commandes",
        "help_command": "commandes - Affiche la liste des applications disponibles",
        "help_open": "ouvrir [app] - Ouvre l'application spécifiée",
        "help_close": "fermer [app] - Ferme l'application spécifiée",
        "help_search": "rechercher [terme] - Recherche des applications",
        "help_add": "ajouter [nom] [commande] - Ajoute une commande personnalisée",
        "help_lang": "langue [fr/en] - Change la langue de l'interface",
        "help_exit": "quitter/exit - Quitte le programme",
        "help_help": "aide/help - Affiche cette aide",
        "language_changed": "Langue changée en {}",
        "unknown_command": "Commande inconnue. Tapez 'aide' pour voir les commandes disponibles."
    },
    "en": {
        "service_initialized": "Command service initialized for platform: {}",
        "welcome": "=== ElderCare Assistant - Command Service ===",
        "available_apps": "Available applications:",
        "command_examples": "Command examples:",
        "open_example": "- open {}",
        "close_example": "- close {}",
        "exit_example": "- exit (to quit)",
        "search_example": "- search [term] (to find an application)",
        "command_prompt": "Your command > ",
        "goodbye": "Thank you for using the command service. Goodbye!",
        "success_open": "Of course, at your service. I opened {} for you.",
        "success_close": "Of course, at your service. I closed {} for you.",
        "command_executed": "Command executed: {}",
        "app_not_found": "Application '{}' not found.",
        "app_suggestion": "Did you mean {} ?",
        "app_not_authorized": "Application '{}' not found or not authorized.",
        "error_executing": "Error executing command for '{}': {}",
        "searching_for": "Searching for applications containing '{}':",
        "no_results": "No results found for '{}'",
        "config_saved": "Configuration saved successfully.",
        "config_loaded": "Configuration loaded successfully.",
        "custom_command_added": "Custom command '{}' added successfully.",
        "help_title": "Command Service Help",
        "help_command": "commands - Display the list of available applications",
        "help_open": "open [app] - Open the specified application",
        "help_close": "close [app] - Close the specified application",
        "help_search": "search [term] - Search for applications",
        "help_add": "add [name] [command] - Add a custom command",
        "help_lang": "language [fr/en] - Change interface language",
        "help_exit": "exit/quit - Exit the program",
        "help_help": "help - Display this help",
        "language_changed": "Language changed to {}",
        "unknown_command": "Unknown command. Type 'help' to see available commands."
    }
}

class CommandeService:
    """Service pour exécuter des commandes système avec messages de confirmation."""
    
    def __init__(self, lang: str = DEFAULT_LANG):
        """
        Initialisation du service de commandes.
        
        Args:
            lang: Code de langue pour l'interface (fr ou en)
        """
        self.os_type = sys.platform
        self.lang = lang
        self.t = TRANSLATIONS[self.lang]
        self.default_commands = self._initialiser_commandes_par_defaut()
        self.custom_commands = {}
        self._charger_configuration()
        self.commandes = {**self.default_commands, **self.custom_commands}
        logger.info(self.t["service_initialized"].format(self.os_type))
    
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
                "cartes": "start bingmaps:",
                "maps": "start bingmaps:",
                "mail": "start outlookmail:",
                "email": "start outlookmail:",
                "photos": "start ms-photos:",
                "paramètres": "start ms-settings:",
                "settings": "start ms-settings:",
                "word": "start winword",
                "excel": "start excel",
                "paint": "start mspaint",
                "musique": "start mswindowsmusic:",
                "music": "start mswindowsmusic:",
                "calendrier": "start outlookcal:",
                "calendar": "start outlookcal:",
                "terminal": "start wt",
                "powershell": "start powershell",
                "cmd": "start cmd"
            }
        elif self.os_type == 'darwin':  # macOS
            return {
                "chrome": "open -a 'Google Chrome'",
                "firefox": "open -a Firefox",
                "safari": "open -a Safari",
                "mail": "open -a Mail",
                "notes": "open -a Notes", 
                "calculatrice": "open -a Calculator",
                "calculator": "open -a Calculator",
                "calendrier": "open -a Calendar",
                "calendar": "open -a Calendar",
                "messages": "open -a Messages",
                "photos": "open -a Photos",
                "musique": "open -a Music",
                "music": "open -a Music",
                "maps": "open -a Maps",
                "cartes": "open -a Maps",
                "terminal": "open -a Terminal",
                "finder": "open -a Finder",
                "explorateur": "open -a Finder",
                "explorer": "open -a Finder",
                "préférences": "open -a 'System Preferences'",
                "preferences": "open -a 'System Preferences'",
                "paramètres": "open -a 'System Preferences'",
                "settings": "open -a 'System Preferences'"
            }
        else:  # Linux
            return {
                "chrome": "google-chrome",
                "firefox": "firefox",
                "calculatrice": "gnome-calculator",
                "calculator": "gnome-calculator",
                "explorateur": "nautilus",
                "explorer": "nautilus",
                "nautilus": "nautilus",
                "thunderbird": "thunderbird",
                "mail": "thunderbird",
                "email": "thunderbird",
                "terminal": "gnome-terminal",
                "gedit": "gedit",
                "text-editor": "gedit",
                "editeur": "gedit",
                "éditeur": "gedit",
                "paramètres": "gnome-control-center",
                "settings": "gnome-control-center",
                "vlc": "vlc",
                "libreoffice": "libreoffice",
                "writer": "libreoffice --writer",
                "calc": "libreoffice --calc",
                "impress": "libreoffice --impress"
            }
    
    def _charger_configuration(self) -> None:
        """Charge la configuration personnalisée depuis le fichier."""
        try:
            if not CONFIG_DIR.exists():
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.custom_commands = config.get("custom_commands", {})
                    saved_lang = config.get("language")
                    if saved_lang and saved_lang in TRANSLATIONS:
                        self.lang = saved_lang
                        self.t = TRANSLATIONS[self.lang]
                logger.info(self.t["config_loaded"])
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            self.custom_commands = {}
    
    def _sauvegarder_configuration(self) -> None:
        """Sauvegarde la configuration personnalisée dans un fichier."""
        try:
            if not CONFIG_DIR.exists():
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                
            config = {
                "custom_commands": self.custom_commands,
                "language": self.lang
            }
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                
            logger.info(self.t["config_saved"])
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
    
    def ajouter_commande_personnalisee(self, nom: str, commande: str) -> bool:
        """
        Ajoute une commande personnalisée à la liste.
        
        Args:
            nom: Nom de la commande
            commande: Commande système à exécuter
            
        Returns:
            bool: Succès de l'opération
        """
        try:
            nom = nom.lower()
            self.custom_commands[nom] = commande
            self.commandes = {**self.default_commands, **self.custom_commands}
            self._sauvegarder_configuration()
            logger.info(f"Commande personnalisée ajoutée: {nom} -> {commande}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la commande personnalisée: {str(e)}")
            return False
    
    def changer_langue(self, lang: str) -> bool:
        """
        Change la langue de l'interface.
        
        Args:
            lang: Code de langue (fr ou en)
            
        Returns:
            bool: Succès de l'opération
        """
        if lang in TRANSLATIONS:
            self.lang = lang
            self.t = TRANSLATIONS[lang]
            self._sauvegarder_configuration()
            return True
        return False
    
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
                # Extraire le nom réel de l'application
                for app in self.commandes.keys():
                    if app in app_name_lower:
                        app_name_lower = app
                        break
                break
        
        # Si le nom contient toujours un des patterns, on l'enlève
        for pattern in fermer_patterns:
            app_name_lower = re.sub(fr'{pattern}\s+', '', app_name_lower)
        
        # Vérifier si l'application est dans la liste des applications autorisées
        if app_name_lower not in self.commandes:
            # Recherche d'applications similaires pour suggestions
            apps_similaires = [app for app in self.commandes.keys() 
                          if app_name_lower in app or 
                          (len(app_name_lower) > 3 and self._similarity_score(app, app_name_lower) > 0.7)]
            
            if apps_similaires:
                suggestions = ", ".join([f"'{app}'" for app in apps_similaires[:3]])
                return False, f"{self.t['app_not_found'].format(app_name_lower)} {self.t['app_suggestion'].format(suggestions)}", ""
            return False, self.t["app_not_authorized"].format(app_name_lower), ""
        
        # Obtenir la commande pour l'application
        if fermer:
            if self.os_type == 'win32':
                commande = f"taskkill /F /IM {self.commandes[app_name_lower].split()[-1]}.exe"
            elif self.os_type == 'darwin':
                # Extraire le nom de l'application du format 'open -a AppName'
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
                # Utiliser start_new_session pour que le processus continue après la fermeture du script
                subprocess.Popen(
                    commande, 
                    shell=True, 
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            logger.info(f"Commande exécutée avec succès: {commande}")
            
            if fermer:
                return True, self.t["success_close"].format(app_name_lower), commande
            else:
                return True, self.t["success_open"].format(app_name_lower), commande
            
        except Exception as e:
            erreur = self.t["error_executing"].format(app_name_lower, str(e))
            logger.error(erreur)
            return False, erreur, commande
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """
        Calcule un score de similarité entre deux chaînes.
        Utile pour les suggestions d'applications.
        
        Args:
            str1: Première chaîne
            str2: Deuxième chaîne
            
        Returns:
            float: Score de similarité entre 0 et 1
        """
        # Utilisation d'une version simplifiée de la distance de Levenshtein
        str1, str2 = str1.lower(), str2.lower()
        if len(str1) == 0 or len(str2) == 0:
            return 0.0
            
        # Calculer les n-grammes de caractères (bi-grammes)
        def get_ngrams(s, n=2):
            return [s[i:i+n] for i in range(len(s)-n+1)]
            
        ngrams1 = set(get_ngrams(str1))
        ngrams2 = set(get_ngrams(str2))
        
        # Coefficient de Dice
        intersection = len(ngrams1.intersection(ngrams2))
        if not intersection:
            return 0.0
        return 2 * intersection / (len(ngrams1) + len(ngrams2))
    
    def rechercher_applications(self, terme: str) -> List[str]:
        """
        Recherche des applications contenant le terme spécifié.
        
        Args:
            terme: Terme de recherche
            
        Returns:
            List[str]: Liste des applications correspondantes
        """
        terme = terme.lower()
        resultats = [app for app in self.commandes.keys() if terme in app]
        
        # Ajouter les résultats par similarité si peu de résultats exacts
        if len(resultats) < 5:
            similaires = [app for app in self.commandes.keys() 
                       if app not in resultats and len(terme) > 2 and 
                       self._similarity_score(app, terme) > 0.6]
            resultats.extend(similaires[:5])  # Limiter à 5 résultats similaires supplémentaires
            
        return sorted(resultats)
    
    def obtenir_applications_disponibles(self) -> List[str]:
        """Obtient la liste des applications disponibles."""
        return sorted(self.commandes.keys())
    
    def obtenir_commande_pour_app(self, app_name: str) -> Optional[str]:
        """
        Obtient la commande associée à une application.
        
        Args:
            app_name: Nom de l'application
            
        Returns:
            Optional[str]: Commande associée ou None
        """
        return self.commandes.get(app_name.lower())


class InterfaceUtilisateur:
    """Interface utilisateur pour le service de commandes."""
    
    def __init__(self, service: CommandeService):
        """
        Initialisation de l'interface utilisateur.
        
        Args:
            service: Instance du service de commandes
        """
        self.service = service
        self.commandes_interpretees = {
            "fr": {
                "help": ["aide", "help", "?", "h"],
                "list": ["liste", "commandes", "applications", "apps"],
                "exit": ["quitter", "exit", "q", "bye", "au revoir", "fin"],
                "search": ["rechercher", "chercher", "search", "find", "recherche"],
                "add": ["ajouter", "add", "nouvelle", "new", "créer", "create"],
                "language": ["langue", "language", "idioma"]
            },
            "en": {
                "help": ["help", "?", "h", "aide"],
                "list": ["list", "commands", "applications", "apps"],
                "exit": ["exit", "quit", "q", "bye", "goodbye"],
                "search": ["search", "find", "lookup", "rechercher"],
                "add": ["add", "new", "create", "ajouter"],
                "language": ["language", "lang", "langue"]
            }
        }
    
    def colorize(self, text: str, color: str) -> str:
        """
        Ajoute des codes couleur ANSI au texte.
        
        Args:
            text: Texte à colorier
            color: Nom de la couleur
            
        Returns:
            str: Texte coloré
        """
        if color.upper() not in COLORS:
            return text
        return f"{COLORS[color.upper()]}{text}{COLORS['RESET']}"
    
    def afficher_aide(self) -> None:
        """Affiche l'aide du programme."""
        t = self.service.t
        print(f"\n{self.colorize(t['help_title'], 'BOLD')}")
        print(f"- {self.colorize(t['help_command'], 'CYAN')}")
        print(f"- {self.colorize(t['help_open'], 'CYAN')}")
        print(f"- {self.colorize(t['help_close'], 'CYAN')}")
        print(f"- {self.colorize(t['help_search'], 'CYAN')}")
        print(f"- {self.colorize(t['help_add'], 'CYAN')}")
        print(f"- {self.colorize(t['help_lang'], 'CYAN')}")
        print(f"- {self.colorize(t['help_exit'], 'CYAN')}")
        print(f"- {self.colorize(t['help_help'], 'CYAN')}")
    
    def afficher_applications(self) -> None:
        """Affiche la liste des applications disponibles."""
        t = self.service.t
        apps = self.service.obtenir_applications_disponibles()
        
        print(f"\n{self.colorize(t['available_apps'], 'BOLD')}")
        
        # Afficher les applications en colonnes
        col_width = max(len(app) for app in apps) + 2  # Largeur de colonne
        nb_colonnes = min(3, max(1, 80 // col_width))  # Nombre de colonnes
        
        for i in range(0, len(apps), nb_colonnes):
            ligne = []
            for j in range(nb_colonnes):
                if i+j < len(apps):
                    app = apps[i+j]
                    if app in self.service.custom_commands:
                        ligne.append(f"- {self.colorize(app.ljust(col_width), 'PURPLE')}")
                    else:
                        ligne.append(f"- {app.ljust(col_width)}")
            print("".join(ligne))
    
    def afficher_accueil(self) -> None:
        """Affiche le message d'accueil."""
        t = self.service.t
        print(f"\n{self.colorize(t['welcome'], 'BOLD')}\n")
        
        # Afficher quelques exemples d'applications disponibles
        apps = self.service.obtenir_applications_disponibles()
        exemples = apps[:3] if len(apps) >= 3 else apps
        
        print(self.colorize(t["command_examples"], 'YELLOW'))
        for exemple in exemples:
            print(self.colorize(t["open_example"].format(exemple), 'GREEN'))
            print(self.colorize(t["close_example"].format(exemple), 'RED'))
        
        print(self.colorize(t["search_example"], 'BLUE'))
        print(self.colorize(t["exit_example"], 'CYAN'))
        print("")  # Ligne vide pour plus de lisibilité
    
    def traiter_commande(self, user_input: str) -> bool:
        """
        Traite une commande utilisateur.
        
        Args:
            user_input: Commande entrée par l'utilisateur
            
        Returns:
            bool: True si on continue, False si on quitte
        """
        if not user_input.strip():
            return True
            
        cmd_type = None
        interp = self.commandes_interpretees[self.service.lang]
        t = self.service.t
        user_input_lower = user_input.lower().strip()
        
        # Déterminer le type de commande
        for cmd, aliases in interp.items():
            if any(user_input_lower.startswith(alias) for alias in aliases):
                cmd_type = cmd
                break
        
        # Traiter selon le type de commande
        if cmd_type == "exit":
            print(f"\n{self.colorize(t['goodbye'], 'YELLOW')}")
            return False
            
        elif cmd_type == "help":
            self.afficher_aide()
            
        elif cmd_type == "list":
            self.afficher_applications()
            
        elif cmd_type == "search":
            # Extraire le terme de recherche
            for alias in interp["search"]:
                if user_input_lower.startswith(alias):
                    terme = user_input[len(alias):].strip()
                    break
            else:
                terme = ""
                
            if terme:
                resultats = self.service.rechercher_applications(terme)
                print(f"\n{self.colorize(t['searching_for'].format(terme), 'BOLD')}")
                
                if resultats:
                    for app in resultats:
                        cmd = self.service.obtenir_commande_pour_app(app)
                        if app in self.service.custom_commands:
                            print(f"- {self.colorize(app, 'PURPLE')} ({cmd})")
                        else:
                            print(f"- {app} ({cmd})")
                else:
                    print(self.colorize(t['no_results'].format(terme), 'RED'))
            else:
                # Si pas de terme, afficher toutes les applications
                self.afficher_applications()
                
        elif cmd_type == "add":
            # Format: ajouter nom_commande commande_système
            parts = user_input.split(" ", 2)
            if len(parts) >= 3:
                nom = parts[1].strip()
                commande = parts[2].strip()
                if self.service.ajouter_commande_personnalisee(nom, commande):
                    print(self.colorize(t['custom_command_added'].format(nom), 'GREEN'))
            else:
                self.afficher_aide()
                
        elif cmd_type == "language":
            # Format: langue fr/en
            parts = user_input.split(" ", 1)
            if len(parts) >= 2:
                langue = parts[1].strip().lower()
                if self.service.changer_langue(langue):
                    # Mettre à jour les traductions
                    t = self.service.t
                    print(self.colorize(t['language_changed'].format(langue.upper()), 'GREEN'))
                    self.afficher_accueil()
            else:
                self.afficher_aide()
                
        else:
            # Considérer comme une commande d'application
            # Vérifier si c'est une commande pour ouvrir une application
            app_name = None
            
            # Patterns d'ouverture d'application
            open_patterns = ["ouvrir ", "ouvre ", "lance ", "démarre ", "exécute ", 
                          "open ", "start ", "run ", "launch "]
            
            # Vérifier les patterns d'ouverture
            for pattern in open_patterns:
                if pattern in user_input_lower:
                    app_name = user_input.split(pattern, 1)[1].strip()
                    break
            
            # Si aucun motif d'ouverture n'est trouvé, considérer le texte entier comme nom d'application
            if not app_name:
                app_name = user_input
            
            # Exécuter la commande
            success, message, commande = self.service.executer_commande(app_name)
            
            # Afficher le résultat
            if success:
                print(f"\n{self.colorize(message, 'GREEN')}")
                print(self.colorize(t['command_executed'].format(commande), 'BLUE'))
            else:
                print(f"\n{self.colorize(message, 'RED')}")
        
        return True


async def main():
    """Fonction principale du service."""
    # Initialiser le service avec la langue par défaut
    service = CommandeService()
    ui = InterfaceUtilisateur(service)
    
    # Afficher l'écran d'accueil
    ui.afficher_accueil()
    
    # Boucle principale
    continuer = True
    while continuer:
        try:
            user_input = input(f"{ui.colorize(service.t['command_prompt'], 'CYAN')}").strip()
            continuer = ui.traiter_commande(user_input)
        except KeyboardInterrupt:
            print(f"\n{ui.colorize(service.t['goodbye'], 'YELLOW')}")
            break
        except Exception as e:
            logger.error(f"Erreur: {str(e)}")
            print(f"{ui.colorize('Erreur: ' + str(e), 'RED')}")
    
    logger.info("Arrêt du service de commandes")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Erreur fatale: {str(e)}")
        print(f"Une erreur est survenue: {str(e)}")