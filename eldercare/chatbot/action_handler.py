"""
Gestionnaire d'actions pour ElderCare Assistant.
Ce module exécute des actions basées sur les intentions détectées.
"""

import os
import sys
import subprocess
import datetime
import webbrowser
import logging
from typing import Dict, List, Tuple, Optional
from commande_service_mini import CommandeService

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ActionHandler:
    """Exécute des actions basées sur les intentions détectées."""
    
    def __init__(self):
        """Initialise le gestionnaire d'actions."""
        self.command_service = CommandeService()
        self.notes = []
        self.reminders = []
        self.os_type = sys.platform
        
        # Dossiers communs
        if self.os_type == 'win32':  # Windows
            self.common_folders = {
                "documents": os.path.join(os.path.expanduser("~"), "Documents"),
                "téléchargements": os.path.join(os.path.expanduser("~"), "Downloads"),
                "images": os.path.join(os.path.expanduser("~"), "Pictures"),
                "bureau": os.path.join(os.path.expanduser("~"), "Desktop"),
                "musique": os.path.join(os.path.expanduser("~"), "Music"),
                "vidéos": os.path.join(os.path.expanduser("~"), "Videos")
            }
        elif self.os_type == 'darwin':  # macOS
            self.common_folders = {
                "documents": os.path.join(os.path.expanduser("~"), "Documents"),
                "téléchargements": os.path.join(os.path.expanduser("~"), "Downloads"),
                "images": os.path.join(os.path.expanduser("~"), "Pictures"),
                "bureau": os.path.join(os.path.expanduser("~"), "Desktop"),
                "musique": os.path.join(os.path.expanduser("~"), "Music"),
                "vidéos": os.path.join(os.path.expanduser("~"), "Movies")
            }
        else:  # Linux
            self.common_folders = {
                "documents": os.path.join(os.path.expanduser("~"), "Documents"),
                "téléchargements": os.path.join(os.path.expanduser("~"), "Downloads"),
                "images": os.path.join(os.path.expanduser("~"), "Pictures"),
                "bureau": os.path.join(os.path.expanduser("~"), "Desktop"),
                "musique": os.path.join(os.path.expanduser("~"), "Music"),
                "vidéos": os.path.join(os.path.expanduser("~"), "Videos")
            }
    
    def handle_action(self, intent: str, params: List[str]) -> Tuple[bool, str, Optional[str]]:
        """
        Exécute une action basée sur l'intention détectée.
        
        Args:
            intent: L'intention détectée
            params: Les paramètres extraits du message utilisateur
            
        Returns:
            Tuple (succès, message_réponse, commande_exécutée)
        """
        # Dispatching basé sur l'intention
        if intent == "ouvrir_application" and params:
            return self._handle_open_app(params[0])
        elif intent == "ouvrir_dossier" and params:
            return self._handle_open_folder(params[0])
        elif intent == "recherche_web" and params:
            return self._handle_web_search(params[0])
        elif intent == "donner_date_heure":
            return self._handle_date_time()
        elif intent == "lire_musique" and params:
            return self._handle_play_music(params[0] if params else "")
        elif intent == "prendre_note" and params:
            return self._handle_take_note(params[0])
        elif intent == "creer_rappel" and params:
            task = params[0]
            time = params[1] if len(params) > 1 else ""
            return self._handle_create_reminder(task, time)
        elif intent == "controle_systeme":
            return self._handle_system_control(params[0] if params else "")
        elif intent == "controle_volume":
            return self._handle_volume_control(params[0] if params else "")
        elif intent == "envoyer_email" and params:
            return self._handle_send_email(params[0])
        elif intent == "salutation":
            return True, "Bonjour ! Comment puis-je vous aider aujourd'hui ?", None
        elif intent == "au_revoir":
            return True, "Au revoir ! Passez une bonne journée.", None
        else:
            return False, "Je ne sais pas comment traiter cette demande.", None
    
    def _handle_open_app(self, app_name: str) -> Tuple[bool, str, str]:
        """Gère l'ouverture d'une application."""
        success, message, command = self.command_service.executer_commande(app_name)
        
        if success:
            # Créer l'équivalent SSH pour affichage avec le format désiré
            if self.os_type == 'win32':
                app_command = app_name.lower()
                if app_command == "chrome" or app_command == "google chrome":
                    ssh_command = "ssh USER@HOST \"google-chrome &\""
                elif app_command == "vs code" or app_command == "vscode" or app_command == "visual studio code":
                    ssh_command = "ssh USER@HOST \"code &\""
                elif app_command == "calculatrice" or app_command == "calculator":
                    ssh_command = "ssh USER@HOST \"gnome-calculator &\""
                elif app_command == "spotify" or app_command == "musique" or app_command == "music":
                    ssh_command = "ssh USER@HOST \"spotify &\""
                elif app_command == "terminal" or app_command == "cmd" or app_command == "powershell":
                    ssh_command = "ssh USER@HOST \"gnome-terminal &\""
                elif app_command == "explorateur" or app_command == "file explorer" or app_command == "gestionnaire de fichiers":
                    ssh_command = "ssh USER@HOST \"nautilus &\""
                elif app_command == "gimp":
                    ssh_command = "ssh USER@HOST \"gimp &\""
                else:
                    ssh_command = f"ssh USER@HOST \"{app_name.lower()} &\""
            else:
                # Pour Linux/Mac, utiliser directement le nom de l'app
                ssh_command = f"ssh USER@HOST \"{app_name.lower()} &\""
            
            # Utiliser un message plus direct comme dans les exemples
            if app_name.lower() == "chrome" or app_name.lower() == "google chrome":
                simple_message = "Bien sûr, j'ouvre Google Chrome maintenant."
            elif app_name.lower() in ["vs code", "vscode", "visual studio code"]:
                simple_message = "D'accord, Visual Studio Code est en cours de lancement."
            elif app_name.lower() in ["calculatrice", "calculator"]:
                simple_message = "Voici votre calculatrice."
            elif app_name.lower() in ["spotify", "musique", "music"]:
                simple_message = "J'ouvre Spotify pour vous."
            elif app_name.lower() in ["terminal", "cmd", "powershell"]:
                simple_message = "Terminal lancé."
            elif app_name.lower() in ["explorateur", "file explorer", "gestionnaire de fichiers"]:
                simple_message = "Bien sûr, gestionnaire de fichiers en cours d'ouverture."
            elif app_name.lower() == "gimp":
                simple_message = "GIMP est prêt pour l'édition d'images."
            else:
                simple_message = f"J'ai ouvert {app_name} pour vous."
            
            # Formater le message avec la commande SSH
            formatted_message = f"{simple_message}\n💻 (Commande SSH exécutée) : {ssh_command.replace('ssh USER@HOST \"', '').replace('\"', '')}"
            
            # Renvoyer le message formaté
            return success, formatted_message, command
        else:
            return success, message, command
    
    def _handle_open_folder(self, folder_name: str) -> Tuple[bool, str, Optional[str]]:
        """Gère l'ouverture d'un dossier."""
        folder_name_lower = folder_name.lower()
        
        # Rechercher dans les dossiers communs
        target_folder = None
        for key, path in self.common_folders.items():
            if key in folder_name_lower or folder_name_lower in key:
                target_folder = path
                break
        
        # Si le dossier n'est pas trouvé dans les dossiers communs
        if not target_folder:
            if os.path.exists(folder_name):
                target_folder = folder_name
            elif os.path.exists(os.path.join(os.path.expanduser("~"), folder_name)):
                target_folder = os.path.join(os.path.expanduser("~"), folder_name)
            else:
                return False, f"Je n'ai pas trouvé le dossier '{folder_name}'.", None
        
        # Ouvrir le dossier
        try:
            if self.os_type == 'win32':
                command = f"explorer \"{target_folder}\""
                subprocess.Popen(command, shell=True)
            elif self.os_type == 'darwin':
                command = f"open \"{target_folder}\""
                subprocess.Popen(command, shell=True)
            else:
                command = f"xdg-open \"{target_folder}\""
                subprocess.Popen(command.split())
            
            return True, f"J'ai ouvert le dossier {folder_name} pour vous.", command
        except Exception as e:
            logger.error(f"Erreur lors de l'ouverture du dossier: {str(e)}")
            return False, f"Erreur lors de l'ouverture du dossier: {str(e)}", None
    
    def _handle_web_search(self, query: str) -> Tuple[bool, str, Optional[str]]:
        """Gère une recherche web."""
        try:
            # Nettoyer la requête
            query = query.strip('"\'')
            
            # Construire l'URL de recherche Google
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            # Ouvrir le navigateur avec l'URL
            webbrowser.open(search_url)
            
            return True, f"J'ai lancé une recherche sur '{query}'.", f"webbrowser.open({search_url})"
        except Exception as e:
            logger.error(f"Erreur lors de la recherche web: {str(e)}")
            return False, f"Erreur lors de la recherche web: {str(e)}", None
    
    def _handle_date_time(self) -> Tuple[bool, str, Optional[str]]:
        """Gère une demande d'heure ou de date."""
        now = datetime.datetime.now()
        date_str = now.strftime("%A %d %B %Y")
        time_str = now.strftime("%H:%M:%S")
        
        message = f"Nous sommes le {date_str} et il est {time_str}."
        return True, message, None
    
    def _handle_play_music(self, music_type: str = "") -> Tuple[bool, str, Optional[str]]:
        """Gère la lecture de musique."""
        # Pour simplifier, nous allons juste ouvrir l'application de musique par défaut
        if self.os_type == 'win32':
            try:
                command = "start mswindowsmusic:"
                subprocess.Popen(command, shell=True)
                return True, "J'ai ouvert l'application Musique pour vous.", command
            except Exception as e:
                # Essayer avec un autre lecteur comme Windows Media Player
                try:
                    command = "start wmplayer.exe"
                    subprocess.Popen(command, shell=True)
                    return True, "J'ai ouvert Windows Media Player pour vous.", command
                except Exception as e2:
                    logger.error(f"Erreur lors de l'ouverture du lecteur de musique: {str(e2)}")
                    return False, "Je n'ai pas pu ouvrir de lecteur de musique.", None
        elif self.os_type == 'darwin':
            try:
                command = "open -a Music"
                subprocess.Popen(command, shell=True)
                return True, "J'ai ouvert l'application Musique pour vous.", command
            except Exception as e:
                logger.error(f"Erreur lors de l'ouverture du lecteur de musique: {str(e)}")
                return False, "Je n'ai pas pu ouvrir de lecteur de musique.", None
        else:
            try:
                command = "rhythmbox"
                subprocess.Popen(command.split())
                return True, "J'ai ouvert Rhythmbox pour vous.", command
            except Exception as e:
                logger.error(f"Erreur lors de l'ouverture du lecteur de musique: {str(e)}")
                return False, "Je n'ai pas pu ouvrir de lecteur de musique.", None
    
    def _handle_take_note(self, note_content: str) -> Tuple[bool, str, Optional[str]]:
        """Gère la prise de notes."""
        try:
            # Ajouter la note avec un horodatage
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.notes.append({
                "timestamp": timestamp,
                "content": note_content
            })
            
            # Enregistrer la note dans un fichier
            notes_dir = os.path.join(os.path.expanduser("~"), "ElderCare_Notes")
            os.makedirs(notes_dir, exist_ok=True)
            
            notes_file = os.path.join(notes_dir, "eldercare_notes.txt")
            with open(notes_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {note_content}\n")
            
            return True, f"J'ai pris note : {note_content}", None
        except Exception as e:
            logger.error(f"Erreur lors de la prise de note: {str(e)}")
            return False, f"Erreur lors de la prise de note: {str(e)}", None
    
    def _handle_create_reminder(self, task: str, time: str = "") -> Tuple[bool, str, Optional[str]]:
        """Gère la création d'un rappel."""
        try:
            # Ajouter le rappel
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            reminder = {
                "created_at": timestamp,
                "task": task,
                "scheduled_for": time,
                "completed": False
            }
            self.reminders.append(reminder)
            
            # Enregistrer le rappel dans un fichier
            reminders_dir = os.path.join(os.path.expanduser("~"), "ElderCare_Reminders")
            os.makedirs(reminders_dir, exist_ok=True)
            
            reminders_file = os.path.join(reminders_dir, "eldercare_reminders.txt")
            with open(reminders_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] Tâche: {task}, Prévu pour: {time}\n")
            
            message = f"J'ai créé un rappel pour : {task}"
            if time:
                message += f" à {time}"
                
            # Ouvrir l'application Calendrier ou Rappels si disponible
            if self.os_type == 'win32':
                try:
                    # Tenter d'ouvrir l'application Calendrier Microsoft
                    subprocess.Popen("start outlookcal:", shell=True)
                except Exception:
                    pass
            
            return True, message, None
        except Exception as e:
            logger.error(f"Erreur lors de la création du rappel: {str(e)}")
            return False, f"Erreur lors de la création du rappel: {str(e)}", None
    
    def _handle_system_control(self, action: str = "") -> Tuple[bool, str, Optional[str]]:
        """Gère le contrôle système (éteindre, redémarrer, etc.)."""
        try:
            action_lower = action.lower() if action else ""
            command = ""
            message = ""
            
            if "éteins" in action_lower or "arrêter" in action_lower:
                if self.os_type == 'win32':
                    command = "shutdown /s /t 60"
                    message = "J'éteindrai l'ordinateur dans 60 secondes. Pour annuler, tapez 'shutdown /a'."
                else:
                    command = "shutdown -h +1"
                    message = "J'éteindrai l'ordinateur dans 1 minute. Pour annuler, utilisez 'shutdown -c'."
            elif "redémarre" in action_lower:
                if self.os_type == 'win32':
                    command = "shutdown /r /t 60"
                    message = "Je redémarrerai l'ordinateur dans 60 secondes. Pour annuler, tapez 'shutdown /a'."
                else:
                    command = "shutdown -r +1"
                    message = "Je redémarrerai l'ordinateur dans 1 minute. Pour annuler, utilisez 'shutdown -c'."
            elif "veille" in action_lower:
                if self.os_type == 'win32':
                    command = "rundll32.exe powrprof.dll,SetSuspendState 0,1,0"
                    message = "Je mets l'ordinateur en veille."
                elif self.os_type == 'darwin':
                    command = "pmset sleepnow"
                    message = "Je mets l'ordinateur en veille."
                else:
                    command = "systemctl suspend"
                    message = "Je mets l'ordinateur en veille."
            
            if command:
                subprocess.Popen(command, shell=True)
                return True, message, command
            else:
                return False, "Je ne comprends pas quelle action système effectuer.", None
        except Exception as e:
            logger.error(f"Erreur lors du contrôle système: {str(e)}")
            return False, f"Erreur lors du contrôle système: {str(e)}", None
    
    def _handle_volume_control(self, action: str = "") -> Tuple[bool, str, Optional[str]]:
        """Gère le contrôle du volume."""
        try:
            action_lower = action.lower() if action else ""
            command = ""
            message = ""
            
            if self.os_type == 'win32':
                if "monte" in action_lower:
                    # Augmenter le volume sous Windows (utilise PowerShell)
                    powershell_cmd = "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"
                    command = f"powershell -Command \"{powershell_cmd}\""
                    message = "J'ai augmenté le volume."
                elif "baisse" in action_lower:
                    # Diminuer le volume sous Windows (utilise PowerShell)
                    powershell_cmd = "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"
                    command = f"powershell -Command \"{powershell_cmd}\""
                    message = "J'ai baissé le volume."
                elif "coupe" in action_lower or "mute" in action_lower:
                    # Couper le son sous Windows (utilise PowerShell)
                    powershell_cmd = "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"
                    command = f"powershell -Command \"{powershell_cmd}\""
                    message = "J'ai coupé le son."
                elif "active" in action_lower or "unmute" in action_lower:
                    # Réactiver le son sous Windows (utilise PowerShell)
                    powershell_cmd = "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"
                    command = f"powershell -Command \"{powershell_cmd}\""
                    message = "J'ai réactivé le son."
            else:
                # Pour les systèmes Unix (macOS/Linux)
                if "monte" in action_lower:
                    if self.os_type == 'darwin':
                        command = "osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'"
                    else:
                        command = "amixer -D pulse sset Master 10%+"
                    message = "J'ai augmenté le volume."
                elif "baisse" in action_lower:
                    if self.os_type == 'darwin':
                        command = "osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'"
                    else:
                        command = "amixer -D pulse sset Master 10%-"
                    message = "J'ai baissé le volume."
                elif "coupe" in action_lower or "mute" in action_lower:
                    if self.os_type == 'darwin':
                        command = "osascript -e 'set volume output muted true'"
                    else:
                        command = "amixer -D pulse set Master mute"
                    message = "J'ai coupé le son."
                elif "active" in action_lower or "unmute" in action_lower:
                    if self.os_type == 'darwin':
                        command = "osascript -e 'set volume output muted false'"
                    else:
                        command = "amixer -D pulse set Master unmute"
                    message = "J'ai réactivé le son."
            
            if command:
                subprocess.Popen(command, shell=True)
                return True, message, command
            else:
                return False, "Je ne comprends pas quelle action effectuer sur le volume.", None
        except Exception as e:
            logger.error(f"Erreur lors du contrôle du volume: {str(e)}")
            return False, f"Erreur lors du contrôle du volume: {str(e)}", None
    
    def _handle_send_email(self, recipient: str) -> Tuple[bool, str, Optional[str]]:
        """Gère l'envoi d'emails."""
        try:
            # Ouvrir le client email par défaut
            if self.os_type == 'win32':
                command = "start outlookmail:"
                subprocess.Popen(command, shell=True)
                message = "J'ai ouvert votre client email."
            elif self.os_type == 'darwin':
                command = "open -a Mail"
                subprocess.Popen(command, shell=True)
                message = "J'ai ouvert l'application Mail."
            else:
                # Tenter d'ouvrir un client email commun sous Linux
                for client in ["thunderbird", "evolution"]:
                    try:
                        subprocess.Popen(client, shell=True)
                        command = client
                        message = f"J'ai ouvert {client}."
                        break
                    except:
                        continue
                else:
                    # Si aucun client n'a pu être ouvert, ouvrir Gmail dans le navigateur
                    webbrowser.open("https://mail.google.com")
                    command = "webbrowser.open('https://mail.google.com')"
                    message = "J'ai ouvert Gmail dans votre navigateur."
            
            return True, message, command
        except Exception as e:
            logger.error(f"Erreur lors de l'ouverture du client email: {str(e)}")
            return False, f"Erreur lors de l'ouverture du client email: {str(e)}", None
    
    def get_notes(self) -> List[Dict]:
        """Récupère la liste des notes."""
        return self.notes
    
    def get_reminders(self) -> List[Dict]:
        """Récupère la liste des rappels."""
        return self.reminders


if __name__ == "__main__":
    # Test du gestionnaire d'actions
    handler = ActionHandler()
    
    # Test de quelques actions
    print("Test : Heure et date")
    print(handler._handle_date_time())
    
    print("\nTest : Ouvrir une application")
    print(handler._handle_open_app("notepad"))
    
    print("\nTest : Prendre une note")
    print(handler._handle_take_note("Acheter du pain")) 