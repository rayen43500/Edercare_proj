import os
import sys
import subprocess
import logging
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemCommandExecutor:
    """
    Module for executing system commands based on user requests.
    Specifically designed for eldercare assistants to help seniors with
    computer tasks like opening applications.
    """
    
    def __init__(self):
        """Initialize the system command executor."""
        self.os_type = sys.platform
        self.allowed_applications = self._get_allowed_applications()
        logger.info(f"SystemCommandExecutor initialized for platform: {self.os_type}")
    
    def _get_allowed_applications(self) -> Dict[str, str]:
        """
        Get a dictionary of allowed applications and their commands.
        Different for each operating system.
        
        Returns:
            Dict mapping application names to their command strings
        """
        # Basic set of allowed applications
        if self.os_type == 'win32':  # Windows
            return {
                "chrome": "start chrome",
                "firefox": "start firefox",
                "edge": "start msedge",
                "notepad": "start notepad",
                "calculator": "start calc",
                "file explorer": "start explorer",
                "weather": "start ms-weather:",
                "maps": "start bingmaps:",
                "mail": "start outlookmail:",
                "photos": "start ms-photos:",
                "settings": "start ms-settings:",
                "word": "start winword",
                "excel": "start excel",
                "paint": "start mspaint",
                "music": "start mswindowsmusic:",
                "calendar": "start outlookcal:"
            }
        elif self.os_type == 'darwin':  # macOS
            return {
                "chrome": "open -a 'Google Chrome'",
                "firefox": "open -a Firefox",
                "safari": "open -a Safari",
                "mail": "open -a Mail",
                "notes": "open -a Notes",
                "calculator": "open -a Calculator",
                "finder": "open .",
                "photos": "open -a Photos",
                "music": "open -a Music",
                "calendar": "open -a Calendar",
                "messages": "open -a Messages",
                "facetime": "open -a FaceTime",
                "preview": "open -a Preview",
                "maps": "open -a Maps",
                "word": "open -a 'Microsoft Word'",
                "excel": "open -a 'Microsoft Excel'"
            }
        else:  # Linux and others
            return {
                "chrome": "google-chrome",
                "firefox": "firefox",
                "calculator": "gnome-calculator",
                "file explorer": "nautilus",
                "terminal": "gnome-terminal",
                "text editor": "gedit",
                "photos": "eog",
                "office": "libreoffice",
                "music": "rhythmbox",
                "email": "thunderbird"
            }
    
    def execute_command(self, app_name: str) -> Tuple[bool, str]:
        """
        Execute a command to open the specified application.
        
        Args:
            app_name: Name of the application to open
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Normalize app name to lowercase
        app_name_lower = app_name.lower()
        
        # Check if the application is in the allowed list
        if app_name_lower not in self.allowed_applications:
            similar_apps = [app for app in self.allowed_applications.keys() 
                           if app_name_lower in app or app in app_name_lower]
            
            if similar_apps:
                suggestion = f"Did you mean {' or '.join(similar_apps)}?"
                return False, f"Application '{app_name}' not found. {suggestion}"
            return False, f"Application '{app_name}' not found or not allowed."
        
        # Get the command for the application
        command = self.allowed_applications[app_name_lower]
        
        try:
            # Afficher la commande qui va être exécutée
            print(f"\nCommande CMD exécutée: {command}")
            
            # Execute the command
            if self.os_type == 'win32':
                subprocess.Popen(command, shell=True)
            else:
                subprocess.Popen(command.split(), start_new_session=True)
            
            logger.info(f"Successfully executed command: {command}")
            return True, f"Commande '{command}' exécutée avec succès pour ouvrir {app_name}."
            
        except Exception as e:
            error_msg = f"Error executing command for '{app_name}': {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_available_applications(self) -> List[str]:
        """
        Get a list of available applications that can be opened.
        
        Returns:
            List of application names
        """
        return list(self.allowed_applications.keys())


# Test function
def main():
    """Test the SystemCommandExecutor"""
    executor = SystemCommandExecutor()
    
    print("Available applications:")
    for app in executor.get_available_applications():
        print(f"- {app}")
    
    while True:
        app_name = input("\nEnter application name to open (or 'quit' to exit): ")
        if app_name.lower() == 'quit':
            break
            
        success, message = executor.execute_command(app_name)
        print(message)


if __name__ == "__main__":
    main() 