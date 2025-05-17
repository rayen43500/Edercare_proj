"""
Détecteur d'intentions pour ElderCare Assistant.
Ce module analyse les messages utilisateur et identifie l'intention.
"""

import re
from typing import Dict, List, Tuple, Optional


class IntentDetector:
    """Détecte l'intention de l'utilisateur basée sur des patterns prédéfinis."""
    
    def __init__(self):
        """Initialise le détecteur d'intentions avec les patterns pour chaque intention."""
        self.intents = {
            "ouvrir_application": {
                "patterns": [
                    r"ouvre\s+(.+)",
                    r"ouvrir\s+(.+)",
                    r"lance\s+(.+)",
                    r"lancer\s+(.+)",
                    r"démarre\s+(.+)",
                    r"démarrer\s+(.+)",
                    r"exécute\s+(.+)",
                    r"exécuter\s+(.+)",
                    r"peux-tu\s+ouvrir\s+(.+)",
                    r"je\s+veux\s+utiliser\s+(.+)",
                    r"start\s+(.+)",
                    r"run\s+(.+)",
                    r"open\s+(.+)"
                ],
                "examples": [
                    "Ouvre Chrome", 
                    "Ouvrir Chrome",
                    "Lance Word", 
                    "Lancer Word",
                    "Démarre Visual Studio", 
                    "Peux-tu ouvrir Spotify ?", 
                    "Je veux utiliser Discord", 
                    "Exécute VLC"
                ]
            },
            "ouvrir_dossier": {
                "patterns": [
                    r"ouvre\s+le\s+dossier\s+(.+)",
                    r"ouvrir\s+le\s+dossier\s+(.+)",
                    r"va\s+dans\s+(.+)",
                    r"aller\s+dans\s+(.+)",
                    r"affiche\s+mes\s+(.+)",
                    r"afficher\s+mes\s+(.+)",
                    r"accède\s+à\s+(.+)",
                    r"accéder\s+à\s+(.+)",
                    r"montre\s+le\s+dossier\s+(.+)",
                    r"montrer\s+le\s+dossier\s+(.+)"
                ],
                "examples": [
                    "Ouvre le dossier Téléchargements", 
                    "Ouvrir le dossier Documents",
                    "Va dans Documents", 
                    "Affiche mes images", 
                    "Accède à mon bureau", 
                    "Montre le dossier \"Projets\""
                ]
            },
            "recherche_web": {
                "patterns": [
                    r"recherche\s+(.+)",
                    r"trouve[- ]moi\s+(.+)",
                    r"c'est\s+quoi\s+(.+)",
                    r"fais\s+une\s+recherche\s+sur\s+(.+)",
                    r"dis-moi\s+qui\s+est\s+(.+)",
                    r"cherche\s+(.+)"
                ],
                "examples": [
                    "Recherche \"climat Paris aujourd'hui\"", 
                    "Trouve-moi une recette de pizza", 
                    "C'est quoi l'intelligence artificielle ?", 
                    "Fais une recherche sur le prix du Bitcoin", 
                    "Dis-moi qui est Elon Musk"
                ]
            },
            "donner_date_heure": {
                "patterns": [
                    r"quelle\s+heure\s+est-il",
                    r"c'est\s+quoi\s+la\s+date",
                    r"donne[- ]moi\s+le\s+jour",
                    r"on\s+est\s+quel\s+jour",
                    r"quelle\s+est\s+l'heure",
                    r"date\s+aujourd'hui"
                ],
                "examples": [
                    "Quelle heure est-il ?", 
                    "C'est quoi la date aujourd'hui ?", 
                    "Donne-moi le jour actuel", 
                    "On est quel jour ?", 
                    "Quelle est l'heure maintenant ?"
                ]
            },
            "lire_musique": {
                "patterns": [
                    r"joue\s+de\s+la\s+musique",
                    r"mets\s+une\s+chanson",
                    r"lance\s+ma\s+playlist",
                    r"peux-tu\s+lire\s+de\s+la\s+musique",
                    r"fais[- ]moi\s+écouter"
                ],
                "examples": [
                    "Joue de la musique", 
                    "Mets une chanson", 
                    "Lance ma playlist préférée", 
                    "Peux-tu lire de la musique calme ?", 
                    "Fais-moi écouter du jazz"
                ]
            },
            "prendre_note": {
                "patterns": [
                    r"prends\s+une\s+note\s*:\s*(.+)",
                    r"note\s+que\s+(.+)",
                    r"mémorise\s+(.+)",
                    r"écris\s+(.+)\s+dans\s+mes\s+notes",
                    r"sauvegarde\s+(.+)"
                ],
                "examples": [
                    "Prends une note : acheter du pain", 
                    "Note que j'ai une réunion demain à 10h", 
                    "Mémorise cette phrase : le mot de passe est 1234", 
                    "Écris ça dans mes notes", 
                    "Sauvegarde cette idée"
                ]
            },
            "creer_rappel": {
                "patterns": [
                    r"rappelle[- ]moi\s+de\s+(.+)\s+à\s+(.+)",
                    r"fais[- ]moi\s+un\s+rappel\s+(.+)\s+à\s+(.+)",
                    r"souviens[- ]toi\s+de\s+(.+)",
                    r"rappelle[- ]moi\s+de\s+(.+)",
                    r"programme\s+un\s+rappel\s+pour\s+(.+)"
                ],
                "examples": [
                    "Rappelle-moi d'appeler maman à 18h", 
                    "Fais-moi un rappel demain à 9h", 
                    "Souviens-toi de mon rendez-vous", 
                    "Rappelle-moi de sortir la poubelle", 
                    "Programme un rappel pour lundi"
                ]
            },
            "controle_systeme": {
                "patterns": [
                    r"éteins\s+l'ordinateur",
                    r"redémarre\s+le\s+pc",
                    r"mets\s+l'ordinateur\s+en\s+veille",
                    r"ferme\s+tout",
                    r"arrête[r]?\s+la\s+machine"
                ],
                "examples": [
                    "Éteins l'ordinateur", 
                    "Redémarre le PC", 
                    "Mets l'ordinateur en veille", 
                    "Ferme tout", 
                    "Je veux arrêter la machine"
                ]
            },
            "controle_volume": {
                "patterns": [
                    r"monte\s+le\s+son",
                    r"baisse\s+le\s+volume",
                    r"coupe[r]?\s+le\s+son",
                    r"mets\s+le\s+volume\s+à\s+(.+)",
                    r"active\s+le\s+son"
                ],
                "examples": [
                    "Monte le son", 
                    "Baisse le volume", 
                    "Couper le son", 
                    "Mets le volume à 50%", 
                    "Active le son"
                ]
            },
            "envoyer_email": {
                "patterns": [
                    r"envoie\s+un\s+mail\s+à\s+(.+)",
                    r"écris\s+un\s+email\s+pour\s+(.+)",
                    r"rédige\s+un\s+message\s+à\s+(.+)",
                    r"peux-tu\s+envoyer\s+un\s+email",
                    r"prépare\s+un\s+email\s+de\s+(.+)"
                ],
                "examples": [
                    "Envoie un mail à Sarah", 
                    "Écris un email pour dire bonjour", 
                    "Rédige un message à mon patron", 
                    "Peux-tu envoyer un email maintenant ?", 
                    "Prépare un email de remerciement"
                ]
            },
            "salutation": {
                "patterns": [
                    r"^(bonjour|salut|hello|hi|hey|coucou|bonsoir)[\s!.]*$"
                ],
                "examples": [
                    "Bonjour", "Salut", "Hello", "Hi", "Hey", "Coucou", "Bonsoir"
                ]
            },
            "au_revoir": {
                "patterns": [
                    r"^(au revoir|bye|à bientôt|à plus tard|à plus|adieu|goodbye|bonne journée)[\s!.]*$"
                ],
                "examples": [
                    "Au revoir", "Bye", "À bientôt", "À plus tard", "À plus", "Adieu", "Goodbye", "Bonne journée"
                ]
            }
        }
    
    def detect_intent(self, message: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Détecte l'intention de l'utilisateur à partir du message.
        
        Args:
            message: Le message utilisateur à analyser
            
        Returns:
            Tuple (intention, paramètres extraits) ou (None, None) si aucune intention détectée
        """
        message_lower = message.lower()
        
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data["patterns"]:
                match = re.search(pattern, message_lower, re.IGNORECASE)
                if match:
                    # Extraire les groupes capturés (paramètres)
                    params = []
                    for i in range(1, len(match.groups()) + 1):
                        if match.group(i):
                            params.append(match.group(i).strip())
                    
                    return intent_name, params
        
        # Si aucun intent n'est détecté
        return None, None
    
    def get_intent_examples(self, intent_name: str) -> List[str]:
        """
        Obtient les exemples pour une intention spécifique.
        
        Args:
            intent_name: Nom de l'intention
            
        Returns:
            Liste d'exemples pour cette intention
        """
        if intent_name in self.intents:
            return self.intents[intent_name]["examples"]
        return []
    
    def get_all_intents(self) -> List[str]:
        """
        Obtient la liste de toutes les intentions disponibles.
        
        Returns:
            Liste des noms d'intentions
        """
        return list(self.intents.keys())
    
    def get_all_examples(self) -> Dict[str, List[str]]:
        """
        Obtient tous les exemples pour toutes les intentions.
        
        Returns:
            Dictionnaire {nom_intention: [exemples]}
        """
        return {intent: data["examples"] for intent, data in self.intents.items()}


if __name__ == "__main__":
    # Test du détecteur d'intentions
    detector = IntentDetector()
    
    test_messages = [
        "Ouvre Chrome",
        "Je veux utiliser Excel",
        "Quelle heure est-il ?",
        "Envoie un mail à Jean",
        "Bonjour",
        "Recherche météo à Paris",
        "Note que j'ai un rendez-vous demain"
    ]
    
    for message in test_messages:
        intent, params = detector.detect_intent(message)
        print(f"Message: '{message}'")
        print(f"Intent détecté: {intent}")
        print(f"Paramètres: {params}")
        print("-" * 50) 