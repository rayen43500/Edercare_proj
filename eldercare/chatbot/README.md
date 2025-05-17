# ElderCare Assistant

## Présentation du Projet

ElderCare Assistant est un chatbot intelligent conçu spécifiquement pour aider les personnes âgées dans leur quotidien. Ce projet implémente un assistant virtuel capable de répondre aux questions, fournir de l'information, et exécuter des commandes système simples comme l'ouverture d'applications.

## Fonctionnalités Principales

- **Conversation intelligente**: Répond à des questions sur divers sujets adaptés aux besoins des personnes âgées
- **Base de connaissances spécialisée**: Informations organisées en catégories pertinentes (santé, technologie, divertissement)
- **Exécution de commandes système**: Peut ouvrir des applications comme Chrome, Word, etc.
- **Interface simple**: Communication en langage naturel, facile à utiliser
- **Support multilingue**: Fonctionne principalement en français
- **Modèles d'IA hybrides**: Utilise des modèles locaux et distants selon les besoins
- **Intégration Azure AI**: Support pour les modèles avancés via Azure AI Inference SDK

## Structure du Projet

```
.
├── chatbot_run.py           # Script pour exécuter le chatbot
├── improved_chatbot.py      # Version améliorée du chatbot avec fonctionnalités avancées
├── main.py                  # Implémentation principale du chatbot
├── main2.py                 # Version alternative du chatbot
├── final.py                 # Version finale optimisée
├── run_chatbot.py           # Script simple pour exécuter le chatbot
├── simple_chatbot.py        # Version simplifiée du chatbot
├── system_commands.py       # Module pour exécuter des commandes système (Chrome, etc.)
├── eldercare_assistant.py   # Script principal de l'assistant ElderCare avec commandes système
├── azure_llm_integration.py # Module d'intégration avec Azure AI Inference SDK
├── eldercare_assistant_azure.py # Version améliorée utilisant Azure AI pour des réponses plus naturelles
├── requirements.txt         # Dépendances Python requises
├── knowledge_base/          # Base de connaissances pour l'assistant
│   ├── health/              # Informations sur la santé et le bien-être
│   ├── technology/          # Guides d'utilisation technologique
│   ├── entertainment/       # Recommandations de divertissement
│   └── companion/           # Contenu pour la conversation générale
└── faiss_index/             # Index vectoriel pour la recherche sémantique
```

## Composants Techniques

1. **Base de Connaissances**
   - Utilise LangChain et FAISS pour la recherche sémantique
   - Documents organisés en catégories thématiques
   - Extraction contextuelle intelligente

2. **Moteurs d'IA**
   - Intégration de modèles de langage (LLM)
   - Support pour OpenAI et modèles locaux
   - Système de fallback en cas d'erreur
   - Intégration avec Azure AI Inference SDK pour des modèles avancés comme GPT-4.1

3. **Exécution de Commandes**
   - Interface avec le système d'exploitation
   - Liste d'applications autorisées pour la sécurité
   - Reconnaissance d'intention pour détecter les commandes

4. **Interface Utilisateur**
   - Interface en ligne de commande simple
   - Traitement du langage naturel
   - Gestion des erreurs avec suggestions

## Installation et Utilisation

### Prérequis
- Python 3.8 ou supérieur
- Pip (gestionnaire de paquets Python)
- GitHub Personal Access Token pour Azure AI (optionnel)

### Installation

1. Cloner le dépôt ou télécharger les fichiers

2. Installer les dépendances:
   ```
   pip install -r requirements.txt
   ```

3. Configurer les variables d'environnement:
   - Pour l'utilisation d'Azure AI:
     ```
     # Bash
     export GITHUB_TOKEN="votre_token_github"
     
     # Windows PowerShell
     $Env:GITHUB_TOKEN="votre_token_github"
     
     # Windows CMD
     set GITHUB_TOKEN=votre_token_github
     ```
   - Pour d'autres services (optionnel):
     - Créer un fichier `.env` avec les clés API nécessaires
     - Format: `OPENAI_API_KEY=votre_clé_api`

### Utilisation

1. Lancer l'assistant standard:
   ```
   python eldercare_assistant.py
   ```

2. Lancer l'assistant avec Azure AI (recommandé):
   ```
   python eldercare_assistant_azure.py
   ```

3. Interagir avec l'assistant:
   - Poser des questions: "Comment puis-je soulager mon arthrite?"
   - Demander de l'aide technologique: "Comment utiliser WhatsApp?"
   - Ouvrir des applications: "Peux-tu ouvrir Chrome pour moi?"

4. Commandes spéciales:
   - `aide`: Afficher l'aide
   - `quitter`: Quitter l'application

## Cas d'Usage Typiques

1. **Assistance technologique**
   - "Comment puis-je envoyer une photo par WhatsApp?"
   - "Peux-tu m'expliquer comment faire un appel vidéo?"

2. **Conseils de santé**
   - "Quels exercices sont bons pour l'arthrite?"
   - "Comment améliorer mon sommeil?"

3. **Recommandations de divertissement**
   - "Suggère-moi des livres de mystère faciles à lire"
   - "Quelles émissions de jardinage pourrais-je regarder?"

4. **Contrôle d'applications**
   - "Ouvre Chrome pour moi"
   - "Lance l'application Photos"

## Versions de l'Assistant

1. **Version Standard (eldercare_assistant.py)**
   - Fonctionnalités de base
   - Exécution de commandes système
   - Base de connaissances locale

2. **Version Azure (eldercare_assistant_azure.py)**
   - Réponses plus naturelles et détaillées
   - Utilise le modèle GPT-4.1 via Azure AI
   - Meilleure compréhension du contexte
   - Nécessite un GitHub Personal Access Token

## Développement Futur

- Intégration de la reconnaissance vocale
- Interface graphique simplifiée
- Support pour plus de commandes système
- Personnalisation avancée selon les besoins de l'utilisateur
- Synchronisation avec des dispositifs médicaux 