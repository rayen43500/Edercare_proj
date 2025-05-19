# ElderCare Assistant

ElderCare est une application d'assistance pour les personnes âgées qui combine un backend Python avec de l'intelligence artificielle et un frontend React pour une interface conviviale.

## Fonctionnalités

- Interface utilisateur conviviale et accessible
- Reconnaissance d'intentions et exécution d'actions
- Conversation assistée par IA grâce à Azure LLM
- Traitement des messages vocaux (simulation)
- Gestion du profil utilisateur

## Structure du Projet

- `eldercare/` - Backend Python
  - `chatbot/` - Modules du chatbot ElderCare
  - `api.py` - API Flask pour exposer les fonctionnalités ElderCare
  - `start.py` - Script pour lancer l'application complète
  - `requirements.txt` - Dépendances Python

- `frontend/` - Application React/TypeScript
  - `src/` - Code source React
  - `public/` - Ressources statiques
  - `package.json` - Dépendances JavaScript

## Configuration

### Prérequis

- Python 3.8+
- Node.js 14+ et npm
- Un jeton GitHub pour utiliser Azure LLM

### Variables d'Environnement

Créez un fichier `.env` à la racine du dossier `eldercare` avec les variables suivantes :

```
GITHUB_TOKEN=votre_token_github
```

### Installation

1. Clonez ce dépôt
2. Installez les dépendances Python :
   ```
   cd eldercare
   pip install -r requirements.txt
   ```

3. Installez les dépendances JavaScript :
   ```
   cd frontend
   npm install
   ```

## Démarrage

Vous pouvez démarrer l'application complète avec un seul script :

```
cd eldercare
python start.py
```

Ou lancer séparément le frontend et le backend :

### Backend

```
cd eldercare
python api.py
```

### Frontend

```
cd frontend
npm start
```

L'application frontend sera accessible sur http://localhost:3000 et l'API backend sur http://localhost:5000.

## Intégration

L'application intègre :
- Un backend Python avec le module ElderCare complet
- Une API Flask exposant les fonctionnalités ElderCare
- Un frontend React pour une interface utilisateur conviviale
- Communication bidirectionnelle entre frontend et backend

## Développement

Pour le développement, vous pouvez modifier :
- `frontend/src/App.tsx` - Interface utilisateur principale
- `frontend/src/services/eldercare-api.ts` - Service de communication avec l'API
- `eldercare/api.py` - Points d'accès de l'API

## Licence

Ce projet est sous licence MIT. 