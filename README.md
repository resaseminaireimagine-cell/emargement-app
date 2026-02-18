# Outil d’émargement — Institut Imagine

Application Streamlit pour gérer l’émargement sur tablette/ordinateur pendant un événement.

## Fonctionnalités
- Import d’un fichier Excel (`.xlsx`) avec détection souple des colonnes (`prénom/firstname`, `nom/lastname`, etc.).
- Recherche robuste (accents/apostrophes ignorés, ordre prénom/nom libre).
- Émargement rapide / annulation, avec horodatage Europe/Paris et nom d’agent.
- Autosauvegarde automatique par fichier importé.
- Exports CSV (compatibles Excel FR) + Excel.
- Préparation d’email (mailto) pour envoi des exports.

## Prérequis
- Python 3.11+
- Dépendances listées dans `requirements.txt`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l’application
```bash
streamlit run app.py
```

## Format de fichier conseillé
Le fichier Excel doit contenir au minimum des colonnes identifiantes (email ou nom/prénom). Les alias suivants sont supportés automatiquement :
- Prénom : `first_name`, `firstname`, `first name`, `prenom`, `prénom`
- Nom : `last_name`, `lastname`, `last name`, `nom`
- Email : `email`, `e-mail`, `mail`, `courriel`
- Société : `company`, `organisation`, `societe`, `société`
- Fonction : `function`, `fonction`, `job`, `poste`

Les colonnes d’état (`present`, `checkin_time`, `checkin_by`) sont ajoutées si absentes.

## Notes d’exploitation
- Les autosauvegardes sont écrites dans `/tmp/imagine_emargement_autosave`.
- Le fichier logo est détecté automatiquement parmi :
  - `logo_rose.png`
  - `LOGO ROSE.png`
  - `LOGO_ROSE.png`
  - `logo.png`
