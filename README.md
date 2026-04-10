# Emargement App

Application Streamlit d'emargement pour evenements et seminaires.

## Ce que fait l'application

- importe une liste de participants en `.xlsx` ou `.csv`
- recherche rapidement un participant
- emarge ou annule un participant
- conserve l'etat de session dans l'URL
- exporte les listes completes, presentes et absentes
- affiche une synthese par societe et les derniers emargements

## Fichiers importants

- `app.py` : application principale
- `requirements.txt` : dependances Python
- `sample_participants.csv` : modele de fichier
- `.streamlit/config.toml` : theme Streamlit

## Format de fichier conseille

Colonnes recommandees :

- `first_name`
- `last_name`
- `email`
- `company`
- `function`

Les noms de colonnes proches sont reconnus automatiquement.

## Lancer en local

Si Python est disponible :

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publier gratuitement avec GitHub + Streamlit Community Cloud

GitHub stocke le code. Pour obtenir un vrai lien web, il faut connecter ce depot a Streamlit Community Cloud.

1. Connecte-toi a [Streamlit Community Cloud](https://share.streamlit.io/).
2. Clique sur `Create app`.
3. Selectionne le depot GitHub `resaseminaireimagine-cell/emargement-app`.
4. Choisis la branche `main`.
5. Choisis le fichier principal `app.py`.
6. Clique sur `Deploy`.

Tu obtiendras ensuite un lien du type :

`https://ton-app.streamlit.app`

## Conseils d'usage

- charge un fichier participants
- renseigne le nom de l'agent
- emarge au fil de l'eau
- copie l'URL du navigateur si tu veux reprendre la session sur une autre tablette
- exporte les presentes et absents a la fin
