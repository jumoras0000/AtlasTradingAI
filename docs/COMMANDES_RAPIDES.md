# Guide « où mettre quoi »

- **Sur PC Windows** : extraire `AtlasTradingAI` dans `C:\\Projets\\AtlasTradingAI` et installer MetaTrader 5 à part. Lancer le terminal PowerShell dans ce dossier. Les commandes exactes figurent dans README.
- **Sur serveur ESIEA** : copier uniquement le dossier `AtlasTradingAI` dans ton espace personnel autorisé, installer `requirements-rl.txt` dans un environnement virtuel, puis entraîner PPO. Ne copier aucun identifiant de courtier sur le serveur.
- **Pour les données** : placer ton historique dans `data/nom.csv` (format défini dans README), puis lancer `python -m trading.cli backtest --csv data/nom.csv`.
- **Pour les modèles entraînés** : `models/xgb.json` et `models/ppo.zip` sont créés par leurs commandes. Ces poids ne sont PAS inclus car ils dépendent des données réellement choisies.
- **Pour les rapports** : consulter `reports/baseline.json` et `reports/baseline.csv` ; tableau de bord à `http://127.0.0.1:8000` après démarrage du serveur.
- **Pour les news** : `python -m trading.cli collect-news` (connexion Internet requise), résultat `data/news.json`. Les news n'activent aucun ordre.
- **Pour MetaTrader** : compte DEMO, symbole réel du courtier (parfois `EURUSD.a`), plugin Python installé sous Windows, commande `export-mt5`.

Tout est OFFLINE / démonstration. Les modèles ne doivent jamais être considérés prêts pour l'argent réel après un simple entraînement.
