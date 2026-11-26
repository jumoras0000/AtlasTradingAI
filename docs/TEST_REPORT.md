# Rapport de vérification — 23 novembre 2026

## Ce qui a réellement été exécuté ici

- `python -m compileall -q trading tests` : OK, compilation syntaxique des modules.
- `python -m pytest -q` : **9 tests réussis**.
- `python -m trading.cli generate-demo` : OK, génération de **1 600** bougies **synthétiques**.
- `python -m trading.cli backtest` : OK, rapport JSON et journal CSV créés.
- `python -m trading.cli train-xgb` : OK, modèle sauvegardé, séparation temporelle 70/30.
- `trading.dashboard.status()` : OK, mode `OFFLINE / NO LIVE ORDERS`.

## Résultats de la démonstration synthétique

- Capital fictif initial : 10 000 unités.
- Capital fictif final : 9 688,26 unités.
- Rendement fictif : **-3,12 %**.
- Drawdown simulé : **-3,69 %**.
- Coût cumulé simulé : 74,44 unités.
- Précision XGBoost hors entraînement : **46,62 %** sur 474 observations.

Ces chiffres ne sont PAS une prévision ni un backtest sur un marché réel. La démonstration ne démontre aucun avantage financier.

## Ce qui n'a PAS pu être vérifié ici

- L'exécution et l'entraînement PPO : `gymnasium` et `stable-baselines3` absents ; tentative d'installation bloquée par indisponibilité DNS de l'environnement.
- Connexion et export MT5 : paquet propriétaire MetaTrader5 absent, terminal Windows et compte démo indisponibles dans cet environnement.
- Actualités RSS en direct : accès Internet non disponible dans l'environnement de validation.
- Déploiement et performance sur les machines personnelles et le Jupyter ESIEA : non testés.
- Stratégie rentable en conditions réelles : **non établie**.

## Conditions de validation supplémentaires

Avant tout passage à un robot exécutant même des ordres DEMO : tests PPO en environnement disposant des dépendances, vérification de la correspondance entre train et inférence, tests de panne du terminal, synchronisation d'horloge, validation de la source des actualités, tests de réconciliation et limites du courtier.

## Interprétation

Ce ZIP est **prêt à lancer pour la chaîne de démonstration** et les composants vérifiés. Il n'est **pas** un bot de trading IA multimarchés intégralement opérationnel ou certifié sans bugs. Une couverture de tests partielle ne peut jamais prouver l'absence totale de bugs.
