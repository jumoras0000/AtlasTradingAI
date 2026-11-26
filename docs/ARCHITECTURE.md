# Architecture et analyse critique

## Portée réellement livrée

1. CSV OHLCV validé, indicateurs calculés sans données futures.
2. Signaux EMA 20/50 et moteur offline long-only sans levier. Les ordres sont simulés à l'ouverture **suivante**.
3. Coûts synthétiques par turnover en points de base. Ce n'est PAS un moteur de microstructure.
4. Réducteur de risque avec plafonds d'exposition, perte quotidienne et drawdown.
5. XGBoost supervisé avec séparation temporelle et métrique de classification hors entraînement.
6. PPO Gymnasium/Stable-Baselines3 facultatif, entraîné sur la partie historique initiale seulement.
7. Export de prix MT5 DEMO, sans ordres ni fonctionnalité de compte réel.
8. Collecte RSS facultative et dashboard local lecture seule.

## Limites de rigueur des expériences

- Les coûts réels incluent spread bid-ask, commission, glissement, swaps et restrictions de taille selon le courtier ; le prototype ne modélise que des frais/slippage forfaitaires.
- L'environnement PPO est une simplification : récompense du rendement de la bougie suivante à l'ouverture, sans coûts de financement ni ordre partiellement exécuté. Il ne reproduit pas la totalité du backtester.
- La gestion des risques PPO doit être intégrée à son environnement avant toute comparaison finale ; l'algorithme livré est un **exemple d'entraînement**, non validé comme politique de trading.
- L'environnement PPO n'est pas encore évalué hors échantillon ni branché sur des signaux d'exécution. Le simple entraînement ne constitue pas une validation.
- La stratégie EMA, les prédictions XGBoost et PPO n'ont aucune preuve d'avantage statistique net.
- Le RSS est collecté sur demande, pas en permanence. Il faut vérifier la licence des données, les délais et l'heure de disponibilité. Un modèle LLM local n'est pas inclus ni prétéléchargé.
- Une architecture multi-actifs véritable exige des adaptateurs indépendants, la conversion des devises, des calendriers, des devises de règlement, une comptabilité et des risques au niveau portefeuille.
- Les limites du risque sont des garde-fous logiciels et ne garantissent pas les pertes maximales en marché réel.

## Étapes avant de connecter la décision à l'exécution démo

1. Collecter des données réelles, horodatées, vérifiées et suffisamment longues pour plusieurs régimes.
2. Calibrer les frais et hypothèses de remplissage avec des instruments du compte DEMO.
3. Créer un outil d'évaluation PPO **hors échantillon** indépendant du train, de préférence walk-forward.
4. Implémenter le même risque pour PPO et l'exécution ; vérifier la réconciliation des positions, le rejet et les doublons.
5. Construire l'adaptateur de passage d'ordres **DEMO ONLY**, d'abord avec approbation manuelle puis simulation automatique.
6. Contrôler les heures de session, données manquantes, pannes du terminal et déconnexions.
7. Ajouter actualités de manière horodatée, empêcher lookahead, comparer par ablation.
8. Vérifier règlement de l'ESIEA et courtier autorisé en France.

**Ne pas ouvrir de trading réel depuis ce dépôt.** La priorité est la reproductibilité des expériences.
