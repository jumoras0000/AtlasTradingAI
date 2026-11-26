# AtlasTradingAI — laboratoire de trading IA (simulation uniquement)

**Ce dépôt est un prototype de recherche exécutable, pas un robot rentable prouvé.**
Pas d'ordres réels, pas de fonction `order_send`, aucune connexion automatique à un compte réel.
La stratégie EMA et le modèle PPO sont des points de départ expérimentaux, **pas des recommandations de trading**.

## Installation PC Windows (PowerShell, Python 3.11 ou 3.12 conseillé)

```powershell
cd AtlasTradingAI
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m trading.cli generate-demo
python -m trading.cli backtest
python -m trading.cli train-xgb
python -m pytest -q
uvicorn trading.dashboard:app --host 127.0.0.1 --port 8000
```

Ouvrir `http://127.0.0.1:8000` (tableau de bord local en lecture seule).
Si `py -3.11` n'est pas installé, utiliser Python 3.12 ou `python -m venv .venv`.
Les cours de `data/demo.csv` sont **synthétiques**, uniquement pour vérifier le pipeline.

## Entraînement PPO sur le serveur Jupyter ESIEA (terminal Linux)

```bash
cd AtlasTradingAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-rl.txt
python -m trading.cli generate-demo
python -m trading.cli train-ppo --steps 15000
python -m pytest -q
```

Si l'école l'autorise, lancer les longs travaux via `tmux` ; les checkpoints PPO sont dans `models/`.
Le matériel 32 Go RAM est utile, mais il n'y a aucune garantie de capacité ou de disponibilité continue.

## Données réelles (sans exécuter d'ordres)

Option A : fournir un CSV dans `data/` aux colonnes exactes :
`time,open,high,low,close,volume`, tri chronologique, UTC recommandé.
Exemple : `python -m trading.cli backtest --csv data/mon_historique.csv --prefix mon_test`.

Option B : sur **Windows avec MetaTrader 5 installé, connecté à un compte DEMO**, installer séparément :

```powershell
pip install -r requirements-mt5.txt
python -m trading.cli export-mt5 --symbol EURUSD --count 2000 --output data/mt5_demo.csv
python -m trading.cli backtest --csv data/mt5_demo.csv --prefix mt5_demo
```

Ce connecteur refuse explicitement les comptes non-démo et n'envoie **aucun ordre**.
MetaTrader 5 n'est pas exécuté sur le serveur Jupyter Linux dans cette architecture.

## Actualités

```bash
python -m trading.cli collect-news
```

Tente de récupérer un flux RSS public BCE dans `data/news.json`. Nécessite une connexion Internet ;
le serveur distant peut en interdire l'accès. Les nouvelles sont journalisées et taguables,
**mais elles ne pilotent PAS automatiquement PPO dans cette version** : l'historisation
horodatée, l'alignement des périodes et la validation sont encore nécessaires. Pas de LLM livré :
les poids, la licence et les ressources sont dépendants du modèle choisi.

## Où se trouvent les éléments ?

- `trading/data.py` : données et indicateurs causaux.
- `trading/backtest.py` : moteur de simulation offline, exécution au bar suivant et frais.
- `trading/risk.py` : limites d'exposition / pertes et arrêt d'urgence.
- `trading/ml.py` : apprentissage XGBoost, coupe chronologique.
- `trading/rl.py` : environnement Gymnasium et entraînement PPO optionnel.
- `trading/mt5_bridge.py` : export de bougies MT5 uniquement, compte DEMO.
- `trading/news.py` : collecte RSS économique (réseau nécessaire).
- `trading/dashboard.py` : dashboard local **lecture seule**.
- `models/` : modèles créés après entraînement.
- `reports/` : fichiers JSON et CSV des backtests.
- `tests/` : tests automatisés et non-régression.
- `docs/ARCHITECTURE.md` : architecture, risques et améliorations nécessaires.
- `.env.example` : exemple de configuration, aucune clé nécessaire en V1.

## État de validation

`python -m pytest -q` vérifie l'ingestion, les OHLC, le risque, les frais, l'absence d'exécution
à partir d'un signal de future bougie, le modèle XGBoost et le dashboard.
Une réussite de ces tests ne valide **pas** la profitabilité. PPO, la bibliothèque MetaTrader5 et le
RSS en direct nécessitent une vérification sur les environnements visés ; voir `docs/TEST_REPORT.md`.

## Sécurité

Aucune garantie financière. Ne pas coller d'identifiants ou clés dans le code. Les connexions aux
courtiers et la fiscalité dépendent de la juridiction et de la disponibilité du prestataire.
Le recours à un serveur de l'ESIEA doit être conforme au règlement de l'école.
