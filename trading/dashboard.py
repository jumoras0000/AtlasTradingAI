"""Local read-only API. Run: uvicorn trading.dashboard:app --host 127.0.0.1 --port 8000"""
from pathlib import Path
import json
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title='AtlasTradingAI - research-only')


@app.get('/api/status')
def status():
    reports = []
    for p in sorted(Path('reports').glob('*.json')):
        try:
            reports.append({'name': p.stem, 'result': json.loads(p.read_text(encoding='utf8'))})
        except (OSError, json.JSONDecodeError):
            pass
    return {'mode': 'OFFLINE / NO LIVE ORDERS', 'reports': reports}


@app.get('/api/live')
def live_status():
    """Live trading status — returns positions and account if MT5 is available."""
    try:
        from .mt5_executor import get_account_info, get_positions
        from .reconciliation import check_session_hours
        account = get_account_info()
        positions = get_positions()
        session = check_session_hours()
        return {
            'connected': True,
            'account': account,
            'positions': positions,
            'session': session,
        }
    except Exception as e:
        return {'connected': False, 'error': str(e)}


@app.get('/api/trade-log')
def trade_log():
    """Return recent trade log entries."""
    log_file = Path('logs/trades.jsonl')
    if not log_file.exists():
        return {'entries': []}
    lines = log_file.read_text(encoding='utf-8').strip().split('\n')
    entries = []
    for line in lines[-100:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {'entries': entries}


@app.get('/api/equity')
def equity():
    """Return equity curve data from latest backtest CSV."""
    csvs = sorted(Path('reports').glob('*.csv'))
    result = {}
    for csv in csvs:
        if 'folds' in csv.stem:
            continue
        try:
            df = pd.read_csv(csv)
            if 'equity' in df.columns and 'time' in df.columns:
                result[csv.stem] = {
                    'time': df['time'].tolist(),
                    'equity': df['equity'].tolist(),
                }
        except Exception:
            pass
    return result


@app.get('/api/walkforward')
def walkforward():
    """Return walk-forward fold results if available."""
    p = Path('reports/walkforward.json')
    if not p.exists():
        return {'available': False}
    try:
        data = json.loads(p.read_text(encoding='utf8'))
        return {'available': True, **data}
    except (OSError, json.JSONDecodeError):
        return {'available': False}


@app.get('/', response_class=HTMLResponse)
def index():
    return '''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AtlasTradingAI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#111827;--surface:#1f2937;--text:#f9fafb;--accent:#3b82f6;--green:#10b981;--red:#ef4444;--border:#374151}
body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:1rem;background:var(--bg);color:var(--text)}
.container{max-width:1100px;margin:0 auto}
h1{font-size:1.5rem;margin-bottom:.25rem}
.subtitle{color:#9ca3af;font-size:.875rem;margin-bottom:1.5rem}
.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));margin-bottom:1.5rem}
.card{background:var(--surface);border-radius:12px;padding:1.25rem;border:1px solid var(--border)}
.card h3{font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;color:#9ca3af;margin:0 0 .5rem}
.card .value{font-size:1.5rem;font-weight:700}
.positive{color:var(--green)}.negative{color:var(--red)}
.chart-box{background:var(--surface);border-radius:12px;padding:1.25rem;border:1px solid var(--border);margin-bottom:1rem}
.chart-box h2{font-size:1rem;margin:0 0 1rem}
canvas{max-height:300px}
pre{white-space:pre-wrap;background:var(--surface);padding:1.25rem;border-radius:12px;font-size:.8rem;border:1px solid var(--border);max-height:400px;overflow:auto}
.tab-row{display:flex;gap:.5rem;margin-bottom:1rem}
.tab{padding:.4rem .8rem;border-radius:8px;background:var(--surface);border:1px solid var(--border);cursor:pointer;font-size:.8rem;color:var(--text)}
.tab.active{background:var(--accent);border-color:var(--accent)}
.section{display:none}.section.active{display:block}
</style>
</head>
<body>
<div class="container">
<h1>AtlasTradingAI</h1>
<p class="subtitle">Laboratoire de recherche — simulation uniquement, aucun ordre réel</p>

<div class="grid" id="kpi-grid"></div>

<div class="tab-row">
  <div class="tab active" onclick="showSection('equity')">Equity</div>
  <div class="tab" onclick="showSection('walkforward')">Walk-Forward</div>
  <div class="tab" onclick="showSection('live')">Live MT5</div>
  <div class="tab" onclick="showSection('reports')">Rapports</div>
</div>

<div class="section active" id="sec-equity">
  <div class="chart-box"><h2>Courbe d'Equity</h2><canvas id="equityChart"></canvas></div>
</div>

<div class="section" id="sec-walkforward">
  <div class="chart-box"><h2>Walk-Forward: Accuracy par Fold</h2><canvas id="wfAccChart"></canvas></div>
  <div class="chart-box"><h2>Walk-Forward: Return par Fold</h2><canvas id="wfRetChart"></canvas></div>
  <div id="wf-summary"></div>
</div>

<div class="section" id="sec-live">
  <div class="grid" id="live-account"></div>
  <div class="chart-box"><h2>Positions ouvertes</h2><div id="live-positions">Connexion MT5 requise...</div></div>
  <div class="chart-box"><h2>Journal de trades</h2><pre id="trade-log" style="max-height:300px;overflow:auto">Aucun trade</pre></div>
</div>

<div class="section" id="sec-reports">
  <pre id="data">Chargement…</pre>
</div>
</div>

<script>
const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6'];

function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('sec-' + name).classList.add('active');
  event.target.classList.add('active');
}

function kpiCard(title, value, cls) {
  return `<div class="card"><h3>${title}</h3><div class="value ${cls||''}">${value}</div></div>`;
}

async function init() {
  // Status & KPIs
  const st = await fetch('/api/status').then(r => r.json());
  document.getElementById('data').textContent = JSON.stringify(st, null, 2);
  const grid = document.getElementById('kpi-grid');
  const baseline = st.reports.find(r => r.name === 'baseline');
  if (baseline) {
    const b = baseline.result;
    const ret = b.net_return_pct;
    grid.innerHTML =
      kpiCard('Return', ret.toFixed(2) + '%', ret >= 0 ? 'positive' : 'negative') +
      kpiCard('Max Drawdown', b.max_drawdown_pct.toFixed(2) + '%', 'negative') +
      kpiCard('Total Costs', '$' + b.total_cost.toFixed(2)) +
      kpiCard('Rebalances', b.rebalance_events);
  }

  // Equity curves
  const eq = await fetch('/api/equity').then(r => r.json());
  const eqCtx = document.getElementById('equityChart').getContext('2d');
  const datasets = [];
  let i = 0;
  for (const [name, d] of Object.entries(eq)) {
    datasets.push({
      label: name, data: d.equity, borderColor: colors[i % colors.length],
      borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.1
    });
    i++;
  }
  if (datasets.length) {
    const labels = Object.values(eq)[0].time.map((t, idx) => idx);
    new Chart(eqCtx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, plugins: { legend: { labels: { color: '#f9fafb' } } },
        scales: {
          x: { display: false },
          y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
        }
      }
    });
  }

  // Walk-forward
  const wf = await fetch('/api/walkforward').then(r => r.json());
  if (wf.available && wf.folds) {
    const foldLabels = wf.folds.map((_, i) => 'Fold ' + (i + 1));
    const accData = wf.folds.map(f => f.accuracy);
    const retData = wf.folds.map(f => f.return_pct);

    new Chart(document.getElementById('wfAccChart').getContext('2d'), {
      type: 'bar',
      data: { labels: foldLabels, datasets: [{ label: 'Accuracy', data: accData, backgroundColor: '#3b82f6' }] },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#f9fafb' } } },
        scales: {
          x: { ticks: { color: '#9ca3af' } },
          y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' }, min: 0, max: 1 }
        }
      }
    });

    new Chart(document.getElementById('wfRetChart').getContext('2d'), {
      type: 'bar',
      data: {
        labels: foldLabels,
        datasets: [{
          label: 'Return %', data: retData,
          backgroundColor: retData.map(r => r >= 0 ? '#10b981' : '#ef4444')
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#f9fafb' } } },
        scales: {
          x: { ticks: { color: '#9ca3af' } },
          y: { ticks: { color: '#9ca3af' }, grid: { color: '#374151' } }
        }
      }
    });

    document.getElementById('wf-summary').innerHTML =
      '<div class="grid">' +
      kpiCard('Folds', wf.n_folds) +
      kpiCard('Mean Accuracy', (wf.mean_accuracy * 100).toFixed(1) + '%') +
      kpiCard('Mean Return', wf.mean_return_pct.toFixed(2) + '%', wf.mean_return_pct >= 0 ? 'positive' : 'negative') +
      kpiCard('Cumul Return', wf.cumulative_return_pct.toFixed(2) + '%', wf.cumulative_return_pct >= 0 ? 'positive' : 'negative') +
      '</div>';
  }

  // Live MT5 panel
  try {
    const live = await fetch('/api/live').then(r => r.json());
    const liveDiv = document.getElementById('live-account');
    if (live.connected) {
      const a = live.account;
      liveDiv.innerHTML =
        kpiCard('Balance', '$' + a.balance.toFixed(2)) +
        kpiCard('Equity', '$' + a.equity.toFixed(2)) +
        kpiCard('Free Margin', '$' + a.margin_free.toFixed(2)) +
        kpiCard('Server', a.server) +
        kpiCard('Market', live.session.market_open ? '<span class="positive">OPEN</span>' : '<span class="negative">CLOSED</span>');

      const posDiv = document.getElementById('live-positions');
      if (live.positions.length === 0) {
        posDiv.textContent = 'Aucune position ouverte';
      } else {
        let html = '<table style="width:100%;border-collapse:collapse;font-size:.85rem">' +
          '<tr style="border-bottom:1px solid var(--border)"><th>Ticket</th><th>Symbol</th><th>Type</th><th>Volume</th><th>Prix</th><th>P/L</th></tr>';
        for (const p of live.positions) {
          const cls = p.profit >= 0 ? 'positive' : 'negative';
          html += `<tr><td>${p.ticket}</td><td>${p.symbol}</td><td>${p.type}</td><td>${p.volume}</td><td>${p.price_open.toFixed(5)}</td><td class="${cls}">${p.profit.toFixed(2)}</td></tr>`;
        }
        html += '</table>';
        posDiv.innerHTML = html;
      }
    } else {
      liveDiv.innerHTML = kpiCard('MT5', '<span class="negative">Non connecté</span>');
      document.getElementById('live-positions').textContent = live.error || 'MT5 non disponible';
    }

    const tlog = await fetch('/api/trade-log').then(r => r.json());
    if (tlog.entries.length > 0) {
      document.getElementById('trade-log').textContent = tlog.entries.map(e => JSON.stringify(e)).join('\\n');
    }
  } catch(e) {
    document.getElementById('live-account').innerHTML = kpiCard('MT5', '<span class="negative">Non connecté</span>');
  }
}
init();
</script>
</body>
</html>'''
