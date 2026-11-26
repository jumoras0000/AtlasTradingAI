"""Local read-only API. Run: uvicorn trading.dashboard:app --host 127.0.0.1 --port 8000"""
from pathlib import Path
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
app=FastAPI(title='AtlasTradingAI - research-only')
@app.get('/api/status')
def status():
    reports=[]
    for p in sorted(Path('reports').glob('*.json')):
        try:reports.append({'name':p.stem,'result':json.loads(p.read_text(encoding='utf8'))})
        except (OSError,json.JSONDecodeError):pass
    return {'mode':'OFFLINE / NO LIVE ORDERS','reports':reports}
@app.get('/',response_class=HTMLResponse)
def index():
    return '''<!DOCTYPE html><html lang="fr"><meta charset="utf-8"><title>AtlasTradingAI</title>
    <style>body{font-family:system-ui;max-width:760px;margin:3rem auto;padding:1rem;background:#111827;color:#f9fafb}
    pre{white-space:pre-wrap;background:#1f2937;padding:1.5rem;border-radius:12px}</style>
    <h1>AtlasTradingAI — Laboratoire de recherche</h1>
    <p>Simulation uniquement. Aucun ordre réel ne peut être envoyé depuis cette application.</p>
    <pre id="data">Chargement…</pre><script>fetch('/api/status').then(r=>r.json()).then(d=>
    document.getElementById('data').textContent=JSON.stringify(d,null,2)).catch(e=>
    document.getElementById('data').textContent=String(e))</script></html>'''
