"""Optional official RSS headlines collector. Not a real-time licensed news feed."""
from urllib.request import Request,urlopen
from xml.etree import ElementTree as ET
from datetime import datetime,timezone
import json
from pathlib import Path

# Public agency feed; availability may change. No credentials necessary.
FEEDS={'ecb':'https://www.ecb.europa.eu/rss/press.html'}

def fetch_rss(url,timeout=8):
    req=Request(url,headers={'User-Agent':'AtlasTradingAI-Research/0.1'})
    with urlopen(req,timeout=timeout) as response:
        data=response.read(2_000_000)
    root=ET.fromstring(data)
    items=[]
    for item in root.findall('.//item')[:50]:
        items.append({'title':(item.findtext('title') or '').strip(),
                      'url':(item.findtext('link') or '').strip(),
                      'published':(item.findtext('pubDate') or '').strip(),
                      'collected_at_utc':datetime.now(timezone.utc).isoformat(),
                      'source_url':url})
    return items

def collect(path='data/news.json',feeds=None):
    feeds=feeds or FEEDS
    records=[];errors={}
    for name,url in feeds.items():
        try:
            for r in fetch_rss(url):r['feed']=name;records.append(r)
        except (OSError,ET.ParseError,ValueError) as e:errors[name]=str(e)
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({'items':records,'errors':errors},indent=2,ensure_ascii=False),encoding='utf8')
    return {'count':len(records),'errors':errors,'path':str(dest)}

def keyword_tag(headline):
    """Transparent non-LLM fallback; NOT a predictive trading signal."""
    txt=headline.lower()
    keys={'inflation':['inflation','consumer prices','cpi'],
          'rates':['interest rate','monetary policy','rate decision'],
          'employment':['unemployment','employment','jobs']}
    return [name for name,words in keys.items() if any(w in txt for w in words)]
