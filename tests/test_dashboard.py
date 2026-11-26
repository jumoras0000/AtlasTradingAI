from trading.dashboard import status,index

def test_read_only_status():
    assert status()['mode']=='OFFLINE / NO LIVE ORDERS'
    assert 'Aucun ordre réel' in index()
