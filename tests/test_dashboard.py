from trading.dashboard import status, index


def test_read_only_status():
    assert status()['mode'] == 'OFFLINE / NO LIVE ORDERS'
    assert 'aucun ordre' in index().lower()


def test_equity_endpoint():
    from trading.dashboard import equity
    result = equity()
    assert isinstance(result, dict)


def test_walkforward_endpoint():
    from trading.dashboard import walkforward
    result = walkforward()
    assert 'available' in result
