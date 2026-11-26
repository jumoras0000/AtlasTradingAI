from trading.config import load_config, make_settings, make_risk


def test_load_default_config():
    cfg = load_config()
    assert cfg['starting_cash'] == 10000.0
    assert 'profiles' in cfg


def test_make_settings():
    cfg = load_config()
    s = make_settings(cfg)
    assert s.starting_cash == 10000.0
    assert s.fee_bps == 5
    assert s.risk.max_exposure == 0.20


def test_profiles_exist():
    cfg = load_config()
    profiles = cfg['profiles']
    assert 'conservative' in profiles
    assert 'realistic' in profiles
    assert 'zero' in profiles
