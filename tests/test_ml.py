from trading.data import generate_demo
from trading.ml import train_xgb

def test_train_xgb(tmp_path):
    p=generate_demo(tmp_path/'data.csv',n=350)
    result=train_xgb(p,output=tmp_path/'xgb.json')
    assert 0<=result['test_accuracy']<=1
    assert (tmp_path/'xgb.json').exists()
    assert result['test_bars']>=20
