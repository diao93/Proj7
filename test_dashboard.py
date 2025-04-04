from dashboard import get_prediction


def test_get_prediction_accord():
    assert get_prediction(192535) == (0.481, 'Accordé')


def test_get_prediction_refus():
    assert get_prediction(420554) == (0.678, 'Refusé')

