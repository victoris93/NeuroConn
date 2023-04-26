from PyConn.data.example_datasets import depression, cobre
from PyConn.gradient.gradient import get_gradients

def test_get_gradients():
    gradients = get_gradients(cobre, 'A00000541', n_components = 10, task = "rest", aligned = False)
    if len(gradients.shape) == 3:
        assert gradients.shape[0] == 2
        assert gradients.shape[1] == 1000
        assert gradients.shape[2] == 10
    else:
        assert gradients.shape[0] == 1000
        assert gradients.shape[1] == 10

def test_get_gradients_aligned():
    gradients = get_gradients(cobre, 'A00000541', n_components = 10, task = "rest", aligned = True)
    assert len(gradients.shape) == 3
    assert gradients.shape[1] == 1000
    assert gradients.shape[2] == 10