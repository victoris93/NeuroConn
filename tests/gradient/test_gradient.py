from PyConn.gradient.gradient import get_gradients
from PyConn.data.example_datasets import fetch_example_data

example_data = fetch_example_data()

def test_get_gradients():
    gradients = get_gradients(example_data, '17017', n_components = 10, task = "rest", aligned = False)
    if len(gradients.shape) == 3:
        assert gradients.shape[0] == 1, "First dimension should be 1 (n_sessions)"
        assert gradients.shape[1] == 1000, "Second dimension should be 1000 (n_parcels)"
        assert gradients.shape[2] == 10, "Third dimension should be 10 (n_components)"
    else:
        assert gradients.shape[0] == 1000, "First dimension should be 1000 (n_parcels)"
        assert gradients.shape[1] == 10, "Second dimension should be 10 (n_components)"

def test_get_gradients_aligned():
    gradients = get_gradients(example_data, '17017', n_components = 10, task = "rest", aligned = True)
    assert len(gradients.shape) == 3, "Aligned gradients should be a 3D array"
    assert gradients.shape[1] == 1000, "Second dimension should be 1000 (n_parcels)"
    assert gradients.shape[2] == 10, "Third dimension should be 10 (n_components)"