import numpy as np
import os
from NeuroConn.preprocessing.preprocessing import RawDataset, FmriPreppedDataSet
from NeuroConn.data.example_datasets import fetch_example_data
from NeuroConn.gradient.gradient import get_gradients

example_data = fetch_example_data('https://drive.google.com/file/d/1XjF5wDJXHzMyfoAjQE6NW2xcj9PulZzH/view?usp=share_link')

def test_subjects_attr():
    raw_data = RawDataset(example_data)
    subjects = raw_data.subjects
    assert len(subjects) > 0, "The subject list is empty"

def test_subject():
    raw_data = RawDataset(example_data)
    subjects = raw_data.subjects
    assert "53" in subjects, "The subject is not on the list"

def test_session_names():
    subject = '53'
    fmriprepped_data = FmriPreppedDataSet(example_data)
    session_names_sub1 = fmriprepped_data.get_sessions(subject)
    assert session_names_sub1 == []

def test_clean_signal_shape():
    subject = '53'
    n_parcels = 1000
    fmriprepped_data = FmriPreppedDataSet(example_data)
    clean_ts_= np.asarray(fmriprepped_data.clean_signal(subject, task = "rest"))

    assert clean_ts_.shape[0] == 1, "First dimension should be 1 (n_sessions)"
    assert clean_ts_.shape[2] == n_parcels, "Third dimension should be 1000 (n_parcels)"

def test_conn_matrix():
    subject = '53'
    fmriprepped_data = FmriPreppedDataSet(example_data)
    conn_matrix = fmriprepped_data.get_conn_matrix(subject, task = "rest", save = True, z_transformed=True)
    path_conn_matrix = os.path.join(f'{fmriprepped_data.data_path}', 'clean_data', f'sub-{subject}', 'func', f'z-conn-matrix-sub-{subject}-rest-schaefer1000.npy')

    assert conn_matrix.shape[0] == 1, "First dimension should be 2 (n_sessions)"
    assert conn_matrix.shape[1] == 1000, "Second dimension should be 1000 (n_parcels)"
    assert conn_matrix.shape[2] == 1000, "Third dimension should be 1000 (n_parcels)"
    
    assert os.path.exists(path_conn_matrix), "Matrix was not saved"

def test_get_gradients():
    gradients = get_gradients(example_data, subject = '53', n_components = 10, task = "rest", aligned = False)
    if len(gradients.shape) == 3:
        assert gradients.shape[0] == 1, "First dimension should be 1 (n_sessions)"
        assert gradients.shape[1] == 1000, "Second dimension should be 1000 (n_parcels)"
        assert gradients.shape[2] == 10, "Third dimension should be 10 (n_components)"
    else:
        assert gradients.shape[0] == 1000, "First dimension should be 1000 (n_parcels)"
        assert gradients.shape[1] == 10, "Second dimension should be 10 (n_components)"

def test_get_gradients_aligned():
    gradients = get_gradients(example_data, '53', n_components = 10, task = "rest", aligned = True)
    assert len(gradients.shape) == 3, "Aligned gradients should be a 3D array"
    assert gradients.shape[1] == 1000, "Second dimension should be 1000 (n_parcels)"
    assert gradients.shape[2] == 10, "Third dimension should be 10 (n_components)"