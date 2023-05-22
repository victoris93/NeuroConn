import numpy as np
import os
from PyConn.preprocessing.preprocessing import RawDataset, FmriPreppedDataSet
from PyConn.data.example_datasets import fetch_example_data

example_data = fetch_example_data()

def test_subjects_attr():
    raw_data = RawDataset(example_data)
    subjects = raw_data.subjects
    assert len(subjects) > 0, "The subject list is empty"

def test_subject():
    raw_data = RawDataset(example_data)
    subjects = raw_data.subjects
    assert "52" in subjects, "The subject is not on the list"

def test_session_names():
    subject = '52'
    fmriprepped_data = FmriPreppedDataSet(example_data)
    session_names_sub1 = fmriprepped_data.get_sessions(subject)
    assert session_names_sub1 == []

def test_clean_signal_shape():
    subject = '52'
    n_parcels = 1000
    fmriprepped_data = FmriPreppedDataSet(example_data)
    clean_ts_= np.asarray(fmriprepped_data.clean_signal(subject, task = "rest"))

    assert clean_ts_.shape[0] == 1, "First dimension should be 1 (n_sessions)"
    assert clean_ts_.shape[2] == n_parcels, "Third dimension should be 1000 (n_parcels)"

def test_conn_matrix():
    subject = '52'
    fmriprepped_data = FmriPreppedDataSet(example_data)
    conn_matrix = fmriprepped_data.get_conn_matrix(subject, task = "rest", save = True)
    path_conn_matrix = os.path.join(f'{fmriprepped_data.data_path}', 'clean_data', f'sub-{subject}', 'func', f'conn-matrix-sub-{fmriprepped_data.subjects[0]}-rest-schaefer1000.npy')

    assert conn_matrix.shape[0] == 1, "First dimension should be 2 (n_sessions)"
    assert conn_matrix.shape[1] == 1000, "Second dimension should be 1000 (n_parcels)"
    assert conn_matrix.shape[2] == 1000, "Third dimension should be 1000 (n_parcels)"
    
    assert os.path.exists(path_conn_matrix), "Matrix was not saved"
