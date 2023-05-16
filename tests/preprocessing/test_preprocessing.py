import numpy as np
import os
from PyConn.preprocessing.preprocessing import RawDataset, FmriPreppedDataSet
from PyConn.data.example_datasets import depression, cobre

def test_cobre_subjects():
    raw_data = RawDataset(cobre)
    subjects = raw_data.subjects
    assert len(subjects) > 0


def test_depression_subjects():
    raw_data = RawDataset(depression)
    subjects = raw_data.subjects
    print(subjects)
    assert len(subjects) > 0

def test_subjects():
    raw_data = RawDataset(cobre)
    subjects = raw_data.subjects
    print(subjects)
    assert "A00000541" in subjects

def test_cobre_session_names():
    fmriprepped_data = FmriPreppedDataSet(cobre)
    session_names_sub1 = fmriprepped_data.get_sessions(fmriprepped_data.subjects[0])
    print(session_names_sub1)
    session_names_sub2 = fmriprepped_data.get_sessions(fmriprepped_data.subjects[1])
    print(session_names_sub2)
    assert session_names_sub1 == ['20100101', '20110101']
    assert session_names_sub2 == ['20110101']

def test_ts_paths():
    fmriprepped_cobre = FmriPreppedDataSet(cobre)
    fmriprepped_depression = FmriPreppedDataSet(depression)

    ts_paths_cobre_2ses = fmriprepped_cobre.get_ts_paths(fmriprepped_cobre.subjects[0], task = "rest")
    ts_paths_cobre_1ses = fmriprepped_cobre.get_ts_paths(fmriprepped_cobre.subjects[1], task = "rest")
    ts_paths_depression_no_ses = fmriprepped_depression.get_ts_paths(fmriprepped_depression.subjects[0], task = "rest")

    assert len(ts_paths_cobre_2ses) == 2
    assert len(ts_paths_cobre_1ses) == 1
    assert len(ts_paths_depression_no_ses) == 1

def test_clean_signal_shape():
    n_parcels = 1000
    fmriprepped_cobre = FmriPreppedDataSet(cobre)
    fmriprepped_depression = FmriPreppedDataSet(depression)

    clean_ts_cobre_2ses = np.asarray(fmriprepped_cobre.clean_signal(subject = fmriprepped_cobre.subjects[0], task = "rest"))
    clean_ts_cobre_1ses = np.asarray(fmriprepped_cobre.clean_signal(subject = fmriprepped_cobre.subjects[1], task = "rest"))
    clean_ts_depression_no_ses = np.asarray(fmriprepped_depression.clean_signal(subject = fmriprepped_depression.subjects[0], task = "rest"))

    assert clean_ts_cobre_2ses.shape[0] == 2
    assert clean_ts_cobre_2ses.shape[2] == n_parcels

    assert clean_ts_cobre_1ses.shape[0] == 1
    assert clean_ts_cobre_1ses.shape[2] == n_parcels

    assert clean_ts_depression_no_ses.shape[0] == 1
    assert clean_ts_depression_no_ses.shape[2] == n_parcels

def test_conn_matrix():
    fmriprepped_cobre = FmriPreppedDataSet(cobre)
    fmriprepped_depression = FmriPreppedDataSet(depression)

    conn_matrix_cobre_2ses = fmriprepped_cobre.get_conn_matrix(subject = fmriprepped_cobre.subjects[0], task = "rest", save = True)
    conn_matrix_depression_no_ses = fmriprepped_depression.get_conn_matrix(subject = fmriprepped_depression.subjects[0], task = "rest", save = True)

    path_conn_matrix_cobre_2ses = os.path.join(f'{fmriprepped_cobre.data_path}', 'clean_data', f'sub-{fmriprepped_cobre.subjects[0]}', 'func', f'conn-matrix-sub-{fmriprepped_cobre.subjects[0]}-rest-schaefer1000.npy')
    path_conn_matrix_depression_no_ses = os.path.join(f'{fmriprepped_depression.data_path}', 'clean_data', f'sub-{fmriprepped_depression.subjects[0]}', 'func', f'conn-matrix-sub-{fmriprepped_depression.subjects[0]}-rest-schaefer1000.npy')

    assert conn_matrix_cobre_2ses.shape[0] == 2
    assert conn_matrix_cobre_2ses.shape[1] == 1000
    assert conn_matrix_cobre_2ses.shape[2] == 1000

    assert conn_matrix_depression_no_ses.shape[0] == 1
    assert conn_matrix_depression_no_ses.shape[1] == 1000
    assert conn_matrix_depression_no_ses.shape[2] == 1000

    assert os.path.exists(path_conn_matrix_cobre_2ses)
    assert os.path.exists(path_conn_matrix_depression_no_ses)
