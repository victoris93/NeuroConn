import nilearn
import numpy as np
import pandas as pd
import os
import nibabel as nib
import json
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from nilearn import signal
from sklearn.impute import SimpleImputer


def z_transform_conn_matrix(conn_matrix):
    conn_matrix = np.arctanh(conn_matrix) # Fisher's z transform
    if np.isnan(conn_matrix).any(): # remove nans and infs in the matrix
        nan_indices = np.where(np.isnan(conn_matrix))
        conn_matrix[nan_indices] = .0000000001
    if np.isinf(conn_matrix).any():
        inf_indices = np.where(np.isinf(conn_matrix))
        conn_matrix[inf_indices] = 1
    return conn_matrix

class RawDataset():

    def __init__(self, BIDS_path):
        self.BIDS_path = BIDS_path
        if self.BIDS_path is not None:
            pass
        else:
            raise ValueError("The path to the dataset in BIDS format must be specified (BIDS_path).")
        self.data_description_path = self.BIDS_path + '/dataset_description.json'
        self.participant_data_path = self.BIDS_path + '/participants.tsv'
        self._participant_data = pd.read_csv(self.participant_data_path, sep = '\t')
        self._name = None
        self._data_description = None
        self._subjects = None

    @property
    def participant_data(self):
        if self._participant_data is None:
            self._participant_data = pd.read_csv(self.participant_data_path, sep = '\t')
        return self._participant_data

    @property
    def subjects(self):
        if self._subjects is None:
            self._subjects = self._participant_data['participant_id'].values
            self._subjects = np.array([i.replace('sub-', '') for i in self._subjects])
        return self._subjects
    
    @property
    def data_description(self):
        if self._data_description is None:
            self._data_description = json.load(open(self.data_description_path))
        return self._data_description

    @property
    def name(self):
        if self._name is None:
            self._name = self.data_description['Name']
        return self._name
    
    def __repr__(self):
        return f'Dataset(Name={self.name},\nSubjects={self.subjects},\nData_Path={self.BIDS_path})'



class FmriPreppedDataSet(RawDataset):

    def __init__(self, BIDS_path):
        super().__init__(BIDS_path)
        self.data_path = self.BIDS_path + '/derivatives'
        self.data_path = self._find_sub_dirs()
        self.default_confounds_path = os.path.join(os.path.dirname(__file__), "default_confounds.txt")
        self.subject_conn_paths = {}
        for subject in self.subjects:
            output_dir =os.path.join(self.data_path,'clean_data', f'sub-{subject}', 'func')
            if os.path.exists(output_dir):
                conn_mat_paths = [f'{output_dir}/{i}' for i in os.listdir(output_dir) if "conn-matrix" in i][0]
                self.subject_conn_paths[subject] = conn_mat_paths
    def __repr__(self):
        return f'Subjects={self.subjects},\n Data_Path={self.data_path})'
    
    def _find_sub_dirs(self):
        """Finds the sub-directories in the derivatives folder if they exist."""
        path_not_found = True
        while path_not_found:
            subdirs = os.listdir(self.data_path)
            for subdir in subdirs:
                if any(subdir.startswith('sub-') for subdir in subdirs):
                        path_not_found = False
                else:
                    if os.path.isdir(os.path.join(self.data_path, subdir)):
                        self.data_path = os.path.join(self.data_path, subdir)
        return self.data_path
    
    def get_ts_paths(self, subject, task): # needs to be adaptred to multiple sessions
        #numpy-style docstring
        """
        Parameters
        ----------
        subject : str
            The subject ID.
        task : str
            The task name.
        Returns
        -------
        ts_paths : list
            A list of paths to the time series files.
        """
        
        subject_dir = os.path.join(self.data_path, f'sub-{subject}')
        session_names = self.get_sessions(subject)
        ts_paths = []
        if len(session_names) != 0:
            for session_name in session_names:
                session_dir = os.path.join(subject_dir, f'ses-{session_name}', 'func')
                if os.path.exists(session_dir):
                    ts_paths.extend([f'{session_dir}/{i}' for i in os.listdir(session_dir) if task in i and i.endswith('MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')])
        else:
            subject_dir = os.path.join(subject_dir, 'func')
            ts_paths = [f'{subject_dir}/{i}' for i in os.listdir(subject_dir) if task in i and i.endswith('MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz')] #sub-01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
        return ts_paths
    
    def get_sessions(self, subject):
        subject_dir = f'{self.data_path}/sub-{subject}'
        subdirs = os.listdir(subject_dir)
        session_names = []
        for subdir in subdirs:
            if subdir.startswith('ses-'):
                session_names.append(subdir[4:])
        return session_names
    
    def _impute_nans_confounds(self, dataframe, pick_confounds = None):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            The dataframe containing the confounds.
        pick_confounds : list or numpy.ndarray
            The confounds to be picked from the dataframe.
        Returns
        -------
        df_no_nans : pandas.DataFrame
            The dataframe with the confounds without NaNs.
        """
        imputer = SimpleImputer(strategy='mean')
        if pick_confounds is None:
            pick_confounds = np.loadtxt(self.default_confounds_path, dtype = 'str')
        if isinstance(pick_confounds, (list, np.ndarray)):
            df_no_nans = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)[pick_confounds]
        else:
            df_no_nans = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
        return df_no_nans
    
    def get_confounds(self, subject, task, no_nans = True, pick_confounds = None):
        """
        Parameters
        ----------
        subject : str
            The subject ID.
        task : str
            The task name.
        no_nans : bool
            Whether to impute NaNs in the confounds.
        pick_confounds : list or numpy.ndarray
            The confounds to be picked from the dataframe.
        Returns
        -------
        confound_list : list
            A list of confounds.
        """
        if pick_confounds == None:
            pick_confounds = np.loadtxt(self.default_confounds_path, dtype = 'str')
        else:
            pick_confounds = np.loadtxt(pick_confounds, dtype = 'str')
        subject_dir = os.path.join(self.data_path, f'sub-{subject}')
        session_names = self.get_sessions(subject)

        if len(session_names) != 0:
            confound_paths = []
            confound_list = []
            for session_name in session_names:
                session_dir = os.path.join(subject_dir, f'ses-{session_name}', 'func')
                if os.path.exists(session_dir):
                    confound_files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if task in f and f.endswith('confounds_timeseries.tsv')]
                    confound_paths.extend(confound_files)
                    
            if no_nans == True:
                for confounds_path in confound_paths:
                    confounds = pd.read_csv(confounds_path, sep = '\t')
                    confounds = self._impute_nans_confounds(confounds)
                    confound_list.append(confounds)
            else:
                for confounds_path in confound_paths:
                    confounds = pd.read_csv(confounds_path, sep = '\t')[pick_confounds]
                    confound_list.append(confounds)
        else:
            func_dir = os.path.join(subject_dir, "func")
            confound_files = [os.path.join(func_dir, f) for f in os.listdir(func_dir) if task in f and f.endswith('confounds_timeseries.tsv')]
            if no_nans == True:
                confound_list = [self._impute_nans_confounds(pd.read_csv(i, sep = '\t'), pick_confounds) for i in confound_files]
            else:
                confound_list = [pd.read_csv(i, sep = '\t') for i in confound_files]

        return confound_list
    
    def parcellate(self, subject, parcellation = 'schaefer',task ="rest", n_parcels = 1000, gsr = False): # adapt to multiple sessions
        """
        Parameters
        ----------
        subject : str
            subject id
        parcellation : str
            parcellation to use
        task : str
            task to use
        n_parcels : int
            number of parcels to use
        gsr : bool  
            whether to use global signal regression
        Returns
        -------
        parc_ts_list : list
            list of parcellated time series
        """
        atlas = None
        if parcellation == 'schaefer':
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_parcels, yeo_networks=7, resolution_mm=1, base_url= None, resume=True, verbose=1)
        masker =  NiftiLabelsMasker(labels_img=atlas.maps, standardize=True, memory='nilearn_cache', verbose=5)

        parc_ts_list = []
        subject_ts_paths = self.get_ts_paths(subject, task)
        confounds = self.get_confounds(subject, task)
        for subject_ts, subject_confounds in zip(subject_ts_paths, confounds):
            if gsr == False:
                parc_ts = masker.fit_transform(subject_ts, confounds = subject_confounds.drop("global_signal", axis = 1))
                parc_ts_list.append(parc_ts)
            else:
                parc_ts = masker.fit_transform(subject_ts, confounds = subject_confounds)
                parc_ts_list.append(parc_ts)
        return parc_ts_list
    
    def clean_signal(self, subject, task="rest", parcellation='schaefer', n_parcels=1000, gsr=False, save = False, save_to = None): # add a save option + path
        """
        Parameters
        ----------
        subject : str
            subject id
        task : str
            task to use
        parcellation : str
            parcellation to use
        n_parcels : int
            number of parcels to use
        gsr : bool
            whether to use global signal regression
        save : bool
            whether to save the cleaned time series
        save_to : str   
            path to save the cleaned time series
        Returns
        -------
        clean_ts_array : np.array 
            cleaned time series
        """
        parc_ts_list = self.parcellate(subject, parcellation, task, n_parcels, gsr)
        clean_ts_array =[]
        for parc_ts in parc_ts_list:
            clean_ts = signal.clean(parc_ts, t_r = 2, low_pass=0.08, high_pass=0.01, standardize=True, detrend=True)
            clean_ts_array.append(clean_ts[10:]) # discarding first 10 volumes
        clean_ts_array = np.array(clean_ts_array)
        if save == True:
            if save_to is None:
                save_dir = os.path.join(f'{self.data_path}', 'clean_data', f'sub-{subject}', 'func')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_to = os.path.join(f'{self.data_path}/sub-{subject}', 'func', f'clean-ts-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
            else:
                save_to = os.path.join(save_to, f'clean-ts-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
            np.save(save_to, clean_ts_array)
        return clean_ts_array
    
    def get_conn_matrix(self, subject, subject_ts = None, parcellation = 'schaefer', task = 'rest', concat_ts = False, n_parcels = 1000, gsr = False, z_transformed = True, save = False, save_to = None):
        """
        Parameters
        ----------
        subject : str
            subject id
        subject_ts : str
            path to the cleaned time series
        parcellation : str
            parcellation to use
        task : str  
            task to use
        concat_ts : bool
            whether to compute the connectivity matrix on concatenated time series (e.g., if several sessions available)
        n_parcels : int
            number of parcels to use
        gsr : bool
            whether to use global signal regression
        z_transformed : bool
            whether to z transform the connectivity matrix
        save : bool
            whether to save the connectivity matrix
        save_to : str
            path to save the connectivity matrix
        Returns
        -------
        conn_matrix : np.array  
            connectivity matrix of shape (n_sessions, n_parcels, n_parcels)
        """
        if subject_ts is None:
            subj_ts_array = self.clean_signal(subject, task, parcellation, n_parcels, gsr)
        else:
            subj_ts_array = np.load(subject_ts)
        if concat_ts == True:
            subj_ts_array = np.row_stack(subj_ts_array)
            conn_matrix = np.corrcoef(subj_ts_array.T)
            if z_transformed == True:
                conn_matrix = z_transform_conn_matrix(conn_matrix)
        else:
            conn_matrix = np.zeros((subj_ts_array.shape[0], n_parcels, n_parcels))
            for i, subj_ts in enumerate(subj_ts_array):
                conn_matrix[i] = np.corrcoef(subj_ts.T)
                if z_transformed == True:
                    conn_matrix[i] = z_transform_conn_matrix(conn_matrix[i])
        if save == True:
            if save_to is None:
                save_dir = os.path.join(f'{self.data_path}', 'clean_data', f'sub-{subject}', 'func')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_to = os.path.join(save_dir, f'conn-matrix-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
            else:
                save_to = os.path.join(save_to, f'conn-matrix-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')

            self.subject_conn_paths[subject] = save_to

            np.save(save_to, conn_matrix)
        return conn_matrix