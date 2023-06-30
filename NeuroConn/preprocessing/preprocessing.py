import nilearn
import numpy as np
import pandas as pd
import os
import nibabel as nib
import json
from nilearn import datasets
import time
from nilearn.maskers import NiftiLabelsMasker
from nilearn import signal
from sklearn.impute import SimpleImputer
import subprocess as sp
import platform
import glob

output_spaces = {
    "anat": "T1w",
    "MNI152NLin6Asym": "MNI152NLin6Asym",
    "MNI152NLin6Asym:res-2": "MNI152NLin6Asym_res-2",
    "MNI152NLin6Asym:res-1": "MNI152NLin6Asym_res-1",
    "MNI152NLin2009cAsym:res-native":"MNI152NLin2009cAsym_res-native",
    "MNI152NLin6Asym":"MNI152NLin6Asym",
    "MNI152NLin2009cAsym":"MNI152NLin2009cAsym",
    "fsaverage":"fsaverage",
    "MNI152NLin2009cAsym:res-2":"MNI152NLin2009cAsym_res-2"
}

def parse_path_windows_docker(path):
    r"""
    Parses a path in Windows format to a path in Docker format.

    Parameters
    ----------
    path : str
        The path to parse.
        On Windows, use a raw string literal (e.g. r'C:\path\to\file').

    Returns
    -------
    str
        The parsed path.

    Examples
    --------
    ```
    parse_path_windows_docker(r'C:\Users\User\Desktop\data')
    '/c/Users/User/Desktop/data'
    ```
    """
    
    path = path.replace('\\', '/')
    path = path.replace(':', '')
    if path[0] == '/':
        path = '/' + path[1].lower() + '/' + path[3:]
    else:
        path = '/' + path[0].lower() + '/' + path[2:]
    return path

def parse_fmriprep_command(data_path, fmriprep_path, fs_license_path, work_path, participant_label, nthreads, output_spaces, fs_recon_all, task, nipreps_wrapper,mem_mb,skip_bids_validation = True, sloppy = False, system = platform.system()):
    r"""
    Parses the arguments for the fmriprep docker command.

    Parameters
    ----------
    data_path : str
        The path to the data.
        On Windows, use a raw string literal (e.g. r'C:\path\to\file').
    fmriprep_path : str
        The path to the fmriprep output directory. By default, the fmriprep output directory is data_path/derivatives/fmriprep.
        On Windows, use a raw string literal (e.g. r'C:\path\to\file').
    fs_license_path : str
        The path to the freesurfer license.
        On Windows, use a raw string literal (e.g. r'C:\path\to\file').
    work_path : str
        The path to the working directory. By default, it is home directory (usually the user directory).
        On Windows, use a raw string literal (e.g. r'C:\path\to\file').
    participant_label : str
        The subject ID.
    skip_bids_validation : bool, optional
        Whether to perform BIDS validation. Default is True.
    nthreads : int
        The number of threads to use.
    output_spaces : str
        The output spaces.
    fs_recon_all : bool
        Whether to run freesurfer's recon-all.
    task : str
        The name of the task to use. Default is 'rest'. If None, all tasks are preprcessed.
    nipreps_wrapper : bool
        Whether to use niprep's wrapper.
    mem_mb : int, optional
        The amount of memory to allocate to the Docker container, in MB. Default is 5000.
    sloppy : bool, optional
        Whether to use a lower rendering power. Default is True.
    system : str
        The operating system system. By default, determined automatically with `platform.system()`.

    Returns
    -------
    str
        Parsed fmriprep command.
    """
    
    fs_recon_all = '--fs-no-reconall' if not fs_recon_all else ''
    skip_bids_validation = '--skip-bids-validation' if skip_bids_validation else ''
    task = '' if task == None else f'--task-id {task}'
    sloppy = '--sloppy' if sloppy else ''

    if not nipreps_wrapper:
        if system == 'Windows':
            data_path = parse_path_windows_docker(data_path)
            fmriprep_path = parse_path_windows_docker(fmriprep_path)
            fs_license_path = parse_path_windows_docker(fs_license_path)
            work_path = parse_path_windows_docker(work_path)
        fmriprep_command = f"""
            docker run -ti --rm -v {data_path}:/data:ro \
                -v {fmriprep_path}:/out \
                -v {work_path}:/work \
                -v {fs_license_path}:/license \
                nipreps/fmriprep /data /out \
                participant --participant-label {participant_label} \
                -w /work \
                {skip_bids_validation} \
                {fs_recon_all} \
                --fs-license-file /license \
                --mem_mb {mem_mb} \
                --output-spaces {output_spaces} \
                {sloppy} \
                --nthreads {nthreads}
            """
    else:
        export_fmriprep_path = '' if system == 'Windows' else 'export PATH=$HOME/.local/bin:$PATH'
        fmriprep_command = f"""
        {export_fmriprep_path}
        fmriprep-docker {data_path} {fmriprep_path} participant --participant-label {participant_label} {skip_bids_validation} --fs-license-file {fs_license_path} {fs_recon_all} {task} --stop-on-first-crash --mem_mb {mem_mb} --output-spaces {output_spaces} -w {work_path} --nthreads {nthreads} {sloppy}
        """
    print('Running fmriprep command: ', fmriprep_command)
    return fmriprep_command


def z_transform_conn_matrix(conn_matrix):
    """
    Applies Fisher's z transform to a connectivity matrix.

    Parameters
    ----------
    conn_matrix : numpy.ndarray
        The connectivity matrix to transform.

    Returns
    -------
    numpy.ndarray
        The transformed connectivity matrix.
    """
    conn_matrix = np.arctanh(conn_matrix) # Fisher's z transform
    if np.isnan(conn_matrix).any(): 
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

    def docker_fmriprep(self, subject, fs_license_path, nthreads, fs_recon_all = False, mem_mb = 5000, task = 'rest', nipreps_wrapper = True, output_spaces = 'MNI152NLin2009cAsym:res-2', skip_bids_validation = True, work_path = os.path.expanduser('~'), sloppy = False):

        r"""
        Runs the fMRIprep pipeline in a Docker container for a given subject.

        Parameters
        ----------
        subject : str
            The label of the participant to process.
        fs_license_path : str
            The path to the (full) FreeSurfer license file.
            On Windows, use a raw string literal (e.g. r'C:\path\to\file').
        nthreads : int
            The number of threads to use for processing.
        skip_bids_validation : bool, optional
            Whether to skip BIDS validation. Default is True.
        fs_recon_all : bool, optional
            Whether to run FreeSurfer's recon-all. Default is False.
        mem_mb : int, optional
            The amount of memory to allocate to the Docker container, in MB. Default is 5000.
        task : str, optional
            The name of the task to use. Default is 'rest'. If None, all tasks are preprocessed.
        nipreps_wrapper : bool, optional
            Whether to use the Nipype workflow wrapper. Default is True.
        output_spaces : str, optional
            The list of output spaces to resample anatomical and functional images to. Default is 'MNI152NLin2009cAsym:res-2'.
            See https://fmriprep.org/en/stable/outputs.html#outputs for a list of available output spaces.
        work_path : str, optional
            The path to the working directory. Default is the user's home directory.
            On Windows, use a raw string literal (e.g. r'C:\path\to\file').
        sloppy : bool, optional
            Whether to use a lower rendering power. Default is True.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the FreeSurfer license file is not found.

        See Also
        --------
        https://fmriprep.org/en/stable/usage.html#command-line-arguments
        https://fmriprep.org/en/stable/outputs.html#outputs
        """
        data_path = self.BIDS_path
        fmriprep_path = os.path.join(data_path, 'derivatives', 'fmriprep')
        if not os.path.exists(fmriprep_path):
            os.makedirs(fmriprep_path)
    
        fmrirep_command = parse_fmriprep_command(data_path, fmriprep_path, fs_license_path, work_path, subject, skip_bids_validation, nthreads, output_spaces, fs_recon_all, task, nipreps_wrapper, mem_mb, sloppy)

        log_dir = f"{data_path}/fmriprep_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = f"{log_dir}/fmriprep_logs_sub-{subject}.txt"
        with open(log_file, "w") as file:
            if platform.system() == "Windows":
                process = sp.Popen(fmrirep_command, shell = True, stdout=file, stderr=file, universal_newlines=True)
            else:
                process = sp.Popen(["bash", "-c", fmrirep_command], stdout=file, stderr=file, universal_newlines=True)

            while process.poll() is None:
                time.sleep(0.1)

        with open(log_file, "r") as file:
            print(file.read())

    def get_sessions(self, subject):
        """
        Returns a list of session names for a given subject. If the subject has no sessions, an empty list is returned.

        Parameters
        ----------
        subject : str
            The label of the subject to retrieve session names for.

        Returns
        -------
        list of str
            A list of session names for the given subject.
        """
        subject_dir = f'{self.BIDS_path}/sub-{subject}'
        subdirs = os.listdir(subject_dir)
        session_names = []
        for subdir in subdirs:
            if subdir.startswith('ses-'):
                session_names.append(subdir[4:])
        return session_names
    
    def get_ts_paths(self, subject, task, output_space = None): 
        """
        Parameters
        ----------
        subject : str
            The subject ID.
        task : str
            The ID of the task to preprocess. Default is 'rest'.

        Returns
        -------
        ts_paths : list
            A list of paths to the time series files.
        """
        if output_space == None:
            output_space = ''
        else:
            output_space = output_spaces[output_space]
        subject_dir = os.path.join(self.BIDS_path, f'sub-{subject}')
        session_names = self.get_sessions(subject)
        ts_paths = []
        if len(session_names) != 0:
            for session_name in session_names:
                session_dir = os.path.join(subject_dir, f'ses-{session_name}', 'func')
                if os.path.exists(session_dir):
                    ts_paths.extend([f'{session_dir}/{i}' for i in os.listdir(session_dir) if task in i and i.endswith('.nii.gz')])
        else:
            subject_dir = os.path.join(subject_dir, 'func')
            ts_paths = [f'{subject_dir}/{i}' for i in os.listdir(subject_dir) if task in i and i.endswith('.nii.gz')] 
        return ts_paths
    
    def _bold_tr(self, subject, task):
        bold_file_path = self.get_ts_paths(subject, task)[0]
        img = nib.load(bold_file_path)
        bold_tr = img.header.get_zooms()[-1]
        if bold_tr == 0:
            print("TR in img header is 0. Trying to find TR in json params.")
            try:
                bold_params_path = bold_file_path.replace('.nii.gz', '.json')
                bold_tr = json.load(open(bold_params_path))['RepetitionTime']
            except FileNotFoundError:
                print(f"No params found for {subject} and task {task}. Looking for group-level params.")
            try:
                bold_params_path = os.path.join(self.BIDS_path, f'task-{task}_bold.json')
                bold_tr = json.load(open(bold_params_path))['RepetitionTime']
            except FileNotFoundError:
                print(f"No BOLD params found. Returning None.")
                bold_tr = None
        return bold_tr
    
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
    def bold_params(self):
        scan_params = json.load(open(f'{self.BIDS_path}/', 'r'))
        # load json
        # return dict
        
        return self._subjects

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
        self.default_output_dir = os.path.join(self.data_path, 'clean_data')
        self.subject_conn_paths = {}
        for subject in self.subjects:
            sub_output_dir = os.path.join(self.default_output_dir, f'sub-{subject}', 'func')
            if os.path.exists(sub_output_dir):
                conn_mat_paths = [f'{sub_output_dir}/{i}' for i in os.listdir(sub_output_dir) if "conn-matrix" in i]
                if len(conn_mat_paths) != 0:
                    self.subject_conn_paths[subject] = conn_mat_paths[0]
    def __repr__(self):
        return f'Subjects={self.subjects},\n Data_Path={self.data_path}'

    
    def _find_sub_dirs(self):
        """
        Finds the subdirectory containing the subject data.

        Returns
        -------
        str
            The path to the subdirectory containing the subject data.
        """
        path_not_found = True
        while path_not_found:
            try:
                subdirs = os.listdir(self.data_path)
            except FileNotFoundError as e:
                if e.filename == self.data_path and e.strerror == 'No such file or directory':
                    raise FileNotFoundError("The data have not been preprocessed with fmriprep: no 'derivatives' directory found.")
                else:
                    raise e
            for subdir in subdirs:
                if any(subdir.startswith('sub-') for subdir in subdirs):
                        path_not_found = False
                else:
                    if os.path.isdir(os.path.join(self.data_path, subdir)):
                        self.data_path = os.path.join(self.data_path, subdir)
        return self.data_path
    
    def get_ts_paths(self, subject, task, output_space = None): 
        #numpy-style docstring
        """
        Parameters
        ----------
        subject : str
            The subject ID.
        task : str
            The ID of the task to preprocess. Default is 'rest'.

        Returns
        -------
        ts_paths : list
            A list of paths to the time series files.
        """
        if output_space == None:
            output_space = ''
        else:
            output_space = output_spaces[output_space]
        subject_dir = os.path.join(self.data_path, f'sub-{subject}')
        session_names = self.get_sessions(subject)
        ts_paths = []
        if len(session_names) != 0:
            for session_name in session_names:
                session_dir = os.path.join(subject_dir, f'ses-{session_name}', 'func')
                if os.path.exists(session_dir):
                    ts_paths.extend([f'{session_dir}/{i}' for i in os.listdir(session_dir) if task in i and i.endswith(f'{output_space}_desc-preproc_bold.nii.gz')])
        else:
            subject_dir = os.path.join(subject_dir, 'func')
            ts_paths = [f'{subject_dir}/{i}' for i in os.listdir(subject_dir) if task in i and i.endswith(f'{output_space}_desc-preproc_bold.nii.gz')] 
        return ts_paths
    
    def get_sessions(self, subject):
        """
        Returns a list of session names for a given subject. If the subject has no sessions, an empty list is returned.

        Parameters
        ----------
        subject : str
            The label of the subject to retrieve session names for.

        Returns
        -------
        list of str
            A list of session names for the given subject.
        """
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
        pick_confounds : list or numpy.ndarray, optional
            The confounds to be picked from the dataframe. If None, the default confounds will be used. Default is None.

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
    
    def _bold_tr(self, subject, task):
        raw_dataset = RawDataset(self.BIDS_path)
        try:
            bold_file_path = raw_dataset.get_ts_paths(subject, task)[0]
            img = nib.load(bold_file_path)
            bold_tr = img.header.get_zooms()[-1]
        except FileNotFoundError:
            print(f"No BOLD file found for {subject} and task {task}. Returning None.")
            bold_file_path = None
            bold_tr = None
        if bold_tr == None:
            print("TR in img header is 0. Trying to find TR in json params.")
            if bold_file_path != None:
                try:
                    bold_params_path = bold_file_path.replace('.nii.gz', '.json')
                    bold_tr = json.load(open(bold_params_path))['RepetitionTime']
                except FileNotFoundError:
                    print(f"No params found for {subject} and task {task}. Looking for group-level params.")
                    bold_tr = None
            if bold_tr == 0 or bold_tr == None:
                try:
                    bold_params_path = os.path.join(self.BIDS_path, f'task-{task}_bold.json')
                    bold_tr = json.load(open(bold_params_path))['RepetitionTime']
                except FileNotFoundError:
                    print(f"No BOLD params found. Returning None.")
                    bold_tr = None
        return bold_tr
    
    def get_confounds(self, subject, task, no_nans = True, pick_confounds = None):
        """
        Returns a list of confounds for a given subject and task.

        Parameters
        ----------
        subject : str
            The ID of the subject.
        task : str
            The name of the task to use. Default is 'rest'.
        no_nans : bool, optional
            Whether to impute NaNs in the confounds. Default is True.
        pick_confounds : list or numpy.ndarray, optional
            The confounds to be picked from the dataframe. If None, the default confounds will be used. Default is None.

        Returns
        -------
        list
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
    
    def parcellate(self, subject, parcellation = 'schaefer',task ="rest", n_parcels = 1000, gsr = False, output_space = None): # adapt to multiple sessions
        """
        Parameters
        ----------
        subject : str
            subject id
        parcellation : str
            parcellation to use
        task : str
            The name of the task to use. Default is 'rest'.
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
        subject_ts_paths = self.get_ts_paths(subject, task, output_space = output_space)
        confounds = self.get_confounds(subject, task)
        for subject_ts, subject_confounds in zip(subject_ts_paths, confounds):
            if gsr == False:
                parc_ts = masker.fit_transform(subject_ts, confounds = subject_confounds.drop("global_signal", axis = 1))
                parc_ts_list.append(parc_ts)
            else:
                parc_ts = masker.fit_transform(subject_ts, confounds = subject_confounds)
                parc_ts_list.append(parc_ts)
        return parc_ts_list
    
    def clean_signal(self, subject, task="rest", parcellation='schaefer', n_parcels=1000, gsr=False, save = False, save_to = None, output_space = None): # add a save option + path
        """
        Cleans the time series for a given subject using a specified parcellation.

        Parameters
        ----------
        subject : str
            The ID of the subject to clean the time series for.
        task : str, optional
            The name of the task to use. Default is 'rest'.
        parcellation : str, optional
            The name of the parcellation to use. Default is 'schaefer'.
        n_parcels : int, optional
            The number of parcels to use. Default is 1000.
        gsr : bool, optional
            Whether to use global signal regression. Default is False.
        save : bool, optional
            Whether to save the cleaned time series. Default is False.
        save_to : str, optional
            The path to save the cleaned time series. If None, the time series will be saved to the default directory. Default is None.

        Returns
        -------
        np.ndarray
            The cleaned time series of shape (n_sessions, n_parcels, n_volumes).
        """
        parc_ts_list = self.parcellate(subject, parcellation, task, n_parcels, gsr, output_space)
        clean_ts_array =[]
        bold_tr = self._bold_tr(subject, task)
        for parc_ts in parc_ts_list:
            clean_ts = signal.clean(parc_ts, t_r = bold_tr, low_pass=0.08, high_pass=0.01, standardize='zscore_sample', detrend=True)
            print("Shape of clean_ts: ", clean_ts.shape)
            clean_ts_array.append(clean_ts[10:]) # discarding first 10 volumes
        clean_ts_array = np.array(clean_ts_array)
        if save == True:
            if save_to is None:
                save_dir = os.path.join(f'{self.default_output_dir}', f'sub-{subject}', 'func')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_to = os.path.join(f'{save_dir}', f'clean-ts-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
            else:
                save_to = os.path.join(save_to, f'clean-ts-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
            np.save(save_to, clean_ts_array)
        return clean_ts_array
    
    def get_conn_matrix(self, subject, subject_ts = None, parcellation = 'schaefer', task = 'rest', concat_ts = False, n_parcels = 1000, gsr = False, z_transformed = True, save = False, save_to = None, output_space = None):
        """
        Computes the connectivity matrix for a given subject.

        Parameters
        ----------
        subject : str
            The ID of the subject to compute the connectivity matrix for.
        subject_ts : str, optional
            The path to the cleaned time series. If None, the time series will be cleaned using the `clean_signal` method. Default is None.
        parcellation : str, optional
            The name of the parcellation to use. Default is 'schaefer'.
        task : str, optional
            The name of the task to use. Default is 'rest'.
        concat_ts : bool, optional
            Whether to compute the connectivity matrix on concatenated time series (e.g., if several sessions available). Default is False.
        n_parcels : int, optional
            The number of parcels to use. Default is 1000.
        gsr : bool, optional
            Whether to use global signal regression. Default is False.
        z_transformed : bool, optional
            Whether to apply Fisher's z transform to the connectivity matrix. Default is True.
        save : bool, optional
            Whether to save the connectivity matrix. Default is False.
        save_to : str, optional
            The path to save the connectivity matrix. If None, the matrix will be saved to the default directory. Default is None.

        Returns
        -------
        np.ndarray
            The connectivity matrix of shape (n_sessions, n_parcels, n_parcels).
        """
        z_suffix = ''
        if subject_ts is None:
            subj_ts_array = self.clean_signal(subject, task, parcellation, n_parcels, gsr, output_space = output_space)
        else:
            subj_ts_array = np.load(subject_ts)
        if concat_ts == True:
            subj_ts_array = np.row_stack(subj_ts_array)
            conn_matrix = np.corrcoef(subj_ts_array.T)
            if z_transformed == True:
                conn_matrix = z_transform_conn_matrix(conn_matrix)
                z_suffix = 'z-'
        else:
            conn_matrix = np.zeros((subj_ts_array.shape[0], n_parcels, n_parcels))
            for i, subj_ts in enumerate(subj_ts_array):
                conn_matrix[i] = np.corrcoef(subj_ts.T)
                if z_transformed == True:
                    conn_matrix[i] = z_transform_conn_matrix(conn_matrix[i])
                    z_suffix = 'z-'
        if save == True:
            if save_to is None:
                save_dir = os.path.join(f'{self.default_output_dir}', f'sub-{subject}', 'func')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_to = os.path.join(save_dir, f'{z_suffix}conn-matrix-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
            else:
                save_to = os.path.join(save_to, f'{z_suffix}conn-matrix-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')

            self.subject_conn_paths[subject] = save_to

            np.save(save_to, conn_matrix)
        return conn_matrix