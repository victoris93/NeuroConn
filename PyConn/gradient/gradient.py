import brainspace
import numpy as np
from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import ProcrustesAlignment
from brainspace.utils.parcellation import map_to_labels
from brainspace.datasets import load_parcellation
from ..preprocessing.preprocessing import FmriPreppedDataSet
import sys
import os

path_margulies_grads = os.path.join(os.path.dirname(__file__), 'margulies_grads_schaefer1000.npy')

def align_gradients(gradients, n_components, custom_ref = None, *args):
    """
    Aligns gradients to a reference set of gradients using Procrustes alignment.

    Parameters
    ----------
    gradients : str or numpy.ndarray
        The gradients to align.
    custom_ref : str or numpy.ndarray, optional
        The reference gradients to align to. If None, the default Margulies et al. (2016) gradients will be used. Default is None.
    *args :
        Additional arguments to pass to ProcrustesAlignment.

    Returns
    -------
    numpy.ndarray
        The aligned gradients.
    """
    if custom_ref is None:
        path_margulies_grads = os.path.join(os.path.dirname(__file__), 'margulies_grads_schaefer1000.npy')
        ref_gradients = np.load(path_margulies_grads)[:n_components]
    else:
        ref_gradients = np.load(custom_ref)
    if isinstance(gradients, str):
        gradients = np.load(gradients)
    if len(gradients.shape) == 2:
        gradients = np.expand_dims(gradients, axis = 0)
    Alignment = ProcrustesAlignment(*args)
    aligned_gradients = np.array(Alignment.fit(gradients, ref_gradients.T).aligned_)
    return aligned_gradients
    
def get_gradients(data, subject, n_components, task, parcellation = 'schaefer', n_parcels = 1000, kernel = 'cosine', approach = 'pca', from_mat = True, aligned = True, save = True, save_to = None):
    """
    Computes gradients from the subject connectivity matrix.

    Parameters
    ----------
    data : str or FmriPreppedDataSet
        The path to the data or the FmriPreppedDataSet object.
    subject : str
        The subject ID.
    n_components : int
        The number of components to extract.
    task : str, optional
        The task name. Default is 'rest'.
    parcellation : str, optional
        The parcellation name. Default is 'schaefer'.
    n_parcels : int, optional
        The number of parcels. Default is 1000.
    kernel : str, optional
        The kernel to use. Default is 'cosine'.
    approach : str, optional
        The approach to use. Default is 'pca'.
    from_mat : bool, optional
        Whether to load the data from a .mat file. Default is True.
    aligned : bool, optional
        Whether to align the gradients to the Margulies et al. (2016) gradients. Default is True.
    save : bool, optional  
        Whether to save the gradients. Default is True.
    save_to : str, optional 
        The path to save the gradients. Default is None.

    Returns
    -------
    numpy.ndarray
        The computed gradients.
    """
    gm = GradientMaps(n_components = n_components, kernel = kernel, approach = approach)
    if isinstance(data, FmriPreppedDataSet):
        fmriprepped_data = data
    elif isinstance(data, str):
        fmriprepped_data = FmriPreppedDataSet(data)
    else:
        raise ValueError("data must be either a FmriPreppedDataSet object or a string.")
    prefix = ''
    if from_mat:
        input_path = fmriprepped_data.subject_conn_paths[subject]
    input_data = np.load(input_path)
    if len(input_data.shape) == 3:
        gradients = []
        for i in input_data:
            gm.fit(i)
            gradients.append(gm.gradients_)
            gm = GradientMaps(n_components = n_components, kernel = kernel, approach = approach)
        gradients = np.asarray(gradients)
    elif len(input_data.shape) == 2:
        gm.fit(input_data)
        gradients = gm.gradients_
    if aligned:
        gradients = align_gradients(gradients, n_components)
        prefix = "aligned-"
    if save:
        if save_to is None:
            save_dir = os.path.join(f'{fmriprepped_data.data_path}', 'clean_data', f'sub-{subject}', 'func')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_to = os.path.join(save_dir, f'{prefix}{n_components}gradients-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
        else:
            save_to = os.path.join(save_to, f'{prefix}{n_components}gradients-sub-{subject}-{task}-{parcellation}{n_parcels}.npy')
        np.save(file = save_to, arr = gradients)

    return gradients