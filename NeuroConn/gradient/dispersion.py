from ..preprocessing.preprocessing import FmriPreppedDataSet, output_spaces
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
import os

def get_dispersion(data, subject, n_grads, n_neighbours, task, from_single_grads = True, save = True, save_to = None, parcellation = 'schaefer', n_parcels = 1000,from_aligned_grads = True):
    """
    Compute the dispersion of a single functional gradient or a combination of gradients for a given subject.
    For the function to work, the gradients must be computed and saved in the output directory of the dataset.

    Parameters
    ----------
    data : str or FmriPreppedDataSet
        The path to a BIDS dataset or a FmriPreppedDataSet object. If a path is given, the dataset must be preprocessed.
    subject : str
        Subject ID.
    n_grads : int
        The number of gradients to use or the order of the gradient.
    n_neighbours : int
        The number of nearest neighbors.
    from_single_grads : bool
        Whether to compute dispersion for a single gradient. If False, compute dispersion for a combination of gradients.
    task : str
        Task name.
    save : bool
        Whether to save the results. Default is True.
    save_to : str, optional
        The directory where to save the results. If None, use the default output directory. Default is None.
    parcellation : str, optional
        The name of the parcellation to use. Default is 'schaefer'.
    n_parcels : int, optional
        The number of parcels in the parcellation. Default is 1000.
    from_aligned_grads : bool, optional
        Whether to use aligned gradients. Default is True.

    Returns
    -------
    numpy.ndarray
        The dispersion of functional gradients for the given subject and task.
    """
    if isinstance(data, FmriPreppedDataSet):
        fmriprepped_data = data
    elif isinstance(data, str):
        fmriprepped_data = FmriPreppedDataSet(data)
    else:
        raise ValueError("data must be either a FmriPreppedDataSet object or a path.")
    if from_single_grads:
        prefix = 'sing'
    else:
        prefix = 'comb'
    func_output = os.path.join(fmriprepped_data.default_output_dir, f'sub-{subject}', 'func')
    if from_aligned_grads:
        grad_prefix = 'aligned'
    else:
        grad_prefix = ''

    gradients =[np.load(f'{func_output}/{i}') for i in os.listdir(func_output) if grad_prefix in i and 'grad' in i and task in i and f'{n_parcels}' and parcellation in i][0]
    if len(gradients.shape) == 2:
        print("Gradients have 2 dimensions. Expanding to 3.")
        gradients = np.expand_dims(gradients, axis = 0)
    assert gradients.shape[1] == n_parcels, f"Dim 2 should be {n_parcels}."
    if from_single_grads:
        gradients = gradients[:,:, n_grads - 1].T
    else:
        gradients = gradients[0,:, :n_grads + 1]

    ngbrs = NearestNeighbors(n_neighbors=n_neighbours, algorithm='brute').fit(gradients)
    distances, _ = ngbrs.kneighbors(gradients)
    subj_disp = distances.mean(axis = 1)
    if save:
        if save_to is None:
            save_to = func_output
            print(save_to)
        np.save(f'{save_to}/disp-{prefix}-{n_grads}grad-{n_neighbours}n-sub-{subject}_schaefer{n_parcels}.npy', subj_disp)
    return subj_disp





