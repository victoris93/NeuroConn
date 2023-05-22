# NeuroConn

NeuroConn is a Python package that provides a user-friendly interface for fMRI preprocessing and computing connectivity matrices and gradients. It is designed as a BIDS application, allowing easy integration with BIDS-formatted datasets.
Documentation: https://victoris93.github.io/NeuroConn/

## Features

<font color="red">**NB! If you wish to run `fmriprep` within this package, install Docker Desktop first. Keep it running when you start** `RawDataset.docker_fmriprep()`</font>
- Preprocessing of fMRI data using the fmriprep pipeline
- Computation of connectivity matrices and gradients
- Direct output of gradients or connectivity matrices for any subject without specifying preprocessing parameters
- Handling of BIDS-formatted datasets

## Installation

You can install NeuroConn using pip: `pip install NeuroConn`

## Usage
**1. fMRIPrep**. The class `RawDataset` features a method to run fmriprep within within your Python environment. Before running it:
1. Register with freesurfer and download the license file `freesurfer_license.txt`)
2. Install Docker Desktop.
3. After having activated your environment, run `pip install fmriprep-docker`.
4. Start Docker Desktop.
Then, give this a try:

```
from NeuroConn.preprocessing.preprocessing import RawDataset, FmriPreppedDataSet
from NeuroConn.data.example_datasets import fetch_example_data
ex_data = fetch_example_data() # from https://openneuro.org/datasets/ds002748
data = RawDataset(ex_data)
subject = '52'
data.docker_fmriprep(subject, fs_reconall = False, fs_license = <path_to_freesurfer_license.txt>)
```

**2. Post-fMRIPrep** Here's an example of how to use the `FmriPreppedDataSet` class provided by NeuroConn:

```
from NeuroConn.preprocessing.preprocessing import RawDataset, FmriPreppedDataSet
from NeuroConn.data.example_datasets import fetch_example_data

# Download the dataset preprocessed with fMRIPrep
example_data = fetch_example_data('https://drive.google.com/file/d/1XjF5wDJXHzMyfoAjQE6NW2xcj9PulZzH/view?usp=share_link') 
# Initialize the dataset object 
dataset = FmriPreppedDataSet(example_data)

# Compute connectivity matrix
conn_matrix = data_prepped.get_conn_matrix(subject, parcellation='schaefer', task='rest', n_parcels=1000, save = True)

# Compute 10 gradients (Margulies et al., 2016)
gradients = get_gradients(data_prepped,subject, task='rest', n_components = 10, approach = "pca")
```

For more detailed information and examples, please refer to the [notebook](https://github.com/victoris93/PyConn/blob/master/NeuroConn.ipynb).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on this GitHub repository.

## License

NeuroConn is released under the MIT License. See the LICENSE file for more details.

## Example Data

Bezmaternykh D.D., Melnikov M.Y., Savelov A.A. et al. Brain Networks Connectivity in Mild to Moderate Depression: Resting State fMRI Study with Implications to Nonpharmacological Treatment. Neural Plasticity, 2021. V. 2021. № 8846097. PP. 1-15. DOI: 10.1155/2021/8846097


