# PyConn

PyConn is a Python package that provides a user-friendly interface for post-fmriprep preprocessing and computing connectivity matrices and gradients. It is designed as a BIDS application, allowing easy integration with BIDS-formatted datasets.

## Features

<font color="red">**NB! If you wish to run `fmriprep` within this package, install Docker Desktop first. Keep it running when you start** `RawDataset.docker_fmriprep()`</font>
- Preprocessing of fMRI data using the fmriprep pipeline
- Computation of connectivity matrices and gradients
- Direct output of gradients or connectivity matrices for any subject without specifying preprocessing parameters
- Handling of BIDS-formatted datasets

## Installation

You can install PyConn using pip: `pip install PyConn`

## Usage
**1. fMRIPrep**. The class `RawDataset` features a method to run fmriprep within within your Python environment. Before running it:
1. Register with freesurfer and download the license file `freesurfer_license.txt`)
2. Install Docker Desktop.
3. After having activated your environment, run `pip install fmriprep-docker`.
4. Start Docker Disktop.
Then, give this a try:

```
from PyConn.preprocessing.preprocessing import RawDataset, FmriPreppedDataSet
from PyConn.data.example_datasets import fetch_example_data
ex_data = fetch_example_data()
data = RawDataset(ex_data)
subject = '17017'
data.docker_fmriprep(subject, fs_reconall = False, fs_license = <path_to_freesurfer_license.txt>)
```

**2. Post-fMRIPrep** Here's an example of how to use the `FmriPreppedDataSet` class provided by PyConn:

```
from PyConn.preprocessing.preprocessing import RawDataset, FmriPreppedDataSet
from PyConn.data.example_datasets import fetch_example_data

# Initialize the dataset object
ex_data = fetch_example_data()
dataset = FmriPreppedDataSet(example_data)

# Access information about the dataset
print(dataset)

# Get the paths to the time series files for a subject and task
ts_paths = dataset.get_ts_paths(subject, task)

# Compute connectivity matrices for a subject
conn_matrix = dataset.get_conn_matrix(subject, subject_ts=ts_paths, parcellation='schaefer', task='rest', n_parcels=1000)

# Compute gradients for a subject
gradients = dataset.compute_gradients(subject, subject_ts=ts_paths, task='rest')
```

For more detailed information and examples, please refer to the notebook.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on this GitHub repository.

## License

PyConn is released under the MIT License. See the LICENSE file for more details.