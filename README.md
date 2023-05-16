# PyConn

PyConn is a Python package that provides a user-friendly interface for post-fmriprep preprocessing and computing connectivity matrices and gradients. It is designed as a BIDS application, allowing easy integration with BIDS-formatted datasets.

## Features

- Preprocessing of fMRI data using the fmriprep pipeline
- Computation of connectivity matrices and gradients
- Direct output of gradients or connectivity matrices for any subject
- Handling of BIDS-formatted datasets

## Installation

You can install PyConn using pip: `pip install pyconn`

## Usage

Here's an example of how to use the `FmriPreppedDataSet` class provided by PyConn:

```
from pyconn import FmriPreppedDataSet

# Initialize the dataset object
dataset = FmriPreppedDataSet(BIDS_path)

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

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

PyConn is released under the MIT License. See the LICENSE file for more details.