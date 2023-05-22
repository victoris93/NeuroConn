import os
import gdown
import zipfile

output = 'example_data'
url = 'https://drive.google.com/file/d/1ijrYstMmsjMmQcM9ThoVf_dMYl9GFcv6/view?usp=share_link'

def unzip_and_delete(file_path, output_dir, dir_name = output):
	"""
    Extracts the contents of a zip file to a specified directory and deletes the zip file.

    Parameters
    ----------
	file_path : str
        The path to the zip file.
    dir_name : str, optional
        The name of the directory to extract the contents of the zip file to. Default is 'output'.
    output_dir : str, optional
        The directory to extract the contents of the zip file to. Default is the directory of the script file.

    Returns
    -------
    None
    """
	
	output_dir = os.path.join(output_dir, dir_name)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	print("Extracting data...")
	with zipfile.ZipFile(file_path, 'r') as zip_ref:
		zip_ref.extractall(output_dir)
	os.system(f'rm {file_path}')
	print("Data extracted.")

def fetch_example_data(gdrive_url = url, output_name = output, output_dir = os.path.dirname(__file__)):
	"""
	Downloads and extracts example data from a Google Drive URL.

	Parameters
	----------
	gdrive_url : str, optional
		The Google Drive URL to download the data from. Default is the URL defined in the function.
	output_name : str, optional
		The name of the output file. Default is the name defined in the function.

	Returns
	-------
	str
		The path to the downloaded data.
	"""
	data_path = os.path.join(output_dir, output_name)
	if os.path.exists(data_path):
		print("Data already downloaded.")
	else:
		print("Downloading data...")
		gdown.download(gdrive_url, data_path + '.zip', quiet=False, fuzzy = True)
		unzip_and_delete(data_path + '.zip', output_dir = output_dir)
		print("Data downloaded.")
	return data_path