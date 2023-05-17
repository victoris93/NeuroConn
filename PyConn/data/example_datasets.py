import os
import gdown
import zipfile

output = 'example_data'
url = 'https://drive.google.com/file/d/1ijrYstMmsjMmQcM9ThoVf_dMYl9GFcv6/view?usp=share_link'

def unzip_and_delete(file_path, output_dir = os.path.dirname(__file__)):
	with zipfile.ZipFile(file_path, 'r') as zip_ref:
			zip_ref.extractall(output_dir)
	os.system(f'rm {file_path}')

def fetch_example_data(gdrive_url = url, output_name = output):
	data_path = os.path.join(os.path.dirname(__file__), output)
	if os.path.exists(data_path):
		print("Data already downloaded.")
	else:
		print("Downloading data...")
		gdown.download(gdrive_url, data_path + '.zip', quiet=False, fuzzy = True)
		unzip_and_delete(data_path + '.zip')
		print("Data downloaded.")
	return data_path