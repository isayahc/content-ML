import gdown
import zipfile
import os

# Download file from Google Drive
url = 'https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8'
zip_file = 'file_name.zip' # specify the desired file name and extension for the downloaded zip file
gdown.download(url, zip_file, quiet=False)

# Unzip file and store its contents in the models directory
models_dir = 'models' # specify the path to the models directory

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(models_dir)

print(f'Contents of {zip_file} extracted to {models_dir}')