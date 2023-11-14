import requests
import zipfile


def download_and_extract_data(subset_num, save_dir):
    assert subset_num in range(0, 10), "subset_num must be between 0 and 9"
    if subset_num < 7:
        url = f'https://zenodo.org/records/3723295/files/subset{subset_num}.zip?download=1'
    else:
        url = f'https://zenodo.org/records/4121926/files/subset{subset_num}.zip?download=1'
    r = requests.get(url, stream=True)
    save_path = save_dir + f'subset{subset_num}.zip'
    print(f'Downloading subset{subset_num} zip')
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    print(f'Extracting zip')
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
