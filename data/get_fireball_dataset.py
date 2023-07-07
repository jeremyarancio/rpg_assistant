"""IMPORTANT NOTE:
Script is actually not used. Check README file for more info.
I keep it here for future reference.
"""
import requests
import logging
import tarfile
from tqdm import tqdm
import tempfile
from typing import Union
import io
import os
import shutil
import gzip
from pathlib import Path

from config import FirebaseConfig


LOGGER = logging.getLogger(__name__)


def extract_data(url: str, extraction_path: Path) -> None:
    """From the dataset url from Firebase github repo, download the data and extract from .tar.gz file.

    Args:
        url (str): link to the .tar.gz file
        extraction_path (Path): directory path of the extracted data
    """
    with tempfile.NamedTemporaryFile() as tmpfile:
        with requests.get(url, allow_redirects=True, stream=True) as response:
            LOGGER.info(f"Downloading data from {url}")
            write_data_into_file(file=tmpfile, response=response)
            LOGGER.info(f"Extracting data into {extraction_path}")
            extract_from_tar_gz(tar_gz_file=tmpfile.name, extraction_path=extraction_path)
    LOGGER.info(f"Converting all .gz files into .jsonl files")
    convert_all_gz_files(dirpath=extraction_path)
    LOGGER.info(f"Finished downloading data")


def write_data_into_file(file: io.BufferedRandom, response: requests.Response) -> None:
    file_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc="Downloading data"
    )
    for chunk in response.iter_content(chunk_size = 1028):  #chunk_size in bytes
        if chunk:
            file.write(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()


def extract_from_tar_gz(tar_gz_file: Union[str, Path], extraction_path: Path) -> None:    
    if not extraction_path.exists():
        extraction_path.mkdir(parents=True)
        assert extraction_path.exists()
    with tarfile.open(tar_gz_file, 'r') as tar:
        tar.extractall(path=extraction_path)


def convert_gz_file(gz_file: Path):
  assert gz_file.exists()
  with gzip.open(gz_file, 'rb') as f_in:
      with open(gz_file.with_suffix(''), 'wb') as f_out: # file.jsonl.gz 
          shutil.copyfileobj(f_in, f_out)
  os.remove(gz_file)
  assert not gz_file.exists()


def convert_all_gz_files(dirpath: Path) -> None:
  for path in dirpath.iterdir():
    if path.is_dir():
      for gz_file in path.iterdir():
        convert_gz_file(gz_file)
    else: 
      raise FileExistsError("There is a file while only folders are expected in the directory")


if __name__ == "__main__":

    extract_data(
       url=FirebaseConfig.url, 
       extraction_path=FirebaseConfig.extraction_path
    )