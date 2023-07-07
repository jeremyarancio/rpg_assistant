"""Config file for data extraction"""
from pathlib import Path
import os

REPO_DIR = Path(os.path.dirname(os.path.realpath(__file__))).parent

class FirebaseConfig:
  url = "https://datasets.mechanus.zhu.codes/fireball-anonymized-nov-28-2022-kfdjl.tar.gz"
  extraction_path = REPO_DIR / "data/datasets/firebase/"