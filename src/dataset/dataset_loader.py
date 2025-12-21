"""
Get raw dataset for protein cleavage site classification.
"""
import requests
import os
import numpy as np


class DatasetLoader:
    @staticmethod
    def load(save_path: str, url: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(save_path):
            DatasetLoader._download(url, save_path)

        with open(save_path, 'r') as f:
            all_lines = f.read().split('\n')

        split_index = all_lines[0]
        dna_values = all_lines[2:-1:2]
        dna_labels = all_lines[1:-1:2]

        return dna_values, np.asarray(dna_labels, dtype=int), split_index

    @staticmethod
    def _download(url: str, save_path: str):
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'w') as f:
            f.write(response.text)
