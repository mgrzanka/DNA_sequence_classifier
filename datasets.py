import os
import requests
import numpy as np


class DonorsDataset:
    URL = 'https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceDTrainKIS.dat'
    SAVE_DIR = 'datasets'
    FILENAME = 'spliceDTrainKIS.dat'
    FILE_PATH = os.path.join(SAVE_DIR, FILENAME)

    @staticmethod
    def load():
        os.makedirs(DonorsDataset.SAVE_DIR, exist_ok=True)
        if not os.path.exists(DonorsDataset.FILE_PATH):
            DonorsDataset._download()

        with open(DonorsDataset.FILE_PATH, 'r') as f:
            all_lines = f.read().split('\n')

        split_index = all_lines[0]
        dna_values = all_lines[2:-1:2]
        dna_labels = all_lines[1:-1:2]

        return dna_values, np.asarray(dna_labels, dtype=int), split_index

    @staticmethod
    def _download():
        response = requests.get(DonorsDataset.URL)
        response.raise_for_status()
        with open(DonorsDataset.FILE_PATH, 'w') as f:
            f.write(response.text)


class AcceptorsDataset:
    URL = 'https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceATrainKIS.dat'
    SAVE_DIR = 'datasets'
    FILENAME = 'spliceATrainKIS.dat'
    FILE_PATH = os.path.join(SAVE_DIR, FILENAME)

    @staticmethod
    def load():
        os.makedirs(AcceptorsDataset.SAVE_DIR, exist_ok=True)
        if not os.path.exists(AcceptorsDataset.FILE_PATH):
            AcceptorsDataset._download()

        with open(AcceptorsDataset.FILE_PATH, 'r') as f:
            all_lines = f.read().split('\n')

        split_index = all_lines[0]
        dna_values = all_lines[2:-1:2]
        dna_labels = all_lines[1:-1:2]

        return dna_values, np.asarray(dna_labels, dtype=int), split_index

    @staticmethod
    def _download():
        response = requests.get(AcceptorsDataset.URL)
        response.raise_for_status()
        with open(AcceptorsDataset.FILE_PATH, 'w') as f:
            f.write(response.text)
