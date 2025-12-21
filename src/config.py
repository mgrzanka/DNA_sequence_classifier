from dataclasses import dataclass
from dotenv import load_dotenv
import os


if not os.path.exists('.env'):
    raise Exception("Check .env file path")
load_dotenv('.env')


@dataclass
class Config:
    dataset_path: str
    url: str
    regex_path: str


class AppConfig:
    DONORS = Config(
        dataset_path='data/raw/raw_donors',
        url='https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceDTrainKIS.dat',
        regex_path='data/proccessed/regex_donors'
    )

    ACCEPTORS = Config(
        dataset_path='data/raw/raw_acceptors',
        url='https://staff.elka.pw.edu.pl/~rbiedrzy/UMA/spliceATrainKIS.dat',
        regex_path='data/proccessed/regex_acceptors'
    )

    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    if not WANDB_API_KEY:
        raise Exception("No WANDB_API_KEY in .env file")
