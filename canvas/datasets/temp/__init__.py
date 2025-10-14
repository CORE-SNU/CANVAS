import os
import pathlib
from .dataset import Dataset
from .dataset_loader import load_dataset
from .dataset_loader import get_dataset_spec
from .dataset_loader import _load_background_image


ASSET_DIR = pathlib.Path(__file__).parent.parent.parent / 'assets'

ETH_UCY_DIR = ASSET_DIR / 'datasets' / 'eth-ucy'
SNU_ASRI_DIR = ASSET_DIR / 'datasets' / 'snu-asri'


NAMES_ETH_UCY = ['eth', 'hotel', 'zara1', 'zara2', 'univ']

PATHS_REGISTERED = {
    'eth': ETH_UCY_DIR / 'biwi_eth.npy',
    'hotel': ETH_UCY_DIR / 'biwi_hotel.npy',
    'zara1': ETH_UCY_DIR / 'crowds_zara01.npy',
    'zara2': ETH_UCY_DIR / 'crowds_zara02.npy',
    'univ': ETH_UCY_DIR / 'students003.npy',
    'snu-asri': SNU_ASRI_DIR / '0.npy'
}

RegisteredDatasets = {}

for name in NAMES_ETH_UCY:
    RegisteredDatasets[name] = Dataset(name=name, data_path=PATHS_REGISTERED[name], dt=0.4)

RegisteredDatasets['snu-asri'] = Dataset(name='snu-asri', data_path=PATHS_REGISTERED['snu-asri'], dt=0.1)