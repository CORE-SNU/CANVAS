import os
from .dataset import Dataset
from .dataset_loader import load_dataset
from .dataset_loader import get_dataset_spec
from .dataset_loader import _load_background_image



ETH_UCY_DIR = os.path.join(os.path.dirname(__file__), 'eth-ucy')
SNU_ASRI_DIR = os.path.join(os.path.dirname(__file__), 'snu-asri')


NAMES_ETH_UCY = ['eth', 'hotel', 'zara1', 'zara2', 'univ']

PATHS_REGISTERED = {
    'eth': os.path.join(ETH_UCY_DIR, 'biwi_eth.npy'),
    'hotel': os.path.join(ETH_UCY_DIR, 'biwi_hotel.npy'),
    'zara1': os.path.join(ETH_UCY_DIR, 'crowds_zara01.npy'),
    'zara2': os.path.join(ETH_UCY_DIR, 'crowds_zara02.npy'),
    'univ': os.path.join(ETH_UCY_DIR, 'students003.npy'),
    'snu-asri': os.path.join(SNU_ASRI_DIR, '0.npy')
}

RegisteredDatasets = {}

for name in NAMES_ETH_UCY:
    RegisteredDatasets[name] = Dataset(name=name, data_path=PATHS_REGISTERED[name], dt=0.4)

RegisteredDatasets['snu-asri'] = Dataset(name='snu-asri', data_path=PATHS_REGISTERED['snu-asri'], dt=0.1)