import os
import pathlib
from .dataset import Dataset
from .dataset_loader import load_dataset
from .dataset_loader import get_dataset_spec
from .dataset_loader import _load_background_image


ASSET_DIR = pathlib.Path(__file__).parent.parent.parent / 'assets'

# Root datasets
ETH_UCY_DIR = ASSET_DIR / 'datasets' / 'eth-ucy'
SNU_ASRI_DIR = ASSET_DIR / 'datasets' / 'snu-asri'

# Snippet roots (zara01)
SNIPPETS_ZARA01_DIR = ASSET_DIR / 'datasets' / 'snippets_zara01'
Z01_RAW_MEAN_DIR = SNIPPETS_ZARA01_DIR / 'raw_mean'
Z01_RAW_MEAN_NON_OOD_DIR = Z01_RAW_MEAN_DIR / 'non_ood'
Z01_RAW_MEAN_OOD_DIR = Z01_RAW_MEAN_DIR / 'ood'
Z01_RAW_QUANTILE_DIR = SNIPPETS_ZARA01_DIR / 'raw_quantile'
Z01_RAW_QUANTILE_NON_OOD_DIR = Z01_RAW_QUANTILE_DIR / 'non_ood'
Z01_RAW_QUANTILE_OOD_DIR = Z01_RAW_QUANTILE_DIR / 'ood'

# NEW: Snippet roots (zara01_20)
SNIPPETS_ZARA01_20_DIR = ASSET_DIR / 'datasets' / 'snippets_zara01_20'
Z01_20_RAW_MEAN_DIR = SNIPPETS_ZARA01_20_DIR / 'raw_mean'
Z01_20_RAW_MEAN_NON_OOD_DIR = Z01_20_RAW_MEAN_DIR / 'non_ood'
Z01_20_RAW_MEAN_OOD_DIR = Z01_20_RAW_MEAN_DIR / 'ood'
Z01_20_RAW_QUANTILE_DIR = SNIPPETS_ZARA01_20_DIR / 'raw_quantile'
Z01_20_RAW_QUANTILE_NON_OOD_DIR = Z01_20_RAW_QUANTILE_DIR / 'non_ood'
Z01_20_RAW_QUANTILE_OOD_DIR = Z01_20_RAW_QUANTILE_DIR / 'ood'

Z01_RAW_MEAN_STATS_DIR = SNIPPETS_ZARA01_DIR / 'raw_mean_stats'
Z01_RAW_MEAN_STATS_NON_OOD_DIR = Z01_RAW_MEAN_STATS_DIR / 'non_ood'
Z01_RAW_MEAN_STATS_OOD_DIR     = Z01_RAW_MEAN_STATS_DIR / 'ood'
# Original ETH/UCY names
NAMES_ETH_UCY = ['eth', 'hotel', 'zara1', 'zara2', 'univ']

# Path registry
PATHS_REGISTERED = {
    # ETH/UCY full datasets
    'eth':    ETH_UCY_DIR / 'biwi_eth.npy',
    'hotel':  ETH_UCY_DIR / 'biwi_hotel.npy',
    'zara1':  ETH_UCY_DIR / 'crowds_zara01.npy',
    'zara2':  ETH_UCY_DIR / 'crowds_zara02.npy',
    'univ':   ETH_UCY_DIR / 'students003.npy',

    # SNU-ASRI
    'snu-asri': SNU_ASRI_DIR / '0.npy',
"""
    # --- snippets_zara01: raw_mean (simplified keys, no timestep ranges) ---
    'zara01_raw_mean_non_ood_seg2': Z01_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg2_124-130.npy',
    'zara01_raw_mean_non_ood_seg3': Z01_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg3_231-316.npy',
    'zara01_raw_mean_non_ood_seg4': Z01_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg4_478-799.npy',

    'zara01_raw_mean_ood_seg1': Z01_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg1_18-123.npy',
    'zara01_raw_mean_ood_seg2': Z01_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg2_131-230.npy',
    'zara01_raw_mean_ood_seg3': Z01_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg3_317-477.npy',
    'zara01_raw_mean_ood_seg4': Z01_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg4_800-901.npy',

    # --- snippets_zara01: raw_quantile ---
    'zara01_raw_quantile_non_ood_seg2': Z01_RAW_QUANTILE_NON_OOD_DIR / 'crowds_zara01_raw_quantile_non_ood_seg2_126-812.npy',
    'zara01_raw_quantile_ood_seg1':     Z01_RAW_QUANTILE_OOD_DIR     / 'crowds_zara01_raw_quantile_ood_seg1_18-125.npy',
    'zara01_raw_quantile_ood_seg2':     Z01_RAW_QUANTILE_OOD_DIR     / 'crowds_zara01_raw_quantile_ood_seg2_813-901.npy',

    # =================== NEW: snippets_zara01_20 ===================

    # raw_mean / non_ood
    'zara01_20_raw_mean_non_ood_seg1': Z01_20_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg1_0-37.npy',
    'zara01_20_raw_mean_non_ood_seg2': Z01_20_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg2_104-150.npy',
    'zara01_20_raw_mean_non_ood_seg3': Z01_20_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg3_211-336.npy',
    'zara01_20_raw_mean_non_ood_seg4': Z01_20_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg4_458-819.npy',
    'zara01_20_raw_mean_non_ood_seg5': Z01_20_RAW_MEAN_NON_OOD_DIR / 'crowds_zara01_raw_mean_non_ood_seg5_901-901.npy',

    # raw_mean / ood
    'zara01_20_raw_mean_ood_seg1': Z01_20_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg1_38-103.npy',
    'zara01_20_raw_mean_ood_seg2': Z01_20_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg2_151-210.npy',
    'zara01_20_raw_mean_ood_seg3': Z01_20_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg3_337-457.npy',
    'zara01_20_raw_mean_ood_seg4': Z01_20_RAW_MEAN_OOD_DIR / 'crowds_zara01_raw_mean_ood_seg4_820-900.npy',

    # raw_quantile / non_ood
    'zara01_20_raw_quantile_non_ood_seg1': Z01_20_RAW_QUANTILE_NON_OOD_DIR / 'crowds_zara01_raw_quantile_non_ood_seg1_0-37.npy',
    'zara01_20_raw_quantile_non_ood_seg2': Z01_20_RAW_QUANTILE_NON_OOD_DIR / 'crowds_zara01_raw_quantile_non_ood_seg2_106-832.npy',
    'zara01_20_raw_quantile_non_ood_seg3': Z01_20_RAW_QUANTILE_NON_OOD_DIR / 'crowds_zara01_raw_quantile_non_ood_seg3_901-901.npy',

    # raw_quantile / ood
    'zara01_20_raw_quantile_ood_seg1': Z01_20_RAW_QUANTILE_OOD_DIR / 'crowds_zara01_raw_quantile_ood_seg1_38-105.npy',
    'zara01_20_raw_quantile_ood_seg2': Z01_20_RAW_QUANTILE_OOD_DIR / 'crowds_zara01_raw_quantile_ood_seg2_833-900.npy',
# --- snippets_zara01: raw_mean_stats (simplified keys, no timestep ranges) ---
'zara01_raw_mean_stats_non_ood_seg1': Z01_RAW_MEAN_STATS_NON_OOD_DIR / 'crowds_zara01_raw_mean_stats_non_ood_seg1_0-101.npy',
'zara01_raw_mean_stats_non_ood_seg2': Z01_RAW_MEAN_STATS_NON_OOD_DIR / 'crowds_zara01_raw_mean_stats_non_ood_seg2_211-901.npy',

'zara01_raw_mean_stats_ood_seg1':     Z01_RAW_MEAN_STATS_OOD_DIR     / 'crowds_zara01_raw_mean_stats_ood_seg1_102-210.npy',
"""
# (manifest for reference)
# Z01_RAW_MEAN_STATS_DIR / 'crowds_zara01_raw_mean_stats_manifest.json'

}

# Build registry with appropriate dt
RegisteredDatasets = {}
for name, path in PATHS_REGISTERED.items():
    dt = 0.1 if name == 'snu-asri' else 0.4   # ETH/UCY and all zara01 snippets sampled @ 0.4s
    RegisteredDatasets[name] = Dataset(name=name, data_path=path, dt=dt)
