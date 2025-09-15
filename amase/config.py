"""
Configuration module for AMASE.
Contains all constants, default parameters, and global settings.
"""

import os
import scipy.constants

# Algorithm Parameters
MAX_MOLS = 50  # Maximum number of possible candidates for a single line
DEFAULT_SIGMA = 6  # Default sigma threshold for peak detection
DEFAULT_TEMPERATURE = 10.0  # Default temperature in Kelvin

# Scoring Thresholds
SCORE_THRESHOLD = 0.7
GLOBAL_THRESHOLD = 93
GLOBAL_THRESHOLD_ORIGINAL = 93
GLOBAL_THRESHOLD_HIGH_ITER = 99  # After iteration 100
HIGH_ITER_CUTOFF = 100

# Graph and Structural Analysis
DISTANCE_THRESHOLD = 11  # Euclidean distance threshold for graph connections
OVERRIDE_THRESHOLD = 3  # Number of times needed to override scoring

# Spectroscopic Constants
SPEED_OF_LIGHT_KMS = 299792.458  # km/s
FREQUENCY_TOLERANCE = 0.5  # MHz
PEAK_WINDOW_DEFAULT = 1.0  # MHz for peak analysis

# Intensity and Line Analysis
MIN_LINE_PRESENCE_RATIO = 0.5  # Minimum ratio of predicted lines that must be present
MIN_LINE_PRESENCE_RATIO_LOW = 0.3  # Lower threshold for fewer lines
TEN_SIGMA_THRESHOLD = 10  # Multiple of RMS for strong line detection
FIVE_SIGMA_THRESHOLD = 5   # Multiple of RMS for weaker line detection

# Valid Atoms (default set)
DEFAULT_VALID_ATOMS = ['C', 'O', 'H', 'N', 'S']

# All possible atoms for when user selects "all"
ALL_VALID_ATOMS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
    'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
    'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
    'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
    'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Isotopologue identifiers for hasIso function
ISOTOPOLOGUE_LIST = [
    '17O', '(17)O', '18O', '(18)O', 'O18', '37Cl', '(37)Cl', 'Cl37', 
    '15N', '(15)N', 'N15', 'D', '(13)C', '13C', 'C13', '(50)Ti', 
    '50Ti', 'Ti50', '33S', '(33)S', 'S33', '34S', '(34)S', 'S34',
    '36S', '(36)S', 'S36', '29Si', '(29)Si', 'Si29'
]

# Molecules to exclude from Splatalogue (invalid/uncertain SMILES)
INVALID_MOLECULES = [
    'Manganese monoxide', 'Bromine Dioxide', 'Magnesium Isocyanide', 
    'Chromium monochloride', 'Scandium monosulfide', 'Hydrochloric acid cation',
    '3-Silanetetrayl-1,2-Propadienylidene', 'UNIDENTIFIED', 'Selenium Dioxide',
    '2-isocyano-3-propynenitrile', 'Aluminum cyanoacetylide', 
    'Silylidynyl cyanomethylene', 'Yttrium monosulfide', 'Chloryl chloride',
    '3-Silanetetrayl-1,2-Propadienylidene ', 'Calcium monochloride',
    'Nickel monocarbonyl', 'Scandium monochloride', 
    'Potassium cyanide, potassium isocyanide', 'Silicon Tetracarbide',
    'Calcium monoisocyanide', 'Iron Monocarbonyl', 'Calcium Monomethyl',
    'Bromine Monoxide', 'Cobalt carbide', 'Hypobromous acid',
    'Aluminum Isocyanitrile'
]

# File and Directory Names
DEFAULT_DIRECTORIES = {
    'local_catalogs': 'local_catalogs',
    'splatalogue_catalogs': 'splatalogue_catalogues', 
    'splat_cat_subdir': 'catalogues',
    'added_catalogs': 'added_catalogs',
    'cdms_pkl': 'cdms_pkl_final',
    'jpl_pkl': 'jpl_pkl_final'
}

DEFAULT_FILENAMES = {
    'dataset': 'dataset.csv',
    'dataset_intermediate': 'dataset_intermediate.csv',
    'dataset_final': 'dataset_final.csv',
    'output_report': 'output_report.txt',
    'edges': 'edges.csv',
    'all_smiles': 'all_smiles.csv',
    'counts': 'counts.csv',
    'all_vectors': 'all_vectors.csv',
    'mol_smiles': 'mol_smiles.csv',
    'combined_list': 'combined_list.pkl',
    'testing_list': 'testing_list.pkl',
    'final_peak_results': 'final_peak_results.csv',
    'u_line_candidates': 'u_line_candidates.csv',
    'u_line_candidates_non_charged': 'u_line_candidates_non_charged.csv',
    'mol2vec_model': 'mol2vec_model_final_70.pkl',
    'transitions_database': 'transitions_database.pkl.gz',
    'all_splat_smiles': 'all_splat_smiles.csv',
    'cdms_catalogs': 'cdms_catalogs.csv',
    'jpl_catalogs': 'jpl_catalogs.csv',
    'all_cdms_final': 'all_cdms_final_official.csv',
    'all_jpl_final': 'all_jpl_final_official.csv'
}

# Progress Bar Settings
PROGRESS_BAR_LENGTH = 50
PROGRESS_BAR_FILL = '█'
PROGRESS_BAR_EMPTY = '-'

# Molsim Integration Settings
MOLSIM_CONTINUUM_TYPE = 'thermal'
MOLSIM_CONTINUUM_PARAMS = 0.0
DEFAULT_COLUMN_DENSITY = 1.E10
DEFAULT_COLUMN_DENSITY_ALT = 1.E9

# Numerical Constants
NEAR_WHOLE_TOLERANCE = 0.05  # For near_whole function
CONVERGENCE_TOLERANCE = 1e-7  # For graph ranking convergence
MAX_ITERATIONS = 5000  # Maximum iterations for graph ranking

# Intensity Scaling Factors
INTENSITY_SCALE_FACTORS = {
    'max_observed_multiple': 6,
    'relative_intensity_multiple': 5,
    'isotopologue_strength_factor': 0.1
}

# Default Astronomical Parameters (used if astronomical observation)
DEFAULT_ASTRO_PARAMS = {
    'dish_size': 100,  # meters
    'source_size': 1.E20,  # arcseconds  
    'vlsr': 0.0,  # km/s
    'dv': 1.0,  # km/s
    'resolution': 0.0014  # MHz
}

def get_directory_path(base_path: str, dir_name: str) -> str:
    """Get full path for a directory."""
    return os.path.join(base_path, DEFAULT_DIRECTORIES[dir_name])

def get_file_path(base_path: str, filename_key: str) -> str:
    """Get full path for a file.""" 
    return os.path.join(base_path, DEFAULT_FILENAMES[filename_key])

def validate_base_directory(base_path: str) -> bool:
    """Validate that base directory exists and is writable."""
    return os.path.exists(base_path) and os.access(base_path, os.W_OK)