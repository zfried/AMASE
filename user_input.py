"""
User input handling for AMASE.
Handles all user interactions, input validation, and parameter collection.
Supports both interactive mode and YAML configuration files.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
from rdkit import Chem
from typing import List, Tuple, Optional, Dict, Any
from config import DEFAULT_VALID_ATOMS, ALL_VALID_ATOMS


def load_parameters_from_yaml(yaml_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate parameters from a YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Dictionary of validated parameters, or None if validation fails
    """
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print()
        print("=" * 60)
        print("Loading Configuration from YAML")
        print("=" * 60)
        
        # Validate required fields
        required_fields = ['spectrum_path', 'directory_path', 'sigma_threshold', 'temperature']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            print(f"Error: Missing required fields in YAML: {', '.join(missing_fields)}")
            return None
        
        # Validate spectrum path
        if not os.path.isfile(config['spectrum_path']):
            print(f"Error: Spectrum file not found: {config['spectrum_path']}")
            return None
        if not config['spectrum_path'].lower().endswith('.txt'):
            print("Error: Spectrum file must be a .txt file")
            return None
        print(f"✓ Spectrum path: {config['spectrum_path']}")
        
        # Validate directory path
        directory_path = config['directory_path']
        if directory_path[-1] != '/':
            directory_path = directory_path + '/'
        if not (os.path.exists(directory_path) and os.access(directory_path, os.W_OK)):
            print(f"Error: Directory not found or not writable: {directory_path}")
            return None
        config['directory_path'] = directory_path
        print(f"✓ Directory path: {directory_path}")
        
        # Validate local catalogs
        local_enabled = config.get('local_catalogs_enabled', False)
        local_dir = None
        local_df = None
        
        print(f"✓ Local catalogs enabled: {local_enabled}")
        
        if local_enabled:
            local_dir = config.get('local_directory')
            if not local_dir:
                print("Error: local_directory must be specified when local_catalogs_enabled is true")
                return None
            if local_dir[-1] != '/':
                local_dir = local_dir + '/'
            if not os.path.exists(local_dir):
                print(f"Error: Local directory not found: {local_dir}")
                return None
            print(f"  - Local directory: {local_dir}")
            
            local_csv = config.get('local_csv_path')
            if not local_csv:
                print("Error: local_csv_path must be specified when local_catalogs_enabled is true")
                return None
            
            try:
                local_df = pd.read_csv(local_csv)
                required_columns = ['name', 'smiles', 'iso']
                if not all(col in local_df.columns for col in required_columns):
                    print(f"Error: Local CSV must contain columns: {required_columns}")
                    return None
                print(f"  - Local CSV: {local_csv} ({len(local_df)} molecules)")
            except Exception as e:
                print(f"Error reading local CSV file: {e}")
                return None
        
        config['local_directory'] = local_dir
        config['local_df'] = local_df
        
        # Validate sigma threshold
        if config['sigma_threshold'] <= 0:
            print("Error: sigma_threshold must be positive")
            return None
        print(f"✓ Sigma threshold: {config['sigma_threshold']}")
        
        # Validate temperature
        if config['temperature'] <= 0:
            print("Error: temperature must be positive")
            return None
        print(f"✓ Temperature: {config['temperature']} K")
        
        # Validate valid_atoms
        valid_atoms = config.get('valid_atoms', 'default')
        if valid_atoms == 'default':
            config['valid_atoms'] = DEFAULT_VALID_ATOMS
            print(f"✓ Valid atoms: default ({', '.join(DEFAULT_VALID_ATOMS)})")
        elif valid_atoms == 'all':
            config['valid_atoms'] = ALL_VALID_ATOMS
            print(f"✓ Valid atoms: all atoms in periodic table")
        elif isinstance(valid_atoms, str):
            # Handle comma-separated string
            atoms = [atom.strip() for atom in valid_atoms.split(',') if atom.strip()]
            invalid_atoms = [atom for atom in atoms if atom not in ALL_VALID_ATOMS]
            if invalid_atoms:
                print(f"Error: Invalid atoms in YAML: {', '.join(invalid_atoms)}")
                return None
            config['valid_atoms'] = atoms
            print(f"✓ Valid atoms: {', '.join(atoms)}")
        else:
            print("Error: valid_atoms must be 'default', 'all', or a comma-separated string of atom symbols")
            return None
        
        # Validate consider_structure
        config['consider_structure'] = config.get('consider_structure', True)
        print(f"✓ Consider structure: {config['consider_structure']}")
        
        # Validate starting_molecules
        starting_molecules = config.get('starting_molecules', '')
        if starting_molecules:
            # Handle comma-separated string only
            if isinstance(starting_molecules, str):
                smiles_list = [s.strip() for s in starting_molecules.split(',') if s.strip()]
            else:
                print("Error: starting_molecules must be a comma-separated string")
                return None
            
            validated_smiles = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    validated_smiles.append(Chem.MolToSmiles(mol))
                else:
                    print(f"Error: Invalid SMILES in YAML: {smiles}")
                    return None
            config['starting_molecules'] = validated_smiles
            if validated_smiles:
                print(f"✓ Starting molecules: {', '.join(validated_smiles)}")
            else:
                print(f"✓ Starting molecules: none")
        else:
            config['starting_molecules'] = []
            print(f"✓ Starting molecules: none")
        
        # Validate artifact_frequencies (optional)
        # artifact_frequencies = config.get('artifact_frequencies', [])
        # if not isinstance(artifact_frequencies, list):
        #     print("Error: artifact_frequencies must be a list")
        #     return None
        # config['artifact_frequencies'] = artifact_frequencies
        
        print("=" * 60)
        print("Configuration loaded successfully!")
        print("=" * 60)
        return config
        
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None


def get_spectrum_path() -> str:
    """
    Get and validate spectrum file path.
    
    The spectrum file must be a .txt file with:
    - First column: frequency in MHz
    - Second column: intensity
    - Tab or space separated values
    """
    while True:
        print("Spectrum file requirements:")
        print("- Must be a .txt file")
        print("- First column: frequency (MHz)")
        print("- Second column: intensity")
        print()
        
        path = input('Please enter full path to spectrum file:\n').strip()
        
        if not os.path.isfile(path):
            print('File not found. Please enter a valid file path.')
            print()
            continue
            
        if not path.lower().endswith('.txt'):
            print('File must be a .txt file.')
            print()
            continue
            
        return path


def get_directory_path() -> str:
    """Get and validate directory path for file storage."""
    while True:
        path = input('Please enter path to directory where required files are stored. This is also where the outputs will be saved:\n').strip()
        # Clean up path
        path = ''.join(path.split())
        if path[-1] != '/':
            path = path + '/'
        
        if os.path.exists(path) and os.access(path, os.W_OK):
            return path
        else:
            print('Directory not found or not writable. Please enter a valid directory path.')
            print()


def get_local_catalogs_info() -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Get local catalogs configuration."""
    while True:
        response = input('Do you have catalogs on your local computer that you would like to consider (y/n): \n').strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            return False, None, None
        else:
            print('Invalid input. Please just type y or n')
            print()

    # Get local directory
    while True:
        local_dir = input('Great! Please enter path to directory to your local spectroscopic catalogs:\n').strip()
        local_dir = ''.join(local_dir.split())
        if local_dir[-1] != '/':
            local_dir = local_dir + '/'
        
        if os.path.exists(local_dir):
            break
        else:
            print('Directory not found. Please enter a valid directory path.')
            print()

    # Get local DataFrame
    while True:
        try:
            df_path = input(
                'Please enter path to the csv file that contains the SMILES strings and isotopic composition of molecules in local catalogs:\n'
            ).strip()
            df = pd.read_csv(df_path)
            required_columns = ['name', 'smiles', 'iso']
            if all(col in df.columns for col in required_columns):
                return True, local_dir, df
            else:
                print(f'CSV file must contain columns: {required_columns}')
                print()
        except Exception as e:
            print(f'Error reading CSV file: {e}')
            print()


def get_sigma_threshold() -> int:
    """Get sigma threshold for line detection."""
    while True:
        try:
            sigma = float(input('Sigma threshold (the code will only attempt to assign lines greater than sigma*rms noise):\n'))
            if sigma > 0:
                return sigma
            else:
                print('Please enter a positive value.')
                print()
        except ValueError:
            print('Please enter a valid value.')
            print()


def get_temperature() -> float:
    """Get experimental temperature."""
    while True:
        try:
            temp = float(input('Please enter the experimental temperature (in Kelvin): \n'))
            if temp > 0:
                return temp
            else:
                print('Temperature must be positive.')
                print()
        except ValueError:
            print('Please enter a valid number.')
            print()


def get_valid_atoms() -> List[str]:
    """Get list of valid atoms that could be present."""
    
    while True:
        print("Which atoms could feasibly be present in the mixture?")
        print("1. Default (C, O, H, N, S)")
        print("2. All atoms in periodic table")
        print("3. Specify custom atoms")
        print()
        
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            return DEFAULT_VALID_ATOMS
        elif choice == '2':
            return ALL_VALID_ATOMS
        elif choice == '3':
            while True:
                atoms_input = input("Enter atoms separated by commas (e.g., C,O,S): ").strip()
                
                # Parse comma-separated atoms
                atoms = [atom.strip() for atom in atoms_input.split(',') if atom.strip()]
                
                if not atoms:
                    print("Please enter at least one atom.")
                    print()
                    continue
                
                # Check if all atoms are in ALL_VALID_ATOMS
                invalid_atoms = [atom for atom in atoms if atom not in ALL_VALID_ATOMS]
                
                if invalid_atoms:
                    print(f"Invalid atoms: {', '.join(invalid_atoms)}")
                    print(f"Please only use atoms from the valid list: {', '.join(ALL_VALID_ATOMS)}")
                    print()
                    continue
                
                return atoms
        else:
            print("Please enter 1, 2, or 3.")
            print()


def get_structure_consideration() -> bool:
    """Ask if user wants to consider structural relevance."""
    while True:
        response = input('Do you want to consider structural relevence? If not, only the spectroscopy will be considered (y/n): \n Note: the structural relevance program is only designed for small astrochemically relevant molecules at the moment. If the mixture may contain larger or unique molecules, please input n.\n').strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print('Invalid input. Please just type y or n')
            print()


def get_smiles_manual() -> List[str]:
    """Get SMILES strings manually from user input."""
    while True:
        try:
            smiles_input = input(
                'Enter the SMILES strings of the initial detected molecules. '
                'Please separate the SMILES string with a comma: \n'
            )
            smiles_list = [s.strip() for s in smiles_input.split(',') if s.strip()]
            
            # Validate SMILES
            validated_smiles = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    validated_smiles.append(Chem.MolToSmiles(mol))
                else:
                    print(f'Invalid SMILES: {smiles}')
                    raise ValueError('Invalid SMILES detected')
            
            return validated_smiles
            
        except Exception as e:
            print('You entered an invalid SMILES string. Please try again.')
            print()


def get_smiles_from_csv() -> List[str]:
    """Get SMILES strings from CSV file."""
    while True:
        try:
            csv_path = input(
                'Please enter path to csv file. This needs to have the detected molecules '
                'in a column listed "SMILES."\n'
            ).strip()
            
            df = pd.read_csv(csv_path)
            if 'SMILES' not in df.columns:
                print('CSV file must contain a "SMILES" column.')
                print()
                continue
            
            smiles_list = df['SMILES'].dropna().tolist()
            
            # Validate SMILES
            validated_smiles = []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    validated_smiles.append(Chem.MolToSmiles(mol))
                else:
                    print(f'Invalid SMILES in CSV: {smiles}')
                    raise ValueError('Invalid SMILES detected')
            
            return validated_smiles
            
        except Exception as e:
            print(f'Error reading CSV file: {e}')
            print()


def get_starting_molecules(consider_structure: bool) -> List[str]:
    """Get initial detected molecules if any."""
    if not consider_structure:
        return []
    
    # Ask if user has known precursors
    while True:
        response = input('Do you have any known molecular precursors? (y/n): \n').strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            return []
        else:
            print('Invalid input. Please just type y or n')
            print()

    # Get input method
    while True:
        method = input(
            'You need to input the SMILES strings of the initial detected molecules.\n'
            'If you would like to type them individually, type 1. If you would like to input a csv file, type 2: \n'
        ).strip()
        
        if method == '1':
            return get_smiles_manual()
        elif method == '2':
            return get_smiles_from_csv()
        else:
            print('Please enter 1 or 2.')
            print()


def get_artifact_frequencies() -> List[float]:
    """Get known artifact frequencies from user."""
    while True:
        response = input('Are there any known instrument artifact frequencies? (y/n)\n').strip().lower()
        if response in ['y', 'yes']:
            break
        elif response in ['n', 'no']:
            return []
        else:
            print('Invalid input. Please just type y or n')
            print()
    
    while True:
        try:
            artifacts_input = input(
                'OK great! Please type the artifact frequencies and separate them with commas.\n'
            )
            artifact_freqs = [float(f.strip()) for f in artifacts_input.split(',') if f.strip()]
            return artifact_freqs
        except ValueError:
            print('Please enter valid frequency values separated by commas.')
            print()


def validate_smiles_string(smiles_string: str) -> Optional[str]:
    """Validate and canonicalize a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        else:
            return None
    except:
        return None


def parse_command_line_args() -> Optional[str]:
    """
    Parse command line arguments to check for config file.
    
    Returns:
        Path to YAML config file if provided, None otherwise
    """
    parser = argparse.ArgumentParser(
        description='AMASE - Automated Molecular Assignment from Spectroscopic Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python3 amase.py
    
  With config file:
    python3 amase.py --config config.yaml
    python3 amase.py -c config.yaml
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file',
        default=None
    )
    
    args = parser.parse_args()
    return args.config


def collect_all_parameters() -> Dict[str, Any]:
    """Collect all user parameters either from YAML or interactively."""
    print('')
    
    # Check for command line config file argument
    yaml_path = parse_command_line_args()
    
    if yaml_path:
        # Config file provided via command line
        if not os.path.isfile(yaml_path):
            print(f"Error: Config file not found: {yaml_path}")
            sys.exit(1)
        
        print()
        parameters = load_parameters_from_yaml(yaml_path)
        if parameters is None:
            print()
            print('Failed to load YAML configuration.')
            sys.exit(1)
        else:
            print()
            print('Thanks for the input! Making the dataset now. This will take a few minutes.')
            return parameters
    
    # Interactive mode (no command line arguments)
    print('-------------------------')
    print('AMASE - Interactive Configuration Mode')
    print('-------------------------')
    print()
    print('No config file specified. Running in interactive mode.')
    print('(To use a config file, run: python3 amase.py --config config.yaml)')
    print()
    
    spectrum_path = get_spectrum_path()
    print('')
    directory_path = get_directory_path()
    print('')
    local_enabled, local_dir, local_df = get_local_catalogs_info()
    print('')
    sigma_threshold = get_sigma_threshold()
    print('')
    temperature = get_temperature()
    print('')
    valid_atoms = get_valid_atoms()
    print('')
    consider_structure = get_structure_consideration()
    print('')
    starting_molecules = get_starting_molecules(consider_structure)
    print('')
    
    parameters = {
        'spectrum_path': spectrum_path,
        'directory_path': directory_path,
        'local_catalogs_enabled': local_enabled,
        'local_directory': local_dir,
        'local_df': local_df,
        'sigma_threshold': sigma_threshold,
        'temperature': temperature,
        'valid_atoms': valid_atoms,
        'consider_structure': consider_structure,
        'starting_molecules': starting_molecules,
    }
    
    print()
    print('Thanks for the input! Making the dataset now. This will take a few minutes.')
    
    return parameters