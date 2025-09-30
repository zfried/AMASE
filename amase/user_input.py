"""
User input handling for AMASE.
Handles all user interactions, input validation, and parameter collection.
"""

import os
import pandas as pd
from rdkit import Chem
from typing import List, Tuple, Optional, Dict, Any
from config import DEFAULT_VALID_ATOMS, ALL_VALID_ATOMS


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
        #print("- Tab or space separated")
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
        response = input('Do you want to consider structural relevence? If not, only the spectroscopy will be considered (y/n): \n').strip().lower()
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


def collect_all_parameters() -> Dict[str, Any]:
    """Collect all user parameters in sequence."""
    print('')
    #print('-------------------------')
    #print('Welcome to AMASE! This program will help you identify molecules in a mixture based on rotational spectroscopic data and known molecular structures. If you have any issues, please email zfried@mit.edu')
    #print('-------------------------')
    #print('')
    
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
    #artifact_frequencies = get_artifact_frequencies()
    
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
        #'artifact_frequencies': artifact_frequencies
    }
    
    print()
    print('Thanks for the input! Making the dataset now. This will take a few minutes.')
    
    return parameters