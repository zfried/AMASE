from .user_input import collect_all_parameters
from .dataset_creation import full_dataset_creation
from .assign_lines import assign_all_lines
from .output_text_file import create_output
from .amase_utils import initial_banner
from .fit_model import full_model


def run_assignment(
    spectrum_path,
    directory_path,
    sigma_threshold,
    temperature,
    local_catalogs_enabled=False,
    local_directory=None,
    local_df=None,
    valid_atoms=['C', 'O', 'H', 'N', 'S'],
    consider_structure=False,
    starting_molecules=None,
    manual_add_smiles = False,
    force_ignore_molecules = [],
    force_include_molecules = [],
    stricter = True,
    spectrum_response_flat = False
):
    """
    Run the AMASE assignment algorithm.

    Parameters:
        spectrum_path (str): Path to the spectrum .txt file
        directory_path (str): Directory path for output and data files. Must include the files downloaded from the Dropbox repository linked in the README
        sigma_threshold (float): Sigma threshold for peak detection
        temperature (float): Temperature in Kelvin
        local_catalogs_enabled (bool, optional): Whether to use local catalogs. Default: False
        local_directory (str, optional): Path to directory containing local .cat files. Default: None
        local_df (str,): Path to CSV file with local catalog metadata. Default: None
        valid_atoms (list, optional): List of valid atoms for molecules. Default: ['C', 'O', 'H', 'N', 'S']
        consider_structure (bool, optional): Whether to consider molecular structure. Default: False
        starting_molecules (list, optional): List of starting molecules (SMILES strings) to initialize the structural relevance graph. 
            If empty, graph only initialized following the first assigned molecule. Default: None
        manual_add_smiles (bool, optional): Enable interactive prompts to manually input SMILES strings for molecules 
            lacking stored SMILES strings in the downloaded files. If False, the molecules without stored SMILES strings will be ignored. Default: None
        force_ignore_molecules (list, optional): Molecule names (from the downloaded CDMS and JPL .csv files or local directory of catalogs)
            that the algorithm will be forced to ignore. Useful if there is a false-positive assignment. Default: []
        force_include_molecules (list, optional): Molecule names (from the downloaded CDMS and JPL .csv files or local directory of catalogs)
            that the algorithm will be forced to include in the fit. Useful to test a molecule's presence. Default: []
        stricter (bool, optional): If True, has stricter molecular filtering during the assignment. 

    Returns:
        None: Results are saved to files in the specified directory
    """

    print('Hello! Beginning the assignment code now! If you run into any issues, please email zfried@mit.edu.')
    # Build params dict from provided arguments
    params = {
        'spectrum_path': spectrum_path,
        'directory_path': directory_path,
        'local_catalogs_enabled': local_catalogs_enabled,
        'local_directory': local_directory,
        'local_df': local_df,
        'sigma_threshold': sigma_threshold,
        'temperature': temperature,
        'valid_atoms': valid_atoms,
        'consider_structure': consider_structure,
        'starting_molecules': starting_molecules if starting_molecules is not None else [],
        'manual_add_smiles' : manual_add_smiles,
        'force_ignore_molecules': force_ignore_molecules,
        'force_include_molecules': force_include_molecules,
        'stricter': stricter,
        'spectrum_response_flat': spectrum_response_flat
    }

    # Run the analysis pipeline
    dataset_results = full_dataset_creation(
        specPath=params['spectrum_path'],
        direc=params['directory_path'],
        sig=params['sigma_threshold'],
        localYN=params['local_catalogs_enabled'],
        localDirec=params['local_directory'],
        temp=params['temperature'],
        dfLocal=params['local_df'],
        manual_add_smiles = params['manual_add_smiles'],
        force_ignore_molecules = params['force_ignore_molecules']
    )

    assignScores = assign_all_lines(
        direc=params['directory_path'],
        startingMols=params['starting_molecules'],
        consider_structure=params['consider_structure'],
        smiles=dataset_results['smiles'],
        edges=dataset_results['edges'],
        countDict=dataset_results['countDict'],
        localFreqInts=dataset_results['localFreqInts'],
        cdmsFreqInts=dataset_results['cdmsFreqInts'],
        jplFreqInts=dataset_results['jplFreqInts'],
        peak_freqs_full=dataset_results['peak_freqs_full'],
        peak_ints_full=dataset_results['peak_ints_full'],
        rms=dataset_results['rms'],
        validAtoms=params['valid_atoms'],
        dv_val_freq=dataset_results['dv_val_freq']
    )

    create_output(
        direc=params['directory_path'],
        startingMols=params['starting_molecules'],
        newTestingScoresListFinal=assignScores['newTestingScoresListFinal'],
        newCombinedScoresList=assignScores['newCombinedScoresList'],
        actualFrequencies=assignScores['actualFrequencies'],
        allIndexTest=assignScores['allIndexTest'],
        allReports=assignScores['allReports']
    )

    delMols = full_model(
        params['spectrum_path'],
        params['directory_path'],
        dataset_results['peak_indices_original'],
        dataset_results['localMolsInput'],
        assignScores['actualFrequencies'],
        assignScores['intensities'],
        params['temperature'],
        dataset_results['dv_val_vel'],
        dataset_results['rms'],
        dataset_results['dv_val_freq'],
        params['stricter'],
        params['spectrum_response_flat'],
        params['local_catalogs_enabled'],
        params['force_include_molecules']
    )
    
    # Create a second output file with warnings for removed molecules
    create_output(
        direc=params['directory_path'],
        startingMols=params['starting_molecules'],
        newTestingScoresListFinal=assignScores['newTestingScoresListFinal'],
        newCombinedScoresList=assignScores['newCombinedScoresList'],
        actualFrequencies=assignScores['actualFrequencies'],
        allIndexTest=assignScores['allIndexTest'],
        allReports=assignScores['allReports'],
        delMols=delMols
    )

    print('This is the end of the program. If you have any issues/feedback, please email zfried@mit.edu. The results are saved in several files in the directory you provided.\n fit_spectrum.html is an interactive plot of the spectra of all assigned molecules overlaid on the observational data.\n final_peak_results.csv then lists the molecule(s) that are the main carriers of every line.\n Have a good day!')
