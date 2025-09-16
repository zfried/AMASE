from user_input import collect_all_parameters
from dataset_creation import full_dataset_creation
from assign_lines import assign_all_lines
from output_text_file import create_output
from fit_model import full_model


#Run the user interaction
params = collect_all_parameters()
#Creating the dataset of lines and molecular candidates
dataset_results = full_dataset_creation(specPath = params['spectrum_path'], direc = params['directory_path'], sig = params['sigma_threshold'], localYN = params['local_catalogs_enabled'], localDirec = params['local_directory'], temp = params['temperature'], dfLocal = params['local_df'])
#Assigning lines to molecules
assignScores = assign_all_lines(direc = params['directory_path'], startingMols = params['starting_molecules'], consider_structure = params['consider_structure'], smiles = dataset_results['smiles'], edges = dataset_results['edges'], countDict = dataset_results['countDict'], localFreqInts = dataset_results['localFreqInts'], cdmsFreqInts = dataset_results['cdmsFreqInts'], jplFreqInts = dataset_results['jplFreqInts'], peak_freqs_full = dataset_results['peak_freqs_full'], peak_ints_full = dataset_results['peak_ints_full'], rms = dataset_results['rms'], validAtoms = params['valid_atoms'], dv_val_freq = dataset_results['dv_val_freq'])
#Creating output text file
create_output(direc=params['directory_path'], startingMols = params['starting_molecules'], newTestingScoresListFinal=assignScores['newTestingScoresListFinal'], newCombinedScoresList=assignScores['newCombinedScoresList'], actualFrequencies=assignScores['actualFrequencies'], allIndexTest=assignScores['allIndexTest'], allReports=assignScores['allReports'])
#Fitting model of assigned molecules to spectrum
full_model(params['spectrum_path'], params['directory_path'], dataset_results['peak_indices_original'], dataset_results['localMolsInput'], assignScores['actualFrequencies'], assignScores['intensities'], params['temperature'], dataset_results['dv_val_vel'], dataset_results['rms'])
print('Thank you for running the AMASE program! If you have any questions or feedback, please reach out to zfried@mit.edu.')