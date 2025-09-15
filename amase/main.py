from user_input import collect_all_parameters
from dataset_creation import full_dataset_creation
from amase_utils import softmax, closest
from assign_lines import assign_all_lines
from output_text_file import create_output
from fit_model import full_model


#Run the user interaction
params = collect_all_parameters()
#Creating the dataset of lines and molecular candidates
edges, smiles, allVectors, countDict, vectorSmiles, noCanFreq, noCanInts, localFreqInts, cdmsFreqInts, jplFreqInts, finalMatrix, peak_freqs_full, peak_ints_full, rms, dv_val_freq, dv_val_vel, localMolsInput, peak_indices_original = full_dataset_creation(specPath = params['spectrum_path'], direc = params['directory_path'], sig = params['sigma_threshold'], localYN = params['local_catalogs_enabled'], localDirec = params['local_directory'], temp = params['temperature'], dfLocal = params['local_df'])
#Assigning lines to molecules
startingMols, newTestingScoresListFinal, newCombinedScoresList, actualFrequencies, allIndexTest, allReports, intensities = assign_all_lines(direc = params['directory_path'], startingMols = params['starting_molecules'], consider_structure = params['consider_structure'], smiles = smiles, edges = edges, countDict = countDict, localFreqInts = localFreqInts, cdmsFreqInts = cdmsFreqInts, jplFreqInts = jplFreqInts, peak_freqs_full = peak_freqs_full, peak_ints_full = peak_ints_full, rms = rms, validAtoms = params['valid_atoms'], dv_val_freq = dv_val_freq)
#Creating output text file
create_output(direc=params['directory_path'], startingMols = params['starting_molecules'], newTestingScoresListFinal=newTestingScoresListFinal, newCombinedScoresList=newCombinedScoresList, actualFrequencies=actualFrequencies, allIndexTest=allIndexTest, allReports=allReports)
#Fitting model of assigned molecules to spectrum
full_model(params['spectrum_path'], params['directory_path'], peak_indices_original, localMolsInput, actualFrequencies, intensities, params['temperature'], dv_val_vel, rms)
print('Thank you for running the AMASE program! If you have any questions or feedback, please reach out to zfried@mit.edu.')