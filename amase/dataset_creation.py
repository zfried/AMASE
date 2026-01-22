"""
Dataset creation handling for AMASE.
Loads spectrum and creates dataset of peaks and molecular
"""


import os
import csv
import gzip
import pickle
import shutil
import time
import pandas as pd
import numpy as np
from rdkit import Chem
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")
from .molsim_classes import Source, Simulation, Continuum
from .molsim_utils import load_obs, find_limits, find_peaks, get_rms, load_mol

from .config import (
    DEFAULT_DIRECTORIES, DEFAULT_FILENAMES, MAX_MOLS, FREQUENCY_TOLERANCE,
    INVALID_MOLECULES, SPEED_OF_LIGHT_KMS, MOLSIM_CONTINUUM_TYPE,
    MOLSIM_CONTINUUM_PARAMS, DEFAULT_COLUMN_DENSITY, DEFAULT_ASTRO_PARAMS
)
from .amase_utils import gaussian, sortTupleArray, find_indices_within_threshold, addToGraph, edgeStringToList, stringToList, getCountDictionary

def load_graph(direc):
    edge = pd.read_csv(os.path.join(direc, 'edges.csv'))
    edges = edgeStringToList(list(edge['edges']))

    full = pd.read_csv(os.path.join(direc, 'all_smiles.csv'))
    smiles = list(full['smiles'])

    countDict = getCountDictionary(direc)

    vectorDF = pd.read_csv(os.path.join(direc, 'all_vectors.csv'))
    vectorSmiles = list(vectorDF['smiles'])
    allVectors = np.array(stringToList(list(vectorDF['vectors'])))

    return edges, smiles, countDict, vectorSmiles, allVectors

    

def estimate_median_linewidth(freq_arr, int_arr, peak_freqs, peak_ints, fwhm_guess, window=5.0):
    """
    freqs: array of frequencies
    intensities: array of corresponding intensities
    peaks: array of peak centers
    window: +/- window around peak for fitting (MHz)
    """
    widths = []
    cIt = 0
    velWidths = []
    for mu in peak_freqs:
        # select points within +/- window
        mask = (freq_arr >= mu - window) & (freq_arr <= mu + window)
        x = freq_arr[mask]
        y = int_arr[mask]

        # initial guesses
        amp_guess = peak_ints[cIt]

        try:
            popt, _ = curve_fit(gaussian, x, y, p0=[amp_guess, mu, fwhm_guess/2.355])
            sigma = abs(popt[2])
            fwhm = 2.355 * sigma
            widths.append(fwhm)
            velFWHM = (fwhm * 299792.458) / mu  # km/s
            velWidths.append(velFWHM)
        except RuntimeError:
            continue
        cIt += 1

    return np.median(widths), widths, np.median(velWidths), velWidths


def load_analyze_dataset(specPath, sig):

    # running molsim peak finder to get first guess at peaks
    data = load_obs(specPath, type='txt')
    ll0, ul0 = find_limits(data.spectrum.frequency)
    freq_arr = data.spectrum.frequency
    int_arr = data.spectrum.Tb
    resolution = data.spectrum.frequency[1] - data.spectrum.frequency[0]
    ckm = SPEED_OF_LIGHT_KMS
    min_separation = resolution * ckm / np.amax(freq_arr)
    peak_indices = find_peaks(freq_arr, int_arr, res=resolution, min_sep=min_separation, sigma=sig)
    peak_indices_original = peak_indices
    peak_freqs = data.spectrum.frequency[peak_indices]
    peak_ints = abs(data.spectrum.Tb[peak_indices])

    # estimating median linewidth
    dVGuess1, dvCalcs, dvVel1, dvVels = estimate_median_linewidth(freq_arr, int_arr, peak_freqs, peak_ints, fwhm_guess=1.0)
    dv_val_freq, dvCalcs2, dv_val_vel, dvVels2 = estimate_median_linewidth(freq_arr, int_arr, peak_freqs, peak_ints, fwhm_guess=dVGuess1)
    print('Frequency linewidth:', round(dv_val_freq,3), 'MHz')
    # running molsim peak finder again with new linewidth
    peak_indices = find_peaks(freq_arr, int_arr, res=resolution, min_sep=min_separation, sigma=sig)
    peak_indices_original = peak_indices
    peak_freqs = data.spectrum.frequency[peak_indices]
    peak_ints = abs(data.spectrum.Tb[peak_indices])
    peak_indices_full = find_peaks(freq_arr, int_arr, res=resolution, min_sep=min_separation, sigma=2)
    peak_freqs_full = data.spectrum.frequency[peak_indices_full]
    peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])
    rms = get_rms(int_arr)

    print('')
    print('Number of peaks at ' + str(sig) + ' sigma significance in the spectrum: ' + str(len(peak_freqs)))
    print('')

    # sorting peaks by intensity
    combPeaks = [(peak_freqs[i], peak_ints[i]) for i in range(len(peak_freqs))]
    sortedCombPeaks = sortTupleArray(combPeaks)
    sortedCombPeaks.reverse()
    spectrum_freqs = [i[0] for i in sortedCombPeaks]
    spectrum_ints = [i[1] for i in sortedCombPeaks]


    return data, ll0, ul0, freq_arr, int_arr, resolution, peak_indices_original, peak_freqs, peak_ints, peak_indices_full, peak_freqs_full, peak_ints_full, rms, dv_val_freq, dv_val_vel, spectrum_freqs, spectrum_ints


def create_dataset_file(spectrum_freqs,spectrum_ints, ll0,ul0, localYN, localDirec, direc, edges, smiles, allVectors, countDict, vectorSmiles, maxMols, temp, dishSize, sourceSize, cont, data, resolution, freq_arr, dv_val_freq, dv_val_vel, dfNames, dfSmiles, dfIso, manual_add_smiles, force_ignore_molecules):
    
    tickScrape = time.perf_counter()
    noCanFreq = []
    noCanInts = []
    ckm = SPEED_OF_LIGHT_KMS

    force_ignore_molecules = set(force_ignore_molecules)

    #creating some required folders
    pathLocal = os.path.join(direc, 'local_catalogs')
    pathSplat = os.path.join(direc, 'splatalogue_catalogues')
    pathExtra = os.path.join(direc, 'added_catalogs')

    if os.path.isdir(pathSplat):
        shutil.rmtree(pathSplat)

    if os.path.isdir(pathLocal):
        shutil.rmtree(pathLocal)

    if os.path.isdir(pathExtra):
        shutil.rmtree(pathExtra)


    os.mkdir(pathSplat)
    os.mkdir(pathLocal)
    pathSplatCat = os.path.join(pathSplat, 'catalogues')
    os.mkdir(pathSplatCat)
    os.mkdir(pathExtra)


    #scraping catalogs and making dataset
    firstLine = ['obs frequency', 'obs intensity']

    for i in range(maxMols):
        idx = i + 1
        firstLine.append('mol name ' + str(idx))
        firstLine.append('mol form ' + str(idx))
        firstLine.append('smiles ' + str(idx))
        firstLine.append('frequency ' + str(idx))
        firstLine.append('uncertainty ' + str(idx))
        firstLine.append('isotope count ' + str(idx))
        firstLine.append('quantum number ' + str(idx))
        firstLine.append('catalog tag ' + str(idx))
        firstLine.append('linelist ' + str(idx))

    matrix = []

    matrix.append(firstLine)

    for i in range(len(spectrum_freqs)):
        if spectrum_freqs.count(spectrum_freqs[i]) == 1:
            matrix.append([spectrum_freqs[i], spectrum_ints[i]])

    # adding local lines to dataset
    newMatrix = []
    newMatrix.append(matrix[0])
    del matrix[0]
    #dischargeFreqs = [float(row[0]) for row in matrix]
    catalogIdx = []
    catalogNames = []
    catalogForm = []
    catalogTags = []
    catalogList = []
    fullCount = 0

    localSmiles = []
    alreadyChecked = []

    localFreqInts = {}
    minFreq = ll0
    maxFreq = ul0

    localMolsInput = {}

    if localYN == True:
        print('scraping local catalogs')
        print('')
        for filename in os.listdir(localDirec):
            q = os.path.join(localDirec, filename)
            # checking if it is a file
            if os.path.isfile(q) and q.endswith(
                    '.cat') and 'super' not in q and '.DS_Store' not in q:
                dfFreq = pd.DataFrame()
                splitName = q.split('.cat')
                splitName2 = splitName[0].split('/')
                molName = splitName2[-1]
                idx = dfNames.index(molName)
                smileValue = dfSmiles[idx]
                consider_molecule = True
                if Chem.MolFromSmiles(smileValue) is None:
                    consider_molecule = False
                    print('ignoring ', molName, ' because SMILES string (' + smileValue + ') is invalid' )

                if molName not in force_ignore_molecules and consider_molecule == True:
                    mol = load_mol(q, type='SPCAT')
                    localMolsInput[molName] = mol
                    minFreq = ll0
                    maxFreq = ul0
                    src = Source(Tex=temp, column=1.E10, dV = dv_val_vel,continuum = cont)
                    sim = Simulation(mol=mol, ll=minFreq, ul=maxFreq, observation = data, source=src, line_profile='Gaussian', use_obs=True)
                    if len(sim.spectrum.freq_profile) > 0:
                        #find peaks in simulated spectrum.
                        peak_indicesIndiv = find_peaks(sim.spectrum.freq_profile, sim.spectrum.int_profile, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_val_freq), is_sim=True)
                        if len(peak_indicesIndiv) > 0:
                            peak_freqs2 = sim.spectrum.freq_profile[peak_indicesIndiv]
                            peak_ints2 = abs(sim.spectrum.int_profile[peak_indicesIndiv])
                            if peak_ints2 is not None:
                                freqs = list(peak_freqs2)
                                ints = list(peak_ints2)
                                dfFreq['frequencies'] = freqs
                                dfFreq['intensities'] = ints
                                localFreqInts[molName] = (freqs,ints)
                                saveName = os.path.join(pathLocal, str(fullCount) + '.csv')
                                dfFreq.to_csv(saveName)

                                catalogIdx.append(fullCount)
                                catalogNames.append(molName)
                                catalogForm.append(molName)
                                catalogTags.append('<NA>')
                                catalogList.append('local')

                                catObj = mol.catalog
                                freqs = list(catObj.frequency)
                                uncs = list(catObj.freq_err)
                                count = 0
                                for freq in freqs:
                                    for i in range(len(spectrum_freqs)):
                                        if spectrum_freqs[i] > freq - dv_val_freq and spectrum_freqs[i] < freq + dv_val_freq:
                                            idx = dfNames.index(molName)
                                            smileValue = dfSmiles[idx]
                                            if smileValue not in alreadyChecked:
                                                alreadyChecked.append(smileValue)
                                                if smileValue not in smiles and 'NEED' not in smileValue:
                                                    print('adding ' + smileValue + ' to graph')
                                                    edges, smiles, allVectors, countDict, vectorSmiles = addToGraph(smileValue, edges,
                                                                                                                    smiles, countDict,
                                                                                                                    allVectors,
                                                                                                                    vectorSmiles, direc)

                                            matrix[i].append(molName)
                                            matrix[i].append(molName)
                                            localSmiles.append((i, smileValue))
                                            isoValue = dfIso[idx]
                                            matrix[i].append(smileValue)
                                            matrix[i].append(freq)
                                            matrix[i].append(uncs[count])
                                            matrix[i].append(isoValue)
                                            matrix[i].append('local')
                                            matrix[i].append('local')
                                            matrix[i].append('local')

                                    count += 1
                    fullCount += 1

    for row in matrix:
        newMatrix.append(row)

    pathDataset = os.path.join(direc, 'dataset.csv')
    file = open(pathDataset, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(newMatrix)

    catDF = pd.DataFrame()
    catDF['index'] = catalogIdx
    catDF['mol name'] = catalogNames
    catDF['formula'] = catalogForm
    catDF['molecule tag'] = catalogTags
    catDF['linelist'] = catalogList

    pathCat = os.path.join(pathLocal, 'catalog_list_local.csv')
    catDF.to_csv(pathCat)

    #print('done with local catalog scraping!')
    #print('')
    #print('querying CDMS/JPL')
    print('')

    '''
    molSmileDF = pd.read_csv(os.path.join(direc,'all_splat_smiles.csv'))
    dataframeMols = list(molSmileDF['mol'])
    dataframeSmiles = list(molSmileDF['smiles'])

    cdmsTagsDF = pd.read_csv(os.path.join(direc,'cdms_catalogs.csv'))
    cdmsTagMols = list(cdmsTagsDF['mols'])
    cdmsTags = list(cdmsTagsDF['tags'])

    jplTagsDF = pd.read_csv(os.path.join(direc,'jpl_catalogs.csv'))
    jplTagMols = list(jplTagsDF['mols'])
    jplTags = list(jplTagsDF['tags'])
    '''


    savedForms = []
    savedList = []
    savedTags = []
    savedCatIndices = []
    savedComb = []

    rowComb = []

    catCount = 0


    fullMatrix = []
    fullMatrix.append(newMatrix[0])
    del newMatrix[0]

    cdmsDirec = os.path.join(direc,'cdms_pkl_final/')
    cdmsFullDF = pd.read_csv(os.path.join(direc,'all_cdms_final_official.csv'))
    cdmsForms = list(cdmsFullDF['splat form'])
    cdmsNames = list(cdmsFullDF['splat name'])
    cdmsTags = list(cdmsFullDF['cat'])
    cdmsSmiles = list(cdmsFullDF['smiles'])
    cdmsTags = [t[1:-4] for t in cdmsTags]

    jplDirec = os.path.join(direc,'jpl_pkl_final/')
    jplFullDF = pd.read_csv(os.path.join(direc,'all_jpl_final_official.csv'))
    jplForms = list(jplFullDF['splat form'])
    jplNames = list(jplFullDF['splat name'])
    jplTags = list(jplFullDF['save tag'])
    jplSmiles = list(jplFullDF['smiles'])

    cdmsFreqInts = {}
    jplFreqInts = {}

    '''
    The following loop combines queries of CDMS and JPL to get all candidate molecules
    for all of the lines in the spectrum along with the required information. For all candidates,
    the spectrum is simulated at the inputted experimental temperature and saved in the 
    splatalogue_catalogs directory.
    '''

    freq_threshold = dv_val_freq

    allScrapedMols = []

    with gzip.open(os.path.join(direc,"transitions_database.pkl.gz"), "rb") as f:
        database_freqs, database_errs, database_tags, database_lists, database_smiles, database_names, database_isos, database_vibs, database_forms = pickle.load(f)
    
    ignore_smiles = []
    unique_smiles = np.unique(database_smiles)

    for s in unique_smiles:
        if 'NEEDS' not in s:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                ignore_smiles.append(s)
                print('ignoring ', s, 'due to an invalid SMILES string' )


    ignore_smiles = set(ignore_smiles)

    
    print('scraping cdms/jpl/lsd molecules')
    tick1 = time.perf_counter()
    for row in newMatrix:
        #print(row)
        sf = float(row[0])
        line_mols = []
        start_idx = np.searchsorted(database_freqs, sf - freq_threshold, side="left")
        end_idx = np.searchsorted(database_freqs, sf + freq_threshold, side="right")
        for match_idx in range(start_idx, end_idx):
            match_tu = (
            database_names[match_idx], database_forms[match_idx], database_smiles[match_idx], database_freqs[match_idx],
            database_errs[match_idx], database_isos[match_idx], database_tags[match_idx], database_lists[match_idx],
            database_vibs[match_idx])

            if database_forms[match_idx] not in force_ignore_molecules and database_smiles[match_idx] not in ignore_smiles:
                line_mols.append(match_tu)

        cdms_keys = {(tup[2], tup[5], tup[8]) for tup in line_mols if tup[7] == 'CDMS' and tup[2] != 'NEEDS SMILES'}
        jpl_tuples = [tup for tup in line_mols if tup[7] == 'JPL' and tup[2] != 'NEEDS SMILES']
        jpl_with_matching_cdms = [tup for tup in jpl_tuples if (tup[2], tup[5], tup[8]) in cdms_keys]

        tag_to_tuple = {tup[6]: tup for tup in line_mols}
        non_hyperfine_match = []

        for ta in tag_to_tuple:
            if ta > 200000:
                base_tag = ta - 200000
                if base_tag in tag_to_tuple:
                    non_hyperfine_match.append(tag_to_tuple[base_tag])

        local_molecule_count = (len(row) - 2) // 9
        #print(local_molecule_count)
        local_smi_iso = [
            (row[2 + 9 * i + 2], row[2 + 9 * i + 5])
            for i in range(local_molecule_count)
        ]

        matching_mols = [
            mol for mol in line_mols
            if [mol[2], mol[5]] in [list(pair) for pair in local_smi_iso]
        ]

        line_mols_final = []
        for lmf in line_mols:
            if lmf not in jpl_with_matching_cdms and lmf not in matching_mols and lmf not in non_hyperfine_match:
                line_mols_final.append(lmf)

        for lmf in line_mols_final:
            row.append(lmf[0])
            row.append(lmf[1])
            row.append(lmf[2])
            row.append(lmf[3])
            row.append(lmf[4])
            row.append(lmf[5])
            row.append(lmf[-2])
            row.append(lmf[-3])
            row.append(lmf[-2])

            if (lmf[1],lmf[-3],lmf[-2],lmf[2]) not in allScrapedMols:
                allScrapedMols.append((lmf[1],lmf[-3],lmf[-2],lmf[2]))

        numMols = int((len(row) - 2) / 9)
        if len(row) > 2:
            rem = maxMols - numMols
            for v in range(rem):
                for g in range(9):
                    row.append('NA')
            fullMatrix.append(row)
        else:
            print('Line at ' + str(row[0]) + ' has no molecular candidates, it is being ignored')
            print('')
            noCanFreq.append(float(row[0]))
            noCanInts.append(float(row[1]))




    tock1 = time.perf_counter()
    #print('time taken for scraping: ' + str(tock1-tick1))
    #print('now simulating')
    tick2 = time.perf_counter()

    src = Source(Tex=temp, column=1.E10, dV = dv_val_vel, continuum = cont)

    catCount = 0
    for asm in allScrapedMols:
        dfFreq = pd.DataFrame()
        if asm[2] == 'CDMS':
            tagString = f"{asm[1]:06d}"
            molDirec = os.path.join(cdmsDirec,tagString+'.pkl')
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            sim = Simulation(mol=mol, ll=minFreq, ul=maxFreq, observation = data, source=src, line_profile='Gaussian', use_obs=True)
            
            peak_indicesIndiv = find_peaks(sim.spectrum.freq_profile, sim.spectrum.int_profile, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_val_freq), is_sim=True)
            if len(peak_indicesIndiv) > 0:
                peak_freqs2 = sim.spectrum.freq_profile[peak_indicesIndiv]
                peak_ints2 = abs(sim.spectrum.int_profile[peak_indicesIndiv])
                if peak_ints2 is not None:
                    freqs = list(peak_freqs2)
                    ints = list(peak_ints2)
                    dfFreq['frequencies'] = freqs
                    dfFreq['intensities'] = ints
                    cdmsFreqInts[asm[0]] = (freqs,ints)
                    saveName = os.path.join(pathSplat, str(catCount) + '.csv')
                    dfFreq.to_csv(saveName)

                    savedCatIndices.append(catCount)
                    savedForms.append(asm[0])
                    savedTags.append(asm[1])
                    savedList.append('CDMS')
        else:
            tagString = str(asm[1])
            molDirec = os.path.join(jplDirec, tagString + '.pkl')
            with open(molDirec, 'rb') as md:
                mol = pickle.load(md)
            sim = Simulation(mol=mol, ll=minFreq, ul=maxFreq, observation = data, source=src, line_profile='Gaussian', use_obs=True)
            peak_indicesIndiv = find_peaks(sim.spectrum.freq_profile, sim.spectrum.int_profile, res=resolution, min_sep=max(resolution * ckm / np.amax(freq_arr),0.5*dv_val_freq), is_sim=True)
            if len(peak_indicesIndiv) > 0:
                peak_freqs2 = sim.spectrum.freq_profile[peak_indicesIndiv]
                peak_ints2 = abs(sim.spectrum.int_profile[peak_indicesIndiv])
                freqs = list(peak_freqs2)
                ints = list(peak_ints2)
                dfFreq['frequencies'] = freqs
                dfFreq['intensities'] = ints
                jplFreqInts[asm[0]] = (freqs,ints)
                saveName = os.path.join(pathSplat, str(catCount) + '.csv')
                dfFreq.to_csv(saveName)

                savedCatIndices.append(catCount)
                savedForms.append(asm[0])
                savedTags.append(asm[1])
                savedList.append('JPL')


        catCount += 1

        smile = asm[-1]
        if smile not in alreadyChecked:
            alreadyChecked.append(smile)
            if smile not in smiles and 'NEED' not in smile:
                print('adding ' + smile + ' to graph')
                edges, smiles, allVectors, countDict, vectorSmiles = addToGraph(smile, edges,
                                                                                smiles, countDict,
                                                                                allVectors,
                                                                                vectorSmiles, direc)


    tock2 = time.perf_counter()
    #print('time taken for scraping: ' + str(tock2-tick2))
    print('done with splatalogue query')

    dfTags = pd.DataFrame()
    dfTags['idx'] = savedCatIndices
    dfTags['formula'] = savedForms
    dfTags['tags'] = savedTags
    dfTags['linelist'] = savedList

    dfTags.to_csv(os.path.join(pathSplatCat, 'catalog_list.csv'))

    qualCount = 0

    # quality control step
    updatedFull = []
    updatedFull.append(fullMatrix[0])
    del fullMatrix[0]
    for row in fullMatrix:
        if len(row) != 9 * maxMols + 2:
            inIndices = []
            for p in range(len(row)):
                if row[p] == None or row[p] == '':
                    if row[p + 1] == None or row[p + 1] == '':
                        inIndices.append(p)

            blankAlready = 0
            for blankIdx in inIndices:
                sI = blankIdx - 5 * blankAlready
                eI = blankIdx - 5 * blankAlready + 5
                del row[sI:eI]
                blankAlready += 1

            updatedFull.append(row)
        else:
            updatedFull.append(row)

        qualCount += 1

    fullMatrix = updatedFull

    finalMatrix2 = []

    for row in fullMatrix:
        if len(row) < 2 + 9 * maxMols:
            rem = 2 + 9 * maxMols - len(row)
            for c in range(rem):
                row.append('NA')

        finalMatrix2.append(row)

    fullMatrix = finalMatrix2

    pathDatasetInt = os.path.join(direc, 'dataset_intermediate.csv')

    file = open(pathDatasetInt, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(fullMatrix)


    '''
    The following loops allow for the user to input the SMILES
    strings of the molecular candidates in which this is not already available.
    '''
    updateSmiles = []
    withoutSmiles = []
    without1 = []
    for row in fullMatrix:
        for q in range(maxMols):
            if row[q * 9 + 4] == 'NEEDS SMILES' and row[q * 9 + 2] not in without1:
                withoutSmiles.append((row[q * 9 + 2], row[q * 9 + 3]))
                without1.append(row[q * 9 + 2])

    if len(withoutSmiles) == 0:
        finalMatrix = fullMatrix
        pathDataset = os.path.join(direc, 'dataset_final.csv')

        file = open(pathDataset, 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(finalMatrix)

    else:
        without12 = []
        withoutSmiles2 = []

        count2000 = 0
        for q in without1:
            foundSmileUp = False
            for row in fullMatrix:
                for e in range(maxMols):
                    if row[e * 9 + 2] == q and 'NEED' not in row[e * 9 + 4]:
                        updateSmile = row[e * 9 + 4]
                        foundSimleUp = True

            if foundSmileUp == True:
                finalMatrix = []
                for row in fullMatrix:
                    alreadyPresentMols = []
                    for m in range(maxMols):
                        if row[m * 9 + 2] != 'NA':
                            alreadyPresentMols.append(row[m * 9 + 4])
                    for e in range(maxMols):
                        if row[e * 9 + 2] == q and 'NEED' in row[e * 9 + 4] and updateSmile not in alreadyPresentMols:
                            row[e * 9 + 4] = updateSmile
                        elif row[e * 9 + 2] == q and 'NEED' in row[e * 9 + 4] and updateSmile in alreadyPresentMols:
                            for g in range(9):
                                row[e * 9 + 2 + g] = 'NA'

                    finalMatrix.append(row)

                fullMatrix = finalMatrix



            else:
                without12.append(q)
                withoutSmiles2.append(withoutSmiles[count2000])

            count2000 += 1

        without1 = without12
        withoutSmiles = withoutSmiles2

        withoutCount = 0
        for q in without1:
            finalMatrix = []
            if manual_add_smiles == True:
                updatedSmileVal = input('Please input the SMILES string for ' + str(withoutSmiles[withoutCount][0]) + ' ' + str(
                    withoutSmiles[withoutCount][
                        1]) + '\n If you want to ignore this molecule, type: ig\n Please do NOT include minor isotopes in SMILES string\n')
            else:
                updatedSmileVal = 'ignore'
            if 'ignore' in updatedSmileVal or 'IGNORE' in updatedSmileVal or 'Ignore' in updatedSmileVal:
                for row in fullMatrix:
                    for e in range(maxMols):
                        if row[e * 9 + 2] == q and 'NEED' in row[e * 9 + 4]:
                            for g in range(9):
                                row[e * 9 + 2 + g] = 'NA'
                    finalMatrix.append(row)
            else:
                try:
                    finalSmile = Chem.MolToSmiles(Chem.MolFromSmiles(updatedSmileVal))
                    if finalSmile not in alreadyChecked:
                        alreadyChecked.append(finalSmile)
                        if finalSmile not in smiles and 'NEED' not in finalSmile:
                            print('adding ' + finalSmile + ' to graph')
                            edges, smiles, allVectors, countDict, vectorSmiles = addToGraph(finalSmile, edges, smiles,
                                                                                            countDict, allVectors,
                                                                                            vectorSmiles, direc)

                    for row in fullMatrix:
                        alreadyPresentMols = []
                        for m in range(maxMols):
                            if row[m * 9 + 2] != 'NA':
                                alreadyPresentMols.append(row[m * 9 + 4])
                        for e in range(maxMols):
                            if row[e * 9 + 2] == q and 'NEED' in row[e * 9 + 4] and finalSmile not in alreadyPresentMols:
                                row[e * 9 + 4] = finalSmile
                            elif row[e * 9 + 2] == q and 'NEED' in row[e * 9 + 4] and finalSmile in alreadyPresentMols:
                                for g in range(9):
                                    row[e * 9 + 2 + g] = 'NA'

                        finalMatrix.append(row)
                except:
                    print('not a valid SMILES string, skipping this molecule.')
                    for row in fullMatrix:
                        for e in range(maxMols):
                            if row[e * 9 + 2] == q and 'NEED' in row[e * 9 + 4]:
                                for g in range(9):
                                    row[e * 9 + 2 + g] = 'NA'
                        finalMatrix.append(row)

            withoutCount += 1
            pathDataset = os.path.join(direc, 'dataset_final.csv')

            file = open(pathDataset, 'w+', newline='')
            with file:
                write = csv.writer(file)
                write.writerows(finalMatrix)

            fullMatrix = finalMatrix

    finalMatrix_correct = []
    for i in finalMatrix:
        count_not_na = sum(1 for x in i if x != 'NA')
        if count_not_na > 2:
            finalMatrix_correct.append(i)
        else:
            print('Line at', float(i[0]), 'has no molecular candidates, it is being ignored')
            noCanFreq.append(float(i[0]))
            noCanInts.append(float(i[1]))

    finalMatrix = finalMatrix_correct
    fullMatrix = finalMatrix_correct

    file = open(pathDataset, 'w+', newline='')
    with file:
        write = csv.writer(file)
        write.writerows(finalMatrix)

    form_save = []
    smi_save = []
    for g in finalMatrix[1:]:
        for h in range(maxMols):
            if g[2 + 9 * h + 2] != 'NA':
                if g[2 + 9 * h + 1] not in form_save:
                    form_save.append(g[2 + 9 * h + 1])
                    smi_save.append(g[2 + 9 * h + 2])

    #saving file required for assigning lines. This contains all molecules in the dataset and their SMILES strings.
    dfSave = pd.DataFrame()
    dfSave['molecules'] = form_save
    dfSave['smiles'] = smi_save
    dfSave.to_csv(os.path.join(direc, 'mol_smiles.csv'))

    tockScrape = time.perf_counter()



    print('')
    print('Ok, thanks! Now running the assignment algorithm.')
    print('')
    scrapeMins = (tockScrape-tickScrape)/60
    scrapeMins2 = "{{:.{}f}}".format(2).format(scrapeMins)
    print('Catalog scraping took ' + str(scrapeMins2) + ' minutes.')

    return edges, smiles, allVectors, countDict, vectorSmiles, noCanFreq, noCanInts, localFreqInts, cdmsFreqInts, jplFreqInts, finalMatrix, localMolsInput


def full_dataset_creation(specPath, direc, sig, localYN, localDirec,temp, dfLocal, manual_add_smiles, force_ignore_molecules):
    """
    Overall function to create dataset of lines and molecular candidates from spectrum.
    """
    if dfLocal is not None:
        dfLocal = pd.read_csv(dfLocal)
        dfNames = list(dfLocal['name'])
        dfSmiles = list(dfLocal['smiles'])
        dfIso = list(dfLocal['iso'])
    else:
        dfNames = []
        dfSmiles = []
        dfIso = []

    cont = Continuum(type='thermal', params=0.0)
    edges, smiles, countDict, vectorSmiles, allVectors = load_graph(direc)
    dishSize = DEFAULT_ASTRO_PARAMS['dish_size']
    sourceSize = DEFAULT_ASTRO_PARAMS['source_size']
    data, ll0, ul0, freq_arr, int_arr, resolution, peak_indices_original, peak_freqs, peak_ints, peak_indices_full, peak_freqs_full, peak_ints_full, rms, dv_val_freq, dv_val_vel, spectrum_freqs, spectrum_ints = load_analyze_dataset(specPath, sig)
    edges, smiles, allVectors, countDict, vectorSmiles, noCanFreq, noCanInts, localFreqInts, cdmsFreqInts, jplFreqInts, finalMatrix, localMolsInput = create_dataset_file(spectrum_freqs,spectrum_ints, ll0,ul0, localYN, localDirec, direc, edges, smiles, allVectors, countDict, vectorSmiles, MAX_MOLS, temp, dishSize, sourceSize, cont, data, resolution, freq_arr, dv_val_freq, dv_val_vel, dfNames, dfSmiles, dfIso, manual_add_smiles, force_ignore_molecules)

    dataset_results = {
        "edges": edges,
        "smiles": smiles,
        "allVectors": allVectors,
        "countDict": countDict,
        "vectorSmiles": vectorSmiles,
        "noCanFreq": noCanFreq,
        "noCanInts": noCanInts,
        "localFreqInts": localFreqInts,
        "cdmsFreqInts": cdmsFreqInts,
        "jplFreqInts": jplFreqInts,
        "finalMatrix": finalMatrix,
        "peak_freqs_full": peak_freqs_full,
        "peak_ints_full": peak_ints_full,
        "rms": rms,
        "dv_val_freq": dv_val_freq,
        "dv_val_vel": dv_val_vel,
        "localMolsInput": localMolsInput,
        "peak_indices_original": peak_indices_original
    }

    return dataset_results
