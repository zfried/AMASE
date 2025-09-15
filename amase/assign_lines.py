"""
Functions that work to assign the lines in a rotational spectrum for AMASE.
Handles frequency, intensity, and structural relevance checks.
"""


from amase_utils import printProgressBar, closest, sortTupleArray, softmax
from config import MAX_MOLS, DEFAULT_SIGMA, DEFAULT_ASTRO_PARAMS, SCORE_THRESHOLD, GLOBAL_THRESHOLD, GLOBAL_THRESHOLD_ORIGINAL
import os
import pandas as pd
import pickle
import csv
import numpy as np
import math
from scipy import stats
from rdkit import Chem
import time



def checkIntensity(tag, linelist, molForm, line_int, molFreq, localFreqInts, cdmsFreqInts, jplFreqInts):
    '''
    Function that checks whether the simulated line intensity is reasonable enough for assignment.
    '''


    foundMol = False
    if linelist == 'local':
        if molForm in localFreqInts:
            dictEntry = localFreqInts[molForm]
            freqs = dictEntry[0]
            peak_ints = np.array(dictEntry[1])
            foundMol = True

    elif linelist == 'CDMS':
        if molForm in cdmsFreqInts:
            dictEntry = cdmsFreqInts[molForm]
            freqs = dictEntry[0]
            peak_ints = np.array(dictEntry[1])
            foundMol = True

    else:
        if molForm in jplFreqInts:
            dictEntry = jplFreqInts[molForm]
            freqs = dictEntry[0]
            peak_ints = np.array(dictEntry[1])
            foundMol = True

    if foundMol:
        closestIdx, closestFreq = closest(freqs, molFreq)

        if freqs.count(closestFreq) > 1:
            indices = [c for c, n in enumerate(freqs) if n == closestFreq]
            intSpecs = [peak_ints[q] for q in indices]
            intIdx = intSpecs.index(max(intSpecs))
            closestIdx = indices[intIdx]

        if abs(closestFreq - molFreq) > 0.5:
            closestFreq = 'NA'
        intValue = peak_ints[closestIdx]
        if intValue == 0 or math.isnan(intValue):
            scaleValue = 1.E20
        else:
            scaleValue = line_int / intValue
        peak_ints_scaled = peak_ints * scaleValue

        combList = [(freqs[q], peak_ints_scaled[q]) for q in range(len(freqs))]
        sortedComb = sortTupleArray(combList)
        sortedComb.reverse()

        maxInt = sortedComb[0][1]
        found = False
        for i in range(len(sortedComb)):
            if sortedComb[i][0] == closestFreq:
                molRank = i
                found = True

        if found == False:
            molRank = 1000
    
    else:
        return 1.E20, 1.E20, molFreq, 1.E-20, foundMol, [],[], 0, 0

    return maxInt, molRank, closestFreq, intValue, foundMol, freqs, peak_ints, closestIdx, closestFreq

def checkAllLines(linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms, foundMol, sim_freqs, sim_ints, closestIdx, closestFreq):
    '''
    This function checks whether at least half of the predicted 10 sigma lines
    of the molecular candidate are present in the spectrum. If at least half arent
    present, rule_out = True is returned
    '''
    #maxInt, molRank, closestFreq, line_int_value, foundMol, freqs_sim, peak_ints_sim, closestIdx, closestFreq

    if foundMol:
        closestActualIdx, closestActualFreq = closest(peak_freqs_full, freq)
        line_int_full = peak_ints_full[closestActualIdx]

        #if abs(closestFreq - freq) > 0.5:
        #    closestFreq = 'NA'
        intValue = sim_ints[closestIdx]
        if intValue == 0 or math.isnan(intValue):
            scaleValue = 1.E20
        else:
            scaleValue = line_int_full / intValue

        sim_ints_scaled = sim_ints * scaleValue
        combList = [(sim_freqs[q], sim_ints_scaled[q]) for q in range(len(sim_freqs))]
        sortedComb = sortTupleArray(combList)
        sortedComb.reverse()

        filteredCombList1 = [i for i in combList if i[1] > 10 * rms]
        filteredCombList = [i for i in filteredCombList1 if i[0] != closestFreq]
        
        correct = 0

        tolerance = 0.2
        rule_out = False
        if len(filteredCombList) > 1:
            for target_value in filteredCombList:
                inThere = np.any((peak_freqs_full >= target_value[0] - tolerance) & (peak_freqs_full <= target_value[0] + tolerance))

                if inThere == True:
                    correct += 1


            if correct / len(filteredCombList) < 0.5 and len(filteredCombList) > 1:
                rule_out = True

        else:
            filteredCombList1 = [i for i in combList if i[1] >= 5 * rms]
            filteredCombList = [i for i in filteredCombList1 if i[0] != closestFreq]
            if len(filteredCombList) > 1:
                for target_value in filteredCombList:
                    inThere = np.any(
                        (peak_freqs_full >= target_value[0] - tolerance) & (peak_freqs_full <= target_value[0] + tolerance))

                    if inThere == True:
                        correct += 1

                if correct / len(filteredCombList) < 0.3 and len(filteredCombList) > 1:
                    rule_out = True
    else:
        rule_out = True

    return rule_out

def runGraphRanking(smiles, detectedSmiles, edges, countDict):
    '''
    This function runs the graph based ranking system given the detected molecules.
    It returns a score for all molecules in the graph.
    '''

    nodeDictInit = {}
    nodeDict = {}
    # initializing the score of each molecule (i.e. assigning all weight to detected molecues)
    for smile in smiles:
        if smile in detectedSmiles:
            nodeDict[smile] = 10 * detectedSmiles[smile]
            nodeDictInit[smile] = 10 * detectedSmiles[smile]
        else:
            nodeDict[smile] = 0
            nodeDictInit[smile] = 0
    # maximum number of iterations of the calculation
    maxIt = 5000
    for i in range(maxIt):

        intermediateDict = nodeDictInit.copy()

        for edge in edges:
            # looping through each of the edges in the graph and updating the weights of the nodes
            updateNode = edge[0]
            partner = edge[1]
            partnerCount = countDict[partner]
            if partnerCount != 0:
                addedValue = nodeDict[partner] / (1.5 * partnerCount)
            else:
                addedValue = 0
            intermediateDict[updateNode] = intermediateDict[updateNode] + addedValue

        converged = True
        # checking for convergence
        for z in intermediateDict:
            if abs(intermediateDict[z] - nodeDict[z]) > 1e-7:
                converged = False
                break

        nodeDict = intermediateDict

        if converged == True:
            break

    # sorting the results
    sorted_dict = sorted(nodeDict.items(), key=lambda x: x[1])
    sorted_dict.reverse()
    sorted_smiles = [q[0] for q in sorted_dict]
    sorted_values = [q[1] for q in sorted_dict]

    return sorted_dict, sorted_smiles, sorted_values


def scaleScores(rule_out_variable, high_intensities, high_smiles, prev_best_variable, intensity_variable, mol_form_variable, smile_input, graph_smiles, graph_values, max_int_val, mol_rank_val, iso_value_indiv, freq_cat, detectedSmiles, consider_structure, validAtoms, correctFreq, dv_val_freq, maxObservedInt, smile):
    '''
    This function scales the score of the molecules based on the structural relevance, frequency and intensity checks.
    '''

    molecule_report = []

    if len(detectedSmiles) > 0 and consider_structure == True:
        if smile_input in graph_smiles:
            newIdx = graph_smiles.index(smile_input)
            value = graph_values[newIdx]
            per = stats.percentileofscore(graph_values, value)
        else:
            value = 0
            per = 0.1
    else:
        value = 10
        per = 100


    #print(per)

    hasInvalid = False
    mol = Chem.MolFromSmiles(smile_input)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in validAtoms:
            hasInvalid = True

    if per < 93:
        molecule_report.append('Structural relevance score not great. ')

    # scaling the score based on the frequency match
    offset = freq_cat - correctFreq
    scale = 1 - abs(offset*0.07 / (0.5*dv_val_freq)) #makes it such that it will rule out any molecule further than 0.5*fwhm away
    scaledPer = scale * per
    if scale < 0.93:
        molecule_report.append('Frequency match not great.')

    # scaling the score based on the intensity match
    if max_int_val > 6 * maxObservedInt:
        scaledPer = 0.5 * scaledPer
        molecule_report.append(
            'Intensity suggests that there should be unreasonably strong lines of this molecule in the spectrum. ')

    if rule_out_variable == True:
        scaledPer = 0.5 * scaledPer
        molecule_report.append('Too many of the simulated 10 sigma lines of this molecule are not present. ')

    if mol_rank_val > 25 and mol_form_variable not in prev_best_variable:
        scaledPer = 0.5 * scaledPer
        molecule_report.append(
            'This is the strongest observed line of this molecule in the spectrum, but it is simulated to be the  number ' + str(
                mol_rank_val) + ' strongest transition. ')

    if mol_form_variable in high_intensities:
        if max_int_val > 5 * high_intensities[mol_form_variable]:
            scaledPer = 0.5 * scaledPer
            molecule_report.append('The simulated relative intensities do not match with what is observed. ')

    else:
        if max_int_val > 5 * intensity_variable:
            scaledPer = 0.5 * scaledPer
            molecule_report.append(
                'This is the strongest observed line of this molecule in spectrum but is simulated to be too weak. ')

    # scaling the score if the molecule is a rare isotopologue with an unrealistic intensity.
    if iso_value_indiv != 0:
        if smile not in high_smiles or intensity_variable > (0.1 ** iso_value_indiv) * high_smiles[smile]:
        #if smile not in high_smiles or intensity_variable > (0.1) * high_smiles[smile]:

            scaledPer = 0.5 * scaledPer
            molecule_report.append('Isotopologue is too strong. ')

    # scaling score if the molecule contains an invalid atom
    if hasInvalid == True:
        scaledPer = 0.5 * scaledPer
        molecule_report.append('Contains an invalid atom. ')

    return scaledPer, molecule_report, value



#smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value
def spectroscopic_checks_single_molecule(smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value, sigmaDict, peak_freqs_full, peak_ints_full, rms, correctFreq, oldHighestIntensities, oldHighestSmiles, previousBest, detectedSmiles, consider_structure, validAtoms, dv_val_freq, maxObservedInt, localFreqInts, cdmsFreqInts, jplFreqInts):
    '''
    This function combines all of the functions for the intensity and frequency checks for each molecular candidate.
    '''
    #maxInt, molRank, closestFreq, intValue, foundMol, freqs, peak_ints, closestIdx, closestFreq
    #tag, linelist, molForm, line_int, molFreq, localFreqInts, cdmsFreqInts, jplFreqInts
    maxInt, molRank, closestFreq, line_int_value, foundMol, freqs_sim, peak_ints_sim, closestIdx, closestFreq = checkIntensity(tag, linelist, formula, intensity_input, freq, localFreqInts, cdmsFreqInts, jplFreqInts)
    #linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms, foundMol, sim_freqs, sim_ints, closestIdx, closestFreq
    rule_out_val = checkAllLines(linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms, foundMol,freqs_sim, peak_ints_sim, closestIdx, closestFreq)

    if correctFreq in sigmaDict:
        sigmaList = sigmaDict[correctFreq]
        sigmaList.append((formula, freq, rule_out_val))
        sigmaDict[correctFreq] = sigmaList
    else:
        sigmaDict[correctFreq] = [(formula, freq, rule_out_val)]
    #rule_out_variable, high_intensities, high_smiles, prev_best_variable, intensity_variable, mol_form_variable, smile_input, graph_smiles, graph_values, max_int_val, mol_rank_val, iso_value_indiv, freq_cat, detectedSmiles, consider_structure, validAtoms, correctFreq, dv_val_freq, maxObservedInt, smile
    scaledPer, newReport, value = scaleScores(rule_out_val, oldHighestIntensities, oldHighestSmiles, previousBest, intensity_input, formula, smile, graph_smiles_main, graph_values_main, maxInt, molRank, iso_value, freq, detectedSmiles, consider_structure, validAtoms, correctFreq, dv_val_freq, maxObservedInt, smile)
    tu2 = [smile, scaledPer, formula, qn, value, iso_value]

    return tu2, newReport

#smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value, rule_out_val
def spectroscopic_checks_single_molecule_final(smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value, rule_out_val, newHighestIntensities, newHighestSmiles, newPreviousBest, detectedSmiles, consider_structure, validAtoms, dv_val_freq, maxObservedInt, correctFreq, localFreqInts, cdmsFreqInts, jplFreqInts):
    '''
    This function combines all of the functions for the intensity and frequency checks for each molecular candidate.
    It is used for the final iteration only.
    '''

    #maxInt, molRank, closestFreq, line_int_value, foundMol, freqs_sim, peak_ints_sim, closestIdx, closestFreq = checkIntensity(tag, linelist, formula, intensity_input, freq)
    #linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms, foundMol, sim_freqs, sim_ints, closestIdx, closestFreq
    #rule_out_val = checkAllLines(linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms, foundMol,freqs_sim, peak_ints_sim, closestIdx, closestFreq)
    maxInt, molRank, closestFreq, line_int_value, foundMol, freqs_sim, peak_ints_sim, closestIdx, closestFreq = checkIntensity(tag, linelist, formula, intensity_input, freq, localFreqInts, cdmsFreqInts, jplFreqInts)

    scaledPer, newReport, value = scaleScores(rule_out_val, newHighestIntensities, newHighestSmiles, newPreviousBest, intensity_input, formula, smile, graph_smiles_main, graph_values_main, maxInt, molRank, iso_value, freq, detectedSmiles, consider_structure, validAtoms, correctFreq, dv_val_freq, maxObservedInt, smile)

    tu2 = [smile, scaledPer, formula, qn, value, iso_value]

    return tu2, newReport


def forwardRun(correctFreq, sorted_dict_previous, newCalc, detectedSmiles, testSmiles, forms, linelists, tags, testIso, testFrequencies, qns, intensityValue, smiles, edges, countDict, sigmaDict, peak_freqs_full, peak_ints_full, rms, oldHighestIntensities, oldHighestSmiles, previousBest, consider_structure, validAtoms, dv_val_freq, maxObservedInt, localFreqInts, cdmsFreqInts, jplFreqInts):
    '''
    This function calls many other functions to rank all of the molecular candidates
    for a line in question.

    In summary:
    This function runs the graph-based ranking system given the detected smiles.
    It then takes the molecular candidates for a given line and checks the frequency and
    intensity match (by calling several of the other functions). It then combines all of the
    scores (i.e. the graph, intensity, and frequency scores) and returns the sorted results for
    each molecular candidate.
    '''

    # running graph calculation.
    reportListForward = []
    if newCalc == True and len(detectedSmiles) > 0:
        sorted_dict, sorted_smiles, sorted_values = runGraphRanking(smiles, detectedSmiles, edges, countDict)
    else:
        sorted_dict = sorted_dict_previous

    # storing graph scores
    newSmiles = [z[0] for z in sorted_dict]
    newValues = [z[1] for z in sorted_dict]

    testingScoresFreq_Updated = []

    # looping through the molecular candidates for each line
    lineReport = []
    for idx in range(len(testSmiles)):
        newReport = []
        smile = testSmiles[idx]
        formula = forms[idx]
        linelist = linelists[idx]
        tag = tags[idx]
        iso = testIso[idx]
        freq = testFrequencies[idx]
        qn = qns[idx]

        #smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value, sigmaDict, peak_freqs_full, peak_ints_full, rms, correctFreq, oldHighestIntensities, oldHighestSmiles, previousBest, detectedSmiles, consider_structure, validAtoms, dv_val_freq, maxObservedInt
        tu2, newReport = spectroscopic_checks_single_molecule(smile, formula, linelist, tag, iso, freq, qn, intensityValue, newSmiles, newValues, iso, sigmaDict, peak_freqs_full, peak_ints_full, rms, correctFreq, oldHighestIntensities, oldHighestSmiles, previousBest, detectedSmiles, consider_structure, validAtoms, dv_val_freq, maxObservedInt, localFreqInts, cdmsFreqInts, jplFreqInts)
        lineReport.append(newReport)
        testingScoresFreq_Updated.append(tu2)

    sortedTuplesCombined, testingScoresFreqSort, topSmile, topGlobalScore, topMol, topScore, topGlobalScoreSecond, best, topReport = get_score_metrics(testingScoresFreq_Updated,lineReport)

    testingScoresSmiles = [i[0] for i in testingScoresFreqSort]
    softScores = [i[-1] for i in testingScoresFreqSort]
    globalScores = [i[1] for i in testingScoresFreqSort]

    return testingScoresFreqSort, testingScoresSmiles, softScores, testingScoresFreqSort, sorted_dict, globalScores, sortedTuplesCombined, topSmile, topGlobalScore, topMol, topScore, topReport



def get_score_metrics(index_test, combined_report):
    '''
    This function takes the scores from all candidate molecules and combines the scores of candidate lines of the same
    molecule so that several lines of the same molecule are not competing with one another.
    '''

    index_test_2 = []
    pc = 0
    for p in index_test:
        p.append(combined_report[pc])
        index_test_2.append(p)
        pc += 1


    index_test = index_test_2

    sortedNewTest = sortTupleArray(index_test)
    sortedNewTest.reverse()

    sortedReports = [z[-1] for z in sortedNewTest]
    percentiles = [z[1] for z in sortedNewTest]


    for p in index_test:
        del p[-1]


    # computing softmax scores
    soft = list(softmax(percentiles))

    for w in range(len(sortedNewTest)):
        sortedNewTest[w].append(soft[w])

    # combining softmax score of multiple lines if from the same molecular carrier
    # This ensures that several lines of the same molecule are not competing.
    testingScoresDictReverse = {}
    for e in sortedNewTest:
        if (e[0], e[2]) not in testingScoresDictReverse:
            testingScoresDictReverse[(e[0], e[2])] = e[-1]
        else:
            currentValue = testingScoresDictReverse[(e[0], e[2])]
            newValue = e[-1] + currentValue
            testingScoresDictReverse[(e[0], e[2])] = newValue

    keys = list(testingScoresDictReverse.keys())
    values = list(testingScoresDictReverse.values())

    tuplesCombined = [(keys[i], values[i]) for i in range(len(keys))]
    sortedTuplesCombined = sortTupleArray(tuplesCombined)
    sortedTuplesCombined.reverse()

    bestSmile = sortedNewTest[0][0]
    bestReport = sortedReports[0]
    bestGlobalScore = sortedNewTest[0][1]

    # storing the top scores and molecules
    for co in sortedTuplesCombined:
        if co[0][0] == bestSmile:
            bestMol = co[0][1]
            bestScore = co[1]
            break

    topGlobalScoreSecond = sortedNewTest[0][1]

    best = sortedTuplesCombined[0]

    return sortedTuplesCombined, sortedNewTest, bestSmile, bestGlobalScore, bestMol, bestScore, topGlobalScoreSecond, best, bestReport



def updateOverride(bestGlobalScore, globalThresh, bestMol, override, best_report):
    '''
    This function updates the override counter. This ultimately provides the algorithm a way
    to override the scoring if there is compelling enough evidence for a molecule being present.
    For example, if a molecule is ranked fairly highly but below the required thresholds enough
    times, the algorithm will override its calculation and list the molecule as detected.
    '''


    if len(best_report) == 1 and 'Structural relevance' in best_report[0]:
        if bestMol in override:
            override[bestMol] = override[bestMol] + 1
        else:
            override[bestMol] = 1

    return override


def updateDetected_Highest(sortedNewTest, newHighestIntensities, intensityReverse, newPreviousBest, bestScore, thresh,
                           bestGlobalScore, globalThresh, newDetectedSmiles, override,
                           intensities, newCombinedScoresList, newHighestSmiles, globalThreshOriginal, bestSmile, direc):
    '''
    This function updates the list of detected molecules and the list of highest observed intensities for each molecule
    after a line is evaluated.
    '''
    for sorted_test in sortedNewTest:
        if sorted_test[1] > globalThreshOriginal:
            if sorted_test[2] not in newHighestIntensities:
                newHighestIntensities[sorted_test[2]] = intensityReverse
            if sorted_test[0] not in newHighestSmiles:
                newHighestSmiles[sorted_test[0]] = intensityReverse
            if sorted_test[2] not in newPreviousBest:
                newPreviousBest.append(sorted_test[2])

    if bestScore > thresh and bestGlobalScore > globalThresh:
        newDetectedSmiles[bestSmile] = 1

    for over in override:
        if override[over] >= 3:
            newTestingScoresFinalMols = [i[0][0][1] for i in newCombinedScoresList]
            firstMolIdx = [c for c, n in enumerate(newTestingScoresFinalMols) if n == over][0]
            topOverrideIntensity = intensities[firstMolIdx]
            newHighestIntensities[over] = topOverrideIntensity

            if over not in newPreviousBest:
                newPreviousBest.append(over)

            molSmilesDF = pd.read_csv(os.path.join(direc, 'mol_smiles.csv'))
            dfMolecules = list(molSmilesDF['molecules'])
            dfSmiles = list(molSmilesDF['smiles'])
            moleculeIndex = dfMolecules.index(over)
            newDetectedSmiles[dfSmiles[moleculeIndex]] = 1

            if dfSmiles[moleculeIndex] not in newHighestSmiles:
                newHighestSmiles[dfSmiles[moleculeIndex]] = topOverrideIntensity

    return newHighestIntensities, newPreviousBest, newDetectedSmiles, newHighestSmiles


def updateMainScores(testingScoresFinal, startingMols, newDetectedSmiles, sortedTuplesCombined, testingScoresSmiles,
                     newHighestIntensities,
                     intensityValue, newPreviousBest, topScore, thresh, topGlobalScore, globalThresh, globalScores,
                     newHighestSmiles, globalThreshOriginal, topSmile):
    '''
    This function also updates the list of detected molecules, top scoring molecules, and the list of
    highest observed intensity for each molecule following the evaulation of a line.
    '''
    for start in startingMols:
        newDetectedSmiles[start] = 1

    for sorted_test in testingScoresFinal:
        if sorted_test[1] > globalThreshOriginal:
            if sorted_test[2] not in newHighestIntensities:
                newHighestIntensities[sorted_test[2]] = intensityValue
            if sorted_test[0] not in newHighestSmiles:
                newHighestSmiles[sorted_test[0]] = intensityValue
            if sorted_test[2] not in newPreviousBest:
                newPreviousBest.append(sorted_test[2])

    if topScore > thresh and topGlobalScore > globalThresh:
        newDetectedSmiles[topSmile] = 1

    return newDetectedSmiles, topSmile, topScore, topGlobalScore, newHighestIntensities, newPreviousBest, newHighestSmiles




def assign_all_lines(direc, startingMols, consider_structure, smiles, edges, countDict, localFreqInts, cdmsFreqInts, jplFreqInts, peak_freqs_full, peak_ints_full, rms, validAtoms, dv_val_freq):
    '''
    
    Full function that loops through all lines in the dataset and assigns molecules to each line.
    
    '''
    
    
    assignTick = time.perf_counter()
    actualFrequencies = []
    allSmiles = []
    allIso = []
    allFrequencies = []
    intensities = []
    molForms = []
    molTags = []
    molLinelist = []
    allQn = []
    file = open(os.path.join(direc, 'dataset_final.csv'), 'r')
    matrix = list(csv.reader(file, delimiter=","))
    del matrix[0]
    maxMols = MAX_MOLS
    numMols = maxMols

    # re-uploading the dataset matrix
    for row in matrix:
        intensities.append(float(row[1]))
        actualFrequencies.append(float(row[0]))
        rowSmiles = []
        rowIso = []
        rowFreq = []
        rowForms = []
        rowTags = []
        rowLinelist = []
        rowQN = []

        for i in range(numMols):
            idx = i * 9 + 4
            if row[idx] != 'NA':
                rowSmiles.append(row[idx])
                rowIso.append(int(row[idx + 3]))
                rowFreq.append(float(row[idx + 1]))
                rowForms.append(row[idx - 1])

                if row[idx + 5] != '<NA>' and row[idx + 5] != 'local':
                    rowTags.append(int(row[idx + 5]))
                else:
                    rowTags.append('NA')
                rowLinelist.append(row[idx + 6])
                rowQN.append(row[idx + 4])

        molForms.append(rowForms)
        allIso.append(rowIso)
        allSmiles.append(rowSmiles)
        allFrequencies.append(rowFreq)
        molTags.append(rowTags)
        molLinelist.append(rowLinelist)
        allQn.append(rowQN)

    maxObservedInt = intensities[0]  # maximum intensity line in the dataset (required for intensity checks)
    sigmaDict = {}
    thresh = SCORE_THRESHOLD
    globalThresh = 93
    globalThreshOriginal = GLOBAL_THRESHOLD_ORIGINAL
    newDetSmiles = True
    newHighestIntensities = {}
    newHighestSmiles = {}
    strongestLineMain = []
    detectedSmiles = {}
    for o in startingMols:
        detectedSmiles[o] = 1
    previousBest = []
    oldTestingScoresList = []
    oldBestReportsFull = []
    oldTestingScoresListFull = []
    oldCombinedTestingScoresList = []
    oldHighestIntensities = {}
    oldHighestSmiles = {}
    rankingList = []
    rankingListFinal = []
    testingScoresList = []
    testingScoresListFinal = []
    sorted_dict_last = {}
    updateCounter = 0
    overallLength = len(actualFrequencies)

    # printing progress bar
    printProgressBar(0, overallLength, prefix='Progress:', suffix='Complete', length=50)

    # beginning the assignment algorithm, looping through each line in the dataset
    for i in range(len(actualFrequencies)):

        # after iteration 100, the threshold is increased from 93 to 99 since the priors should
        # be more informed at this point.

        if i >= 100:
            globalThresh = 99

        if len(allSmiles[i]) > 0:

            loopIter = i

            # getting all molecular candidates for a line.
            correctFreq = actualFrequencies[i]
            testSmiles = allSmiles[i]
            testIso = allIso[i]
            testFrequencies = allFrequencies[i]
            forms = molForms[i]
            tags = molTags[i]
            linelists = molLinelist[i]
            qns = allQn[i]

            intensityValue = intensities[i]


            # running graph based ranking algorithm and obtaining scores for each molecular candidate.
            # This function call includes the checking of structural relevance, frequency, and intensity match.

            newCalc = False
            if len(detectedSmiles) == 0:
                newCalc = False
            else:
                if newDetSmiles == True:
                    if len(detectedSmiles) < 7:
                        newCalc = True
                    else:
                        updateCounter += 1
                        if updateCounter == 5:
                            newCalc = True
                            updateCounter = 0
            if consider_structure == False:
                newCalc = False

            #correctFreq, sorted_dict_previous, newCalc, detectedSmiles, testSmiles, forms, linelists, tags, testIso, testFrequencies, qns, intensityValue, smiles, edges, countDict, sigmaDict, peak_freqs_full, peak_ints_full, rms, oldHighestIntensities, oldHighestSmiles, previousBest
            #correctFreq, sorted_dict_previous, newCalc, detectedSmiles, testSmiles, forms, linelists, tags, testIso, testFrequencies, qns, intensityValue, smiles, edges, countDict, sigmaDict, peak_freqs_full, peak_ints_full, rms, oldHighestIntensities, oldHighestSmiles, previousBest, consider_structure, validAtoms, dv_val_freq, maxObservedInt
            testingScoresFinal, testingScoresSmiles, softScores, testingScores, sorted_dict, globalScores, sortedTuplesCombined, topSmile, topGlobalScore, topMol, topScore, bestReport_forward = forwardRun(correctFreq, sorted_dict_last,newCalc, detectedSmiles, testSmiles, forms, linelists, tags, testIso, testFrequencies, qns, intensityValue, smiles, edges, countDict, sigmaDict, peak_freqs_full, peak_ints_full, rms, oldHighestIntensities, oldHighestSmiles, previousBest, consider_structure, validAtoms, dv_val_freq, maxObservedInt, localFreqInts, cdmsFreqInts, jplFreqInts)

            sorted_dict_last = sorted_dict
            sorted_smiles = [q[0] for q in sorted_dict]
            sorted_values = [q[1] for q in sorted_dict]

        newBestReportsFinal = []
        newTestingScoresListFinal = []
        newDetectedSmiles = {}
        newPreviousBest = []
        newCombinedScoresList = []
        newBestGlobalScoresFull = []

        topReverseSmiles = []
        newHighestIntensities = {}
        newHighestSmiles = {}

        override = {}

        indicesBefore = list(range(i))
        if newCalc == True:
            # Now re-checking all previous assignments
            if len(indicesBefore) > 0:
                if topGlobalScore > 50 and topGlobalScore < globalThresh:
                    override[topMol] = 1
                for index in indicesBefore:
                    oldCombinedScores = oldCombinedTestingScoresList[index]
                    newIndexTest = []
                    testSmiles = allSmiles[index]
                    testIso = allIso[index]
                    testFrequencies = allFrequencies[index]
                    correctFreq = actualFrequencies[index]
                    intensityReverse = intensities[index]
                    formsReverse = molForms[index]
                    tagsReverse = molTags[index]
                    linelistsReverse = molLinelist[index]
                    qnsReverse = allQn[index]
                    report = []
                    sigmaListReverse = sigmaDict[correctFreq]

                    for z in range(len(testSmiles)):
                        # looking at the candidate molecules for each line
                        subReport = []
                        smile = testSmiles[z]
                        freq = testFrequencies[z]
                        iso = testIso[z]
                        form = formsReverse[z]
                        linelist = linelistsReverse[z]
                        tag = tagsReverse[z]
                        qn = qnsReverse[z]

                        # checking intensity and frequency match for all candidate molecules
                        #maxInt, molRank, closestFreq, intValue, foundMol, freqs, peak_ints, closestIdx, closestFreq
                        maxInt, molRank, closestFreq, line_int_value, foundMol, freqs_sim, peak_ints_sim, closestIdx, closestFreq = checkIntensity(tag, linelist, form, intensityReverse, freq, localFreqInts, cdmsFreqInts, jplFreqInts)


                        for sig in sigmaListReverse:
                            if sig[0] == form and sig[1] == freq:
                                rule_out_reverse = sig[2]

                                #rule_out_variable, high_intensities, high_smiles, prev_best_variable, intensity_variable, mol_form_variable, smile_input, graph_smiles, graph_values, max_int_val, mol_rank_val, iso_value_indiv, freq_cat, detectedSmiles, consider_structure, validAtoms, correctFreq, dv_val_freq, maxObservedInt, smile

                        scaledPer, subReport, value = scaleScores(rule_out_reverse, newHighestIntensities, newHighestSmiles,
                                                        newPreviousBest, intensityReverse, form, smile, sorted_smiles,
                                                        sorted_values, maxInt, molRank, iso, freq, detectedSmiles,consider_structure,validAtoms, correctFreq, dv_val_freq, maxObservedInt, smile)

                        tu2 = [smile, scaledPer, form, qn, value, iso]


                        newIndexTest.append(tu2)
                        report.append(subReport)

                    # compiling top scores

                    sortedTuplesCombinedReverse, sortedNewTest, bestSmile, bestGlobalScore, bestMol, bestScore, topGlobalScoreSecond, best, topReport = get_score_metrics(newIndexTest, report)
                    newBestReportsFinal.append(topReport)
                    newTestingScoresListFinal.append(sortedNewTest)
                    newCombinedScoresList.append(sortedTuplesCombinedReverse)
                    newBestGlobalScoresFull.append(bestGlobalScore)
                    topReverseSmiles.append(bestSmile)

                    # updating the override counter
                    override = updateOverride(bestGlobalScore, globalThresh, bestMol, override, topReport)

                    bestOld = oldCombinedScores[0]
                    oldBestSmile = bestOld[0][0]
                    oldBestScore = bestOld[1]
                    oldBestGlobalScore = oldTestingScoresList[index]

                    # updating lists storing detected molecules and top scores

                    '''
                    sortedNewTest, newHighestIntensities, intensityReverse, newPreviousBest, bestScore, thresh,
                           bestGlobalScore, globalThresh, newDetectedSmiles, override,
                           intensities, newCombinedScoresList, newHighestSmiles, globalThreshOriginal, bestSmile, direc
                    
                    '''

                    newHighestIntensities, newPreviousBest, newDetectedSmiles, newHighestSmiles = updateDetected_Highest(
                        sortedNewTest, newHighestIntensities, intensityReverse, newPreviousBest, bestScore, thresh,
                        bestGlobalScore, globalThresh, newDetectedSmiles, override, intensities, newCombinedScoresList,
                        newHighestSmiles, globalThreshOriginal, bestSmile, direc)

            newTestingScoresListFinal.append(testingScoresFinal)
            newCombinedScoresList.append(sortedTuplesCombined)
            newBestReportsFinal.append(bestReport_forward)


            newDetectedSmiles, topSmile, topScore, topGlobalScore, newHighestIntensities, newPreviousBest, newHighestSmiles = updateMainScores(
                testingScoresFinal, startingMols, newDetectedSmiles, sortedTuplesCombined, testingScoresSmiles,
                newHighestIntensities, intensityValue, newPreviousBest, topScore, thresh, topGlobalScore, globalThresh,
                globalScores, newHighestSmiles, globalThreshOriginal, topSmile)

            newBestGlobalScoresFull.append(topGlobalScore)

            previousBest = newPreviousBest
            newDetSmiles = False
            for newDet in newDetectedSmiles:
                if newDet not in detectedSmiles:
                    newDetSmiles = True

            detectedSmiles = newDetectedSmiles
            oldTestingScoresListFull = newTestingScoresListFinal
            oldBestReportsFull = newBestReportsFinal
            oldTestingScoresList = newBestGlobalScoresFull

            oldCombinedTestingScoresList = newCombinedScoresList
            oldHighestIntensities = newHighestIntensities
            oldHighestSmiles = newHighestSmiles
        else:
            for teVal in testingScoresFinal:
                if teVal[1] > globalThreshOriginal:
                    if teVal[2] not in oldHighestIntensities:
                        oldHighestIntensities[teVal[2]] = intensityValue
                    if teVal[0] not in oldHighestSmiles:
                        oldHighestSmiles[teVal[0]] = intensityValue
                    if teVal[2] not in previousBest:
                        previousBest.append(teVal[2])

            oldTestingScoresListFull.append(testingScoresFinal)
            oldBestReportsFull.append(bestReport_forward)
            oldTestingScoresList.append(topGlobalScore)

            oldCombinedTestingScoresList.append(sortedTuplesCombined)
            newDetSmiles = False
            if topScore > thresh and topGlobalScore > globalThresh:
                if topSmile not in detectedSmiles:
                    detectedSmiles[topSmile] = 1
                    newDetSmiles = True
            overNew = {}
            if consider_structure == True:
                for o in range(len(oldBestReportsFull)):
                    if len(oldBestReportsFull[o]) == 1 and 'Structural relevance' in oldBestReportsFull[o][0]:
                        if oldTestingScoresListFull[o][0][2] not in overNew:
                            overNew[oldTestingScoresListFull[o][0][2]] = 1
                        else:
                            overNew[oldTestingScoresListFull[o][0][2]] = overNew[oldTestingScoresListFull[o][0][2]] + 1

                for over in overNew:
                    if overNew[over] >= 3:
                        oldTestingScoresFinalMols = [m[0][0][1] for m in oldCombinedTestingScoresList]
                        firstMolIdx = [c for c, n in enumerate(oldTestingScoresFinalMols) if n == over][0]
                        topOverrideIntensity = intensities[firstMolIdx]
                        if over not in oldHighestIntensities:
                            newDetSmiles = True
                        elif oldHighestIntensities[over] != topOverrideIntensity:
                            newDetSmiles = True
                        oldHighestIntensities[over] = topOverrideIntensity

                        if over not in previousBest:
                            previousBest.append(over)

                        molSmilesDF = pd.read_csv(os.path.join(direc, 'mol_smiles.csv'))
                        dfMolecules = list(molSmilesDF['molecules'])
                        dfSmiles = list(molSmilesDF['smiles'])
                        moleculeIndex = dfMolecules.index(over)
                        if dfSmiles[moleculeIndex] not in detectedSmiles:
                            detectedSmiles[dfSmiles[moleculeIndex]] = 1
                            newDetSmiles = True

                        if dfSmiles[moleculeIndex] not in oldHighestSmiles:
                            oldHighestSmiles[dfSmiles[moleculeIndex]] = topOverrideIntensity


        printProgressBar(i + 1, overallLength, prefix='Progress:', suffix='Complete', length=50)


    if consider_structure == True or consider_structure == False:
        print('running final iteration, just a few more minutes!')
        print('')


        # Running calculation and checking all lines one final time
        if consider_structure == True:
            sorted_dict, sorted_smiles, sorted_values = runGraphRanking(smiles, detectedSmiles, edges, countDict)
            newSmiles = [z[0] for z in sorted_dict]
            newValues = [z[1] for z in sorted_dict]
            #saving the highest ranked non-detected molecules
            candidate_mols = []
            for smi in sorted_smiles:
                if len(candidate_mols) < 10000:
                    candidate_mols.append(smi)

            dfCand = pd.DataFrame()
            dfCand['smiles'] = candidate_mols
            dfCand.to_csv(os.path.join(direc,'u_line_candidates.csv'))

            idxListCharge = []
            idxListRad = []

            for i in range(len(candidate_mols)):
                if '+' not in candidate_mols[i] and '-' not in candidate_mols[i]:
                    idxListCharge.append(i)

            dfCharge = dfCand.iloc[idxListCharge]
            dfCharge.to_csv(os.path.join(direc,'u_line_candidates_non_charged.csv'))
        else:
            sorted_dict = {}
            newSmiles = [z[0] for z in sorted_dict]
            newValues = [z[1] for z in sorted_dict]
            #detectedSmiles = []



        newTestingScoresListFinal = []
        newBestReportsFinal = []
        newDetectedSmiles = {}
        newPreviousBest = []
        newCombinedScoresList = []
        newBestGlobalScoresFull = []
        topReverseSmiles = []
        newHighestIntensities = {}
        newHighestSmiles = {}
        override = {}
        indicesAll = list(range(len(allFrequencies)))

        allIndexTest = []
        allReports = []


        # looping through all lines
        for index in indicesAll:
            newIndexTest = []
            testSmiles = allSmiles[index]
            testIso = allIso[index]
            testFrequencies = allFrequencies[index]
            intensityReverse = intensities[index]
            formsReverse = molForms[index]
            tagsReverse = molTags[index]
            linelistsReverse = molLinelist[index]
            qnsReverse = allQn[index]
            correctFreq = actualFrequencies[index]
            report = []
            newIndexTestOrd = []

            sigmaListReverse = sigmaDict[correctFreq]

            # looping for all candidate molecules for each line
            for z in range(len(testSmiles)):
                subReport = []
                smile = testSmiles[z]
                freq = testFrequencies[z]
                iso = testIso[z]
                form = formsReverse[z]
                linelist = linelistsReverse[z]
                tag = tagsReverse[z]
                qn = qnsReverse[z]

                foundSig = False
                for sig in sigmaListReverse:
                    if sig[0] == form and sig[1] == freq:
                        rule_out_reverse = sig[2]
                        foundSig = True

                if foundSig == False:
                    rule_out_reverse = True
                    print('not present in rule out reverse')

                #smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value, rule_out_val, newHighestIntensities, newHighestSmiles, newPreviousBest, detectedSmiles, consider_structure, validAtoms, dv_val_freq, maxObservedInt, correctFreq
                tu2, subReport = spectroscopic_checks_single_molecule_final(smile,form,linelist,tag,iso,freq,qn,intensityReverse,newSmiles,newValues, iso, rule_out_reverse, newHighestIntensities, newHighestSmiles, newPreviousBest, detectedSmiles, consider_structure, validAtoms, dv_val_freq, maxObservedInt, correctFreq, localFreqInts, cdmsFreqInts, jplFreqInts)



                newIndexTestOrd.append(tu2)
                newIndexTest.append(tu2)
                report.append(subReport)
            allIndexTest.append(newIndexTestOrd)
            allReports.append(report)

            # obtaining and storing scores

            sortedTuplesCombinedReverse, sortedNewTest, bestSmile, bestGlobalScore, bestMol, bestScore, topGlobalScoreSecond, best, topReport = get_score_metrics(
                newIndexTestOrd,report)

            newBestReportsFinal.append(topReport)
            newTestingScoresListFinal.append(sortedNewTest)
            newCombinedScoresList.append(sortedTuplesCombinedReverse)
            newBestGlobalScoresFull.append(bestGlobalScore)
            topReverseSmiles.append(bestSmile)

            # updating override counter
            override = updateOverride(bestGlobalScore, globalThresh, bestMol, override, topReport)

            newHighestIntensities, newPreviousBest, newDetectedSmiles, newHighestSmiles = updateDetected_Highest(
                sortedNewTest, newHighestIntensities, intensityReverse, newPreviousBest, bestScore, thresh,
                bestGlobalScore, globalThresh, newDetectedSmiles, override, intensities, newCombinedScoresList,
                newHighestSmiles, globalThreshOriginal, bestSmile, direc)
            
    
    saveCombFile = os.path.join(direc, 'combined_list.pkl')
    
    saveTestFile = os.path.join(direc, 'testing_list.pkl')
    with open(saveCombFile, "wb") as fp:
        pickle.dump(newCombinedScoresList, fp)

    with open(saveTestFile, "wb") as fp:
        pickle.dump(newTestingScoresListFinal, fp)

    assignTick = time.perf_counter()
    print(('Line assignment completed in', round(assignTick - assignTick, 2), 'seconds'))
    return startingMols, newTestingScoresListFinal, newCombinedScoresList, actualFrequencies, allIndexTest, allReports, intensities
        
