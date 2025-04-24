import statistics
import os
import rdkit
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # turn off RDKit warning message
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
import scipy
from scipy import stats
from scipy import signal
import gc
import random
import time
import itertools
import csv
import math
import warnings
from astropy import units as u
from astropy import constants as c
import requests
import lxml.html as html
from lxml import etree
import pubchempy as pcp
from fastdist import fastdist
import mol2vec
from mol2vec import features
from mol2vec import helpers
import gensim
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
import pickle
warnings.filterwarnings("ignore")
import numpy as np
import io
import base64
from scipy.stats import norm
import plotly.graph_objects as go
import shutil

tick = time.perf_counter()

#required for some dependencies
np.float = float
np.int = int
np.object = object
np.bool = bool

import molsim

#maximum number of possible candidates for a single line
maxMols = 50


def near_whole(number):
    # Check if the absolute difference between the number and its rounded value is less than or equal to 0.05
    return abs(number - round(number)) <= 0.05


def inAddedArt(number, added_art_input):
    '''
    This function checks if a number is equal to a user inputted artifact frequency
    '''
    return round(number) in added_art_input



def sortTupleArray(tup):
    '''
    Function to sort arrays of tuples
    '''
    tup.sort(key=lambda x: x[1])
    return tup


def find_nearest(arr,val):
	idx = np.searchsorted(arr, val, side="left")
	if idx > 0 and (idx == len(arr) or math.fabs(val - arr[idx-1]) \
		 < math.fabs(val - arr[idx])):
		return idx-1
	else:
		return idx

def deleteDuplicates(list):
    '''
    Function to delete duplicate entries in a list
    '''
    seen = {}
    pos = 0
    for item in list:
        if item not in seen:
            seen[item] = True
            list[pos] = item
            pos += 1
    del list[pos:]


def str2float(string):
    split = list(string.split(','))
    floats_split = []
    for i in range(len(split)):
        floats = float(split[i])
        floats_split.append(floats)
    return floats_split


def stringToList(vectors):
    bracket_removed_mol2vec = []
    for i in range(len(vectors)):
        new_strings = vectors[i].replace('[', '')
        newer_strings = new_strings.replace(']', '')
        bracket_removed_mol2vec.append(newer_strings)

    # Convert all vectors
    xList = []
    for i in range(len(bracket_removed_mol2vec)):
        float_vec = str2float(bracket_removed_mol2vec[i])
        xList.append(float_vec)

    return xList


def listToString(vectors):
    string_indices = []
    for i in range(len(vectors)):
        knn_string = ', '.join(str(k) for k in vectors[i])
        string_indices.append(knn_string)

    bracket_string_indices = []
    for i in range(len(string_indices)):
        bracket_string = '[' + string_indices[i] + ']'
        bracket_string_indices.append(bracket_string)

    return bracket_string_indices


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def edgeStringToList(stringList):
    '''
    Function to convert graph edges into format required for algorithm
    '''
    fullList = []
    for test in stringList:
        test = test[1:-1]
        test = test.split(",")
        newList = []
        newList.append(test[0].strip())
        newList.append(test[1].strip())
        # newList.append(float(test[2]))

        fullList.append(newList)

    return fullList


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def getCountDictionary():
    '''
    Format to convert edge connection counts into format required for algorithm.
    '''
    countDict = {}
    full = pd.read_csv(os.path.join(direc, 'counts.csv'))
    smiles = list(full['smiles'])
    edgeCount = list(full['count'])

    for i in range(len(smiles)):
        countDict[smiles[i]] = edgeCount[i]
    return countDict


def closest(lst, K):
    '''
    Function to find the closest value to an input value (K) is a list (lst)
    '''
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx, lst[idx]



def checkIntensity(tag, linelist, molForm, line_int, molFreq):
    '''
    Function that checks whether the simulated line intensity is reasonable enough for assignment.
    '''

    if linelist == 'local':
        dictEntry = localFreqInts[molForm]
        freqs = dictEntry[0]
        peak_ints = np.array(dictEntry[1])

    elif linelist == 'CDMS':
        dictEntry = cdmsFreqInts[molForm]
        freqs = dictEntry[0]
        peak_ints = np.array(dictEntry[1])

    else:
        dictEntry = jplFreqInts[molForm]
        freqs = dictEntry[0]
        peak_ints = np.array(dictEntry[1])


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

    return maxInt, molRank, closestFreq, intValue



def checkIntensityFinal3(tag, linelist, molForm, line_int, molFreq):
    '''
    Function that checks whether the simulated line intensity is reasonable enough for assignment.
    This specific function is used only for interactive output.
    '''
    if linelist == 'local':
        formIdx = totalForms_local.index(molForm)
        indexStr = str(formIdx)

        individualDF = pd.read_csv(os.path.join(pathLocal, indexStr + '.csv'))
        freqs = list(individualDF['frequencies'])
        peak_ints = np.array(list(individualDF['intensities']))
    else:
        if linelist == "CDMS" or linelist == "JPL":
            formIdx = totalTags.index(tag)
            indexStr = str(totalIndices[formIdx])
            # indexStr = str(formIdx)
        else:
            formIdx = totalForms.index(molForm)
            # indexStr = str(formIdx)
            indexStr = str(totalIndices[formIdx])

        individualDF = pd.read_csv(os.path.join(pathSplat, indexStr + '.csv'))
        freqs = list(individualDF['frequencies'])
        peak_ints = np.array(list(individualDF['intensities']))

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

    return maxInt, molRank, closestFreq

def hasIso(mol):
    '''
    Function that returns number of rare isotopologues in formula
    '''
    isotopologue = 0
    isoList = ['17O', '(17)O', '18O', '(18)O', 'O18', '37Cl', '(37)Cl', 'Cl37', '15N', '(15)N', 'N15',
               'D', '(13)C', '13C', 'C13', '(50)Ti', '50Ti', 'Ti50', '33S', '(33)S', 'S33', '34S', '(34)S', 'S34',
               '36S', '(36)S', 'S36', '29Si', '(29)Si', 'Si29']
    for iso in isoList:
        if iso in mol:
            isotopologue += 1

    if "C13C" in mol:
        isotopologue = isotopologue - 1

    if "D2" in mol:
        isotopologue += 1

    if "D3" in mol:
        isotopologue += 2

    if "D4" in mol:
        isotopologue += 3

    return isotopologue


def sentences2vec(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.

    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032
    Returns
    -------
    np.array
    """
    # keys = list(model.wv.vocab.key)
    keys = set(model.wv.index_to_key)
    vec = []
    if unseen:
        unseen_vec = model.wv.get_vector(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys
                            else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.get_vector(y) for y in sentence
                            if y in set(sentence) & keys]))
    return np.array(vec)


def getFeatureVector(smile):
    '''
    This function takes a SMILES string of a main isotopologue as an input.
    It then uses a trained Mol2vec model to create a 70
    dimensional feature vector for the inputted molecule.
    It then calls the addIsotopologueData() function to add 19 dimensions
    that encode isotopic substitution.

    The function returns a list containing the resulting 89 dimensional feature vector

    '''

    modelPath = os.path.join(direc, 'mol2vec_model_final_70.pkl')
    model = word2vec.Word2Vec.load(modelPath)

    detectSmiles = [smile]

    molNames = [Chem.MolFromSmiles(smile) for smile in detectSmiles]
    sentences = [features.mol2alt_sentence(mole, 1) for mole in molNames]
    vectors = sentences2vec(sentences, model)
    vectors = [list(i) for i in vectors][0]

    return vectors


def addToGraph(addSmile, edges, smiles, countDict, allVectors, vectorSmiles):
    '''
    Function that adds a molecule to the graph.
    '''
    indivVector = np.array(getFeatureVector(addSmile))
    dist = fastdist.vector_to_matrix_distance(indivVector, allVectors, fastdist.euclidean, "euclidean")
    newEdges = []
    addedCount = 0
    for a in range(len(vectorSmiles)):
        if dist[a] < 11:
            newEdges.append([addSmile, vectorSmiles[a], dist[a]])
            newEdges.append([vectorSmiles[a], addSmile, dist[a]])
            addedCount += 1

    if addedCount > 0:
        # allVectors.append(indivVector)
        allVectors = np.vstack((allVectors, indivVector))
        smiles.append(addSmile)
        edges = edges + newEdges
        countDict[addSmile] = addedCount
        vectorSmiles.append(addSmile)

    return edges, smiles, allVectors, countDict, vectorSmiles



def checkAllLines(linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms):
    '''
    This function checks whether at least half of the predicted 10 sigma lines
    of the molecular candidate are present in the spectrum. If at least half arent
    present, rule_out = True is returned
    '''
    if linelist == 'local':
        dictEntry = localFreqInts[formula]
        sim_freqs = dictEntry[0]
        sim_ints = np.array(dictEntry[1])

    elif linelist == 'CDMS':
        dictEntry = cdmsFreqInts[formula]
        sim_freqs = dictEntry[0]
        sim_ints = np.array(dictEntry[1])

    else:
        dictEntry = jplFreqInts[formula]
        sim_freqs = dictEntry[0]
        sim_ints = np.array(dictEntry[1])

    closestActualIdx, closestActualFreq = closest(peak_freqs_full, freq)
    line_int_full = peak_ints_full[closestActualIdx]

    closestIdx, closestFreq = closest(sim_freqs, freq)
    if sim_freqs.count(closestFreq) > 1:
        indices = [c for c, n in enumerate(sim_freqs) if n == closestFreq]
        intSpecs = [sim_ints[q] for q in indices]
        intIdx = intSpecs.index(max(intSpecs))
        closestIdx = indices[intIdx]

    if abs(closestFreq - freq) > 0.5:
        closestFreq = 'NA'
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


def scaleScores(rule_out_variable, high_intensities, high_smiles, prev_best_variable, intensity_variable, mol_form_variable, smile_input, graph_smiles, graph_values, max_int_val, mol_rank_val, iso_value_indiv, freq_cat):
    '''
    This function scales the score of the molecules based on the structural relevance, frequency and intensity checks.
    '''

    molecule_report = []

    if len(detectedSmiles) > 0:
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

    hasInvalid = False
    mol = Chem.MolFromSmiles(smile_input)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in validAtoms:
            hasInvalid = True

    if per < 93:
        molecule_report.append('Structural relevance score not great. ')

    # scaling the score based on the frequency match
    offset = freq_cat - correctFreq
    scale = 1 - abs(offset / 3.5)
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


def spectroscopic_checks_single_molecule(smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value):
    '''
    This function combineds all of the functions for the intensity and frequency checks for each molecular cnadidate.
    '''
    maxInt, molRank, closestFreq, line_int_value = checkIntensity(tag, linelist, formula, intensity_input, freq)
    rule_out_val = checkAllLines(linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms)

    if correctFreq in sigmaDict:
        sigmaList = sigmaDict[correctFreq]
        sigmaList.append((formula, freq, rule_out_val))
        sigmaDict[correctFreq] = sigmaList
    else:
        sigmaDict[correctFreq] = [(formula, freq, rule_out_val)]

    scaledPer, newReport, value = scaleScores(rule_out_val, oldHighestIntensities, oldHighestSmiles, previousBest, intensity_input, formula, smile, graph_smiles_main, graph_values_main, maxInt, molRank, iso_value, freq)
    tu2 = [smile, scaledPer, formula, qn, value, iso_value]

    return tu2, newReport


def spectroscopic_checks_single_molecule_final(smile, formula, linelist, tag, iso, freq, qn, intensity_input, graph_smiles_main, graph_values_main, iso_value, rule_out_val):
    '''
    This function combineds all of the functions for the intensity and frequency checks for each molecular cnadidate.
    It is used for the final iteration only.
    '''
    maxInt, molRank, closestFreq, line_int_value = checkIntensity(tag, linelist, formula, intensity_input, freq)

    scaledPer, newReport, value = scaleScores(rule_out_val, newHighestIntensities, newHighestSmiles, newPreviousBest, intensity_input, formula, smile, graph_smiles_main, graph_values_main, maxInt, molRank, iso_value, freq)

    tu2 = [smile, scaledPer, formula, qn, value, iso_value]

    return tu2, newReport



def forwardRun(correctFreq, sorted_dict_previous):
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

        tu2, newReport = spectroscopic_checks_single_molecule(smile, formula, linelist, tag, iso, freq, qn, intensityValue, newSmiles, newValues, iso)
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
                           intensities, newCombinedScoresList, newHighestSmiles):
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
                     newHighestSmiles):
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


def checkIntensityOutput(tag, linelist, molForm, line_int, molFreq, low, high):
    '''
    Function that checks whether the simulated line intensity is reasonable enough for assignment.
    '''
    if linelist == 'local':
        formIdx = totalForms_local.index(molForm)
        indexStr = str(formIdx)

        individualDF = pd.read_csv(os.path.join(pathLocal, indexStr + '.csv'))
        freqs = list(individualDF['frequencies'])
        peak_ints = np.array(list(individualDF['intensities']))
    else:
        if linelist == "CDMS" or linelist == "JPL":
            formIdx = totalTags.index(tag)
            indexStr = str(totalIndices[formIdx])
            # indexStr = str(formIdx)
        else:
            formIdx = totalForms.index(molForm)
            # indexStr = str(formIdx)
            indexStr = str(totalIndices[formIdx])

        individualDF = pd.read_csv(os.path.join(pathSplat, indexStr + '.csv'))
        freqs = list(individualDF['frequencies'])
        peak_ints = np.array(list(individualDF['intensities']))

    closeCount = 0
    closeLines = []
    for o in range(len(freqs)):
        if abs(freqs[o] - molFreq) <= 0.5:
            closeCount += 1
            closeLines.append((freqs[o], peak_ints[o], o))

    sortedClose = sortTupleArray(closeLines)
    sortedClose.reverse()
    closestFreq = sortedClose[0][0]
    closestIdx = sortedClose[0][2]

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

    scaledFreqs = [e[0] for e in sortedComb]
    scaledInts = [e[1] for e in sortedComb]

    stickFreqs = []
    stickInts = []

    stickFreqs.append(low)
    stickInts.append(0)

    for m in range(len(freqs)):
        mu = freqs[m]
        sigma = 0.005

        stickFreqs.append(freqs[m] - sigma)
        stickInts.append(0)

        peak_height = peak_ints_scaled[m]

        stickFreqs.append(freqs[m] - 0.5 * sigma)
        stickInts.append(peak_height)
        stickFreqs.append(freqs[m])
        stickInts.append(peak_height)
        stickFreqs.append(freqs[m] + 0.5 * sigma)
        stickInts.append(peak_height)

        stickFreqs.append(freqs[m] + sigma)
        stickInts.append(0)

    stickFreqs.append(high)
    stickInts.append(0)

    stickFreqs2 = []

    for q in stickFreqs:
        off = q * vlsr_value / 299792
        newFreq = q - off
        stickFreqs2.append(newFreq)

    stickFreqs = stickFreqs2

    return scaledFreqs, scaledInts, stickFreqs, stickInts



def find_indices_within_threshold(values, target, threshold=0.5):
    """
    Find indices of values within a certain threshold of the target value using NumPy.

    Parameters:
    values (list of floats): List of values to search through.
    target (float): The target value to compare against.
    threshold (float): The threshold distance from the target value.

    Returns:
    list of int: Indices of the values within the threshold distance from the target.
    """
    values_array = np.array(values)
    indices = np.where(np.abs(values_array - target) <= threshold)[0]
    return indices.tolist()



sigmaDict = {}

# next few lines are interacting with the user
specPath = input('Please enter path to spectrum:\n')
print('')
direc = input('Please enter path to directory where files will be stored:\n')
print('')

direc = ''.join(direc.split())
if direc[-1] != '/':
    direc = direc + '/'
print('')
print('Thanks! Just a second, uploading dataset now.')
alreadyChecked = []
alreadyOut = []

cont = molsim.classes.Continuum(type='thermal', params=0.0)

edge = pd.read_csv(os.path.join(direc, 'edges.csv'))
edges = edgeStringToList(list(edge['edges']))

full = pd.read_csv(os.path.join(direc, 'all_smiles.csv'))
smiles = list(full['smiles'])

countDict = getCountDictionary()

vectorDF = pd.read_csv(os.path.join(direc, 'all_vectors.csv'))
vectorSmiles = list(vectorDF['smiles'])
allVectors = np.array(stringToList(list(vectorDF['vectors'])))

del vectorDF
del full
del edge

found_loc = False

while found_loc == False:
    localYN_input = input('Do you have catalogs on your local computer that you would like to consider (y/n): \n')
    if localYN_input == 'y' or localYN_input == 'Y' or localYN_input == 'n' or localYN_input == 'N':
        found_loc = True
    else:
        print('Invalid input. Please just type y or n')
        print('')

if 'y' in localYN_input or 'Y' in localYN_input:
    localYN = True
else:
    localYN = False

if localYN == True:
    localDirec = input('Great! Please enter path to directory to your local spectroscopic catalogs:\n')
    localDirec = ''.join(localDirec.split())
    if localDirec[-1] != '/':
        localDirec = localDirec + '/'
    print('')

    localDF = input(
        'Please enter path to the csv file that contains the SMILES strings and isotopic composition of molecules in local catalogs:\n')
    df = pd.read_csv(localDF)
    dfNames = list(df['name'])
    dfSmiles = list(df['smiles'])
    dfIso = list(df['iso'])

print('')
sig = int(input('What sigma lines do you want to consider (6 is recommended)?\n'))
print('')
temp = float(input('Please enter the experimental temperature (in Kelvin): \n'))
print('')
validInput = input(
    'Which atoms could feasibly be present in the mixture?\n If you type default, the valid atoms will be set to C, O, H, N, and S \n If you type all, all atoms in the periodic table will be considered. It is highly recommended that you specify (or choose default), however. \n If you would like to specify, please separate the atoms by commas (i.e. type C,O,S for carbon, oxygen and sulfur)\n')
print('')
validLower = ''.join(validInput.split()).lower()
validSpace = ''.join(validInput.split())
if validLower == 'default':
    validAtoms = ['C', 'O', 'H', 'N', 'S']
elif validLower == 'all':
    validAtoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
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
                  'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

else:
    validAtoms = []
    splitValid = validSpace.split(',')
    for i in splitValid:
        if i != '':
            validAtoms.append(i)

found_det = False
while found_det == False:
    hasDetInp = input('Do you have any known molecular precursors? (y/n): \n')
    if hasDetInp == 'y' or hasDetInp == 'Y' or hasDetInp == 'n' or hasDetInp == 'N':
        found_det = True
    else:
        print('Invalid input. Please just type y or n')
        print('')


print('')
if 'y' in hasDetInp or 'Y' in hasDetInp:
    hasDetInp = True
else:
    hasDetInp = False

if hasDetInp == False:
    startingMols = []

if hasDetInp == True:
    csvType = input(
        'You need to input the SMILES strings of the initial detected molecules. \n If you would like to type them individually, type 1. If you would like to input a csv file, type 2: \n')

    print('')

    if '1' in csvType:
        validStart = False
        while validStart == False:
            startingMols1 = input(
                'Enter the SMILES strings of the initial detected molecules. Please separate the SMILES string with a comma: \n')
            try:
                startingMolsSpace = ''.join(startingMols1.split())
                startingMols2 = startingMolsSpace.split(',')
                startingMols = [Chem.MolToSmiles(Chem.MolFromSmiles(u)) for u in startingMols2]
                validStart = True
                print('')

            except:
                print('You entered an invalid SMILES. ')

    elif '2' in csvType:
        validStart = False
        while validStart == False:
            try:
                csvDetPath = input(
                    'Please enter path to csv file. This needs to have the detected molecules in a column listed "SMILES." \n')
                csvDetPath = ''.join(csvDetPath.split())
                dfDet = pd.read_csv(csvDetPath)
                startingMols2 = list(dfDet['SMILES'])
                startingMols = [Chem.MolToSmiles(Chem.MolFromSmiles(u)) for u in startingMols2]
                validStart = True
                print('')
            except:
                print('There is an invalid SMILES in your .csv')

#astro = input('Is this an astronomical observation (y/n) - the code is currently only set up for single dish observations.\n')
#print('')
'''
inputtedRMS = False

found_rms = False
while found_rms == False:
    hasRMSInp = input('Do you want to manually input the RMS noise level? (y/n) \n Otherwise, the algorithm will determine it automatically: \n')
    if hasRMSInp == 'y' or hasRMSInp == 'Y':
        inputtedRMS = True
        found_rms = True
    elif hasRMSInp == 'n' or hasRMSInp == 'N':
        found_rms = True
    else:
        print('Invalid input. Please just type y or n')
        print('')

if inputtedRMS == True:
    rms = input('Please enter the RMS noise level of data:\n')
    rms = float(rms)
'''

inputtedRMS = False



astro = 'n'

if 'y' in astro or 'Y' in astro:
    dishSize = input('Please input the dish size in meters:\n')
    dishSize = float(dishSize)
    print('')
    sourceSize = input('Please input the source size in arcseconds:\n')
    sourceSize = float(sourceSize)
    print('')
    vlsr_value = input('Please input the vlsr in km/s:\n')
    vlsr_value = float(vlsr_value)
    print('')
    dv_value = input('Please input dV of lines in km/s:\n')
    dv_value = float(dv_value)
    artifactFreqs = []
    added_art = []

else:

    vlsr_value = 0
    '''
    print('Determining suspected artifact frequencies\n')
    print('')

    data = molsim.file_handling.load_obs(specPath, type='txt')
    ll0, ul0 = molsim.functions.find_limits(data.spectrum.frequency)
    freq_arr = data.spectrum.frequency
    int_arr = data.spectrum.Tb
    resolution = data.spectrum.frequency[1] - data.spectrum.frequency[0]
    ckm = (scipy.constants.c * 0.001)
    min_separation = resolution * ckm / np.amax(freq_arr)
    peak_indices = molsim.analysis.find_peaks(freq_arr, int_arr, res=resolution, min_sep=min_separation, sigma=3)
    peak_freqs = data.spectrum.frequency[peak_indices]
    peak_ints = abs(data.spectrum.Tb[peak_indices])

    intFreqsRound = []
    intFreqs = []
    for q in peak_freqs:
        if near_whole(q) == True:
            intFreqsRound.append(round(q))
            intFreqs.append(q)

    # print(intFreqs)

    artifactFreqs = []
    constantDifferences = []
    for i in range(len(intFreqsRound)):
        differences = []
        for q in range(len(intFreqsRound)):
            if q > i:
                differences.append(intFreqsRound[q] - intFreqsRound[i])

        for e in range(len(differences)):
            found = False
            if 2 * differences[e] in differences:
                if 3 * differences[e] <= max(differences):
                    if 3 * differences[e] in differences:
                        found = True
                else:
                    found = True

            if found == True and differences[e] not in constantDifferences:
                constantDifferences.append(differences[e])


            if found == True:
                fac = 0
                while fac * differences[e] <= max(differences):
                    if fac * differences[e] in differences or fac * differences[e] == 0:
                        facIdx = intFreqsRound.index(intFreqsRound[i] + fac * differences[e])
                        if intFreqs[facIdx] not in artifactFreqs:
                            artifactFreqs.append(intFreqs[facIdx])

                    fac += 1

    artifactFreqs.sort()
    print('These are the artifact frequencies I found:')
    print(artifactFreqs)

    found_art1 = False
    while found_art1 == False:
        artInput1 = input('Do you suspect any of these are actually molecular signal? (y/n)\n')
        if artInput1 == 'y' or artInput1 == 'Y' or artInput1 == 'n' or artInput1 == 'N':
            found_art1 = True
        else:
            print('Invalid input. Please just type y or n')
            print('')

    if 'y' in artInput1 or 'Y' in artInput1:
        print('')
        artValue = input('Ok, please provide the exact frequencies and separate them with commas.\n')
        artValue = ''.join(artValue.split())
        artValue = artValue.split(',')
        artValue = [float(i) for i in artValue]
        print(artValue)
        artifactFreqs2 = []
        for q in artifactFreqs:
            if q not in artValue:
                artifactFreqs2.append(q)

        artifactFreqs = artifactFreqs2

    # need to raise exception about smiles
    print('')
    found_art2 = False
    while found_art2 == False:
        artInput = input('Are there any other known instrument artifact frequencies? (y/n)\n')
        if artInput == 'y' or artInput == 'Y' or artInput == 'n' or artInput == 'N':
            found_art2 = True
        else:
            print('Invalid input. Please just type y or n')
            print('')


    if 'y' in artInput or 'Y ' in artInput:
        print('')
        art2 = input('OK great! Please type the artifact frequencies and separate them with commas.\n')
        art2 = ''.join(art2.split())
        art2Space = art2.split(',')
        # print(startingMols2)
        added_art = [float(u) for u in art2Space]

    else:
        added_art = []
    '''



added_art = []
artifactFreqs =[]

print('')

print('Thanks for the input! Making the dataset now. This will take a few minutes.')
tickScrape = time.perf_counter()


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


#if 'y' in astro or 'Y' in astro:


# running molsim peak finder
data = molsim.file_handling.load_obs(specPath, type='txt')
ll0, ul0 = molsim.functions.find_limits(data.spectrum.frequency)
freq_arr = data.spectrum.frequency
int_arr = data.spectrum.Tb
resolution = data.spectrum.frequency[1] - data.spectrum.frequency[0]
ckm = (scipy.constants.c * 0.001)
min_separation = resolution * ckm / np.amax(freq_arr)
peak_indices = molsim.analysis.find_peaks(freq_arr, int_arr, res=resolution, min_sep=min_separation, sigma=sig)
peak_indices_original = peak_indices
peak_freqs = data.spectrum.frequency[peak_indices]
peak_ints = abs(data.spectrum.Tb[peak_indices])

peak_freqs_vlsr = []

for i in peak_freqs:
    off = i * vlsr_value / 299792
    newFreq = i + off
    peak_freqs_vlsr.append(newFreq)

peak_freqs = np.array(peak_freqs_vlsr)

# finding all two sigma lines. This is required for future analysis.
peak_indices_full = molsim.analysis.find_peaks(freq_arr, int_arr, res=resolution, min_sep=min_separation, sigma=2)
peak_freqs_full = data.spectrum.frequency[peak_indices_full]
peak_ints_full = abs(data.spectrum.Tb[peak_indices_full])
if inputtedRMS == False:
    rms = molsim.stats.get_rms(int_arr)

print('noise level determined to be: ' + str(rms))
print('')

peak_freqs_vlsr = []

for i in peak_freqs_full:
    off = i * vlsr_value / 299792
    newFreq = i + off
    peak_freqs_vlsr.append(newFreq)

peak_freqs_full = np.array(peak_freqs_vlsr)

noCanFreq =[]
noCanInts = []

peak_indices3 = []
peak_ints3 = []
peak_freqs3 = []

for i in range(len(peak_indices)):
    if peak_freqs[i] not in artifactFreqs and inAddedArt(peak_freqs[i], added_art) == False:
        peak_ints3.append(peak_ints[i])
        peak_indices3.append(peak_indices[i])
        peak_freqs3.append(peak_freqs[i])



peak_indices = peak_indices3
peak_ints = peak_ints3
peak_freqs = peak_freqs3

print('')
print('Number of peaks at ' + str(sig) + ' sigma significance in the spectrum: ' + str(len(peak_freqs)))
print('')


# sorting peaks by intensity
combPeaks = [(peak_freqs[i], peak_ints[i]) for i in range(len(peak_freqs))]
sortedCombPeaks = sortTupleArray(combPeaks)
sortedCombPeaks.reverse()
spectrum_freqs = [i[0] for i in sortedCombPeaks]
spectrum_ints = [i[1] for i in sortedCombPeaks]


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
dischargeFreqs = [float(row[0]) for row in matrix]
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
            mol = molsim.file_handling.load_mol(q, type='SPCAT')
            minFreq = ll0
            maxFreq = ul0
            if 'y' in astro or 'Y' in astro:
                observatory1 = molsim.classes.Observatory(dish=dishSize)
                observation1 = molsim.classes.Observation(observatory=observatory1)
                src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=dv_value, continuum = cont)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014, observation=observation1)
            else:
                src = molsim.classes.Source(Tex=temp, column=1.E9, dV = 0.15,continuum = cont)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014)
            peak_freqs2 = sim.spectrum.frequency
            peak_ints2 = sim.spectrum.Tb
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
                        if spectrum_freqs[i] > freq - 0.5 and spectrum_freqs[i] < freq + 0.5:
                            idx = dfNames.index(molName)
                            smileValue = dfSmiles[idx]
                            if smileValue not in alreadyChecked:
                                alreadyChecked.append(smileValue)
                                if smileValue not in smiles and 'NEED' not in smileValue:
                                    print('adding ' + smileValue + ' to graph')
                                    edges, smiles, allVectors, countDict, vectorSmiles = addToGraph(smileValue, edges,
                                                                                                    smiles, countDict,
                                                                                                    allVectors,
                                                                                                    vectorSmiles)

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
print('querying CDMS/JPL')
print('')

molSmileDF = pd.read_csv(direc + 'all_splat_smiles.csv')
dataframeMols = list(molSmileDF['mol'])
dataframeSmiles = list(molSmileDF['smiles'])

cdmsTagsDF = pd.read_csv(direc + 'cdms_catalogs.csv')
cdmsTagMols = list(cdmsTagsDF['mols'])
cdmsTags = list(cdmsTagsDF['tags'])

jplTagsDF = pd.read_csv(direc + 'jpl_catalogs.csv')
jplTagMols = list(jplTagsDF['mols'])
jplTags = list(jplTagsDF['tags'])

# List of some invalid molecules from Splatalogue (since I wasnt sure of the correct SMILES strings)
invalid = ['Manganese monoxide', 'Bromine Dioxide',
           'Magnesium Isocyanide', 'Chromium monochloride', 'Scandium monosulfide',
           'Hydrochloric acid cation', '3-Silanetetrayl-1,2-Propadienylidene', 'UNIDENTIFIED',
           'Selenium Dioxide', '2-isocyano-3-propynenitrile', 'Aluminum cyanoacetylide', 'Silylidynyl cyanomethylene',
           'Yttrium monosulfide', 'Chloryl chloride', '3-Silanetetrayl-1,2-Propadienylidene ', 'Calcium monochloride',
           'Nickel monocarbonyl', 'Scandium monochloride', 'Potassium cyanide, potassium isocyanide',
           'Silicon Tetracarbide',
           'Calcium monoisocyanide', 'Iron Monocarbonyl', 'Calcium Monomethyl', 'Bromine Monoxide', 'Cobalt carbide', 'Hypobromous acid',
           'Aluminum Isocyanitrile']

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

cdmsDirec =  direc  + 'cdms_cats_final/'
cdmsFullDF = pd.read_csv(direc + 'all_cdms_final_official.csv')
cdmsForms = list(cdmsFullDF['splat form'])
cdmsNames = list(cdmsFullDF['splat name'])
cdmsTags = list(cdmsFullDF['cat'])
cdmsSmiles = list(cdmsFullDF['smiles'])
cdmsTags = [t[1:-4] for t in cdmsTags]


'''
The following loop combines queries of CDMS and JPL to get all candidate molecules
for all of the lines in the spectrum along with the required information. For all candidates,
the spectrum is simulated at the inputted experimental temperature and saved in the 
splatalogue_catalogs directory.
'''

cdmsFreqInts = {}
jplFreqInts = {}

for i in range(len(cdmsTags)):
    dfFreq = pd.DataFrame()
    q = cdmsDirec + str(cdmsTags[i]) + '.cat'
    mol = molsim.file_handling.load_mol(q, type='SPCAT')
    smile = cdmsSmiles[i]
    molPresent = False

    catObj = mol.catalog
    freqs = list(catObj.frequency)
    uncs = list(catObj.freq_err)

    for row in newMatrix:
        freq = float(row[0])
        if smile not in row:
            close_idx = find_indices_within_threshold(freqs, freq)
            if len(close_idx) != 0:
                molPresent = True
                for q in close_idx:
                    row.append(cdmsNames[i])
                    row.append(cdmsForms[i])
                    row.append(smile)
                    row.append(freqs[q])
                    row.append(uncs[q])
                    row.append(hasIso(cdmsForms[i]))
                    row.append('cdms')
                    row.append(cdmsTags[i])
                    row.append('CDMS')


    if molPresent == True:
        if smile not in alreadyChecked:
            alreadyChecked.append(smile)
            if smile not in smiles and 'NEED' not in smile:
                print('adding ' + smile + ' to graph')
                edges, smiles, allVectors, countDict, vectorSmiles = addToGraph(smile, edges,
                                                                                smiles, countDict,
                                                                                allVectors,
                                                                                vectorSmiles)

        if 'y' in astro or 'Y' in astro:
            observatory1 = molsim.classes.Observatory(dish=dishSize)
            observation1 = molsim.classes.Observation(observatory=observatory1)
            src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=0, continuum = cont)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014, observation=observation1)
        else:
            src = molsim.classes.Source(Tex=temp, column=1.E9, dV = 0.15, continuum = cont)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014)

        peak_freqs2 = sim.spectrum.frequency
        peak_ints2 = sim.spectrum.Tb
        if peak_ints2 is not None:
            freqs = list(peak_freqs2)
            ints = list(peak_ints2)
            dfFreq['frequencies'] = freqs
            dfFreq['intensities'] = ints
            cdmsFreqInts[cdmsForms[i]] = (freqs,ints)
            saveName = os.path.join(pathSplat, str(catCount) + '.csv')
            dfFreq.to_csv(saveName)

            savedCatIndices.append(catCount)
            savedForms.append(cdmsForms[i])
            savedTags.append(cdmsTags[i])
            savedList.append('CDMS')

        catCount += 1


jplDirec = direc  + 'jpl_cats_final/'
jplFullDF = pd.read_csv(direc  + 'all_jpl_final_official.csv')
jplForms = list(jplFullDF['splat form'])
jplNames = list(jplFullDF['splat name'])
jplTags = list(jplFullDF['save tag'])
jplSmiles = list(jplFullDF['smiles'])



for i in range(len(jplTags)):
    dfFreq = pd.DataFrame()
    q = jplDirec + str(jplTags[i]) + '.cat'
    mol = molsim.file_handling.load_mol(q, type='SPCAT')
    smile = jplSmiles[i]

    catObj = mol.catalog
    freqs = list(catObj.frequency)
    uncs = list(catObj.freq_err)

    molPresent = False

    for row in newMatrix:
        freq = float(row[0])
        if smile not in row:
            close_idx = find_indices_within_threshold(freqs, freq)
            if len(close_idx) != 0:
                molPresent = True
                for q in close_idx:
                    row.append(jplNames[i])
                    row.append(jplForms[i])
                    row.append(smile)
                    row.append(freqs[q])
                    row.append(uncs[q])
                    row.append(hasIso(jplForms[i]))
                    row.append('jpl')
                    row.append(jplTags[i])
                    row.append('JPL')


    if molPresent == True:
        if smile not in alreadyChecked:
            alreadyChecked.append(smile)
            if smile not in smiles and 'NEED' not in smile:
                print('adding ' + smile + ' to graph')
                edges, smiles, allVectors, countDict, vectorSmiles = addToGraph(smile, edges,
                                                                                smiles, countDict,
                                                                                allVectors,
                                                                                vectorSmiles)

        if 'y' in astro or 'Y' in astro:
            observatory1 = molsim.classes.Observatory(dish=dishSize)
            observation1 = molsim.classes.Observation(observatory=observatory1)
            src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=0, continuum = cont)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014, observation=observation1)
        else:
            src = molsim.classes.Source(Tex=temp, column=1.E9, dV = 0.15, continuum = cont)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014)

        peak_freqs2 = sim.spectrum.frequency
        peak_ints2 = sim.spectrum.Tb
        if peak_ints2 is not None:
            freqs = list(peak_freqs2)
            ints = list(peak_ints2)
            dfFreq['frequencies'] = freqs
            dfFreq['intensities'] = ints
            jplFreqInts[jplForms[i]] = (freqs,ints)
            saveName = os.path.join(pathSplat, str(catCount) + '.csv')
            dfFreq.to_csv(saveName)

            savedCatIndices.append(catCount)
            savedForms.append(jplForms[i])
            savedTags.append(jplTags[i])
            savedList.append('JPL')

        catCount += 1




for row in newMatrix:
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
        updatedSmileVal = input('Please input the SMILES string for ' + str(withoutSmiles[withoutCount][0]) + ' ' + str(
            withoutSmiles[withoutCount][
                1]) + '\n If you want to ignore this molecule, type: ignore\n Please do NOT include minor isotopes in SMILES string\n')
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
                                                                                        vectorSmiles)

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

tockScrape = time.perf_counter()



print('')
print('Ok, thanks! Now running the assignment algorithm.')
print('')
scrapeMins = (tockScrape-tickScrape)/60
scrapeMins2 = "{{:.{}f}}".format(2).format(scrapeMins)
print('Catalog scraping took ' + str(scrapeMins2) + ' minutes.')

analysisTick = time.perf_counter()


mol_smileMols = []
mol_smileSmiles = []

fullMatrix2 = fullMatrix
del fullMatrix2[0]
for row in fullMatrix2:
    for z in range(maxMols):
        molIdx = 9 * z + 3
        smileIdx = 9 * z + 4
        if row[molIdx] not in mol_smileMols:
            mol_smileMols.append(row[molIdx])
            mol_smileSmiles.append(row[smileIdx])

dfMolSmiles = pd.DataFrame()
dfMolSmiles['molecules'] = mol_smileMols
dfMolSmiles['smiles'] = mol_smileSmiles
dfMolSmiles.to_csv(os.path.join(direc, 'mol_smiles.csv'))

formDF = pd.read_csv(os.path.join(pathSplatCat, 'catalog_list.csv'))
totalForms = list(formDF['formula'])
totalTags = list(formDF['tags'])
#totalTags = [str(i) for i in totalTags]
totalIndices = list(formDF['idx'])

formDF_local = pd.read_csv(os.path.join(direc, 'local_catalogs/catalog_list_local.csv'))
totalForms_local = list(formDF_local['formula'])
totalTags_local = list(formDF_local['molecule tag'])


# uploading graph edges, smiles and connection counts


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


# setting threshold values
thresh = 0.7
globalThresh = 93
globalThreshOriginal = 93
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


        testingScoresFinal, testingScoresSmiles, softScores, testingScores, sorted_dict, globalScores, sortedTuplesCombined, topSmile, topGlobalScore, topMol, topScore, bestReport_forward = forwardRun(correctFreq, sorted_dict_last)

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
                    maxInt, molRank, closestFreq, line_int_value = checkIntensity(tag, linelist, form, intensityReverse, freq)


                    for sig in sigmaListReverse:
                        if sig[0] == form and sig[1] == freq:
                            rule_out_reverse = sig[2]

                    scaledPer, subReport, value = scaleScores(rule_out_reverse, newHighestIntensities, newHighestSmiles,
                                                       newPreviousBest, intensityReverse, form, smile, sorted_smiles,
                                                       sorted_values, maxInt, molRank, iso, freq)

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
                newHighestIntensities, newPreviousBest, newDetectedSmiles, newHighestSmiles = updateDetected_Highest(
                    sortedNewTest, newHighestIntensities, intensityReverse, newPreviousBest, bestScore, thresh,
                    bestGlobalScore, globalThresh, newDetectedSmiles, override, intensities, newCombinedScoresList,
                    newHighestSmiles)

        newTestingScoresListFinal.append(testingScoresFinal)
        newCombinedScoresList.append(sortedTuplesCombined)
        newBestReportsFinal.append(bestReport_forward)

        newDetectedSmiles, topSmile, topScore, topGlobalScore, newHighestIntensities, newPreviousBest, newHighestSmiles = updateMainScores(
            testingScoresFinal, startingMols, newDetectedSmiles, sortedTuplesCombined, testingScoresSmiles,
            newHighestIntensities, intensityValue, newPreviousBest, topScore, thresh, topGlobalScore, globalThresh,
            globalScores, newHighestSmiles)

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

    # updating progress bar
    printProgressBar(i + 1, overallLength, prefix='Progress:', suffix='Complete', length=50)


print('running final iteration, just a few more minutes!')
print('')


# Running calculation and checking all lines one final time
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

        tu2, subReport = spectroscopic_checks_single_molecule_final(smile,form,linelist,tag,iso,freq,qn,intensityReverse,newSmiles,newValues, iso, rule_out_reverse)
        
        

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
        newHighestSmiles)



tock = time.perf_counter()

# Writing the output file
f = open(os.path.join(direc, 'output_report.txt'), "w")

f.write('Initial Detected Molecules: ')
if len(startingMols) == 0:
    f.write('Nothing inputted')
else:
    for u in startingMols:
        f.write(str(u))
        f.write(' ')
f.write('\n')
totalTime = (tock - tick) / 60

f.write('Total Time Taken: ' + str(totalTime) + ' Minutes \n')
f.write('\n')
f.write('\n')

f.write('--------------------------------------\n')
category = []


#saving the lists of results as Pickle files
saveCombFile = os.path.join(direc, 'combined_list.pkl')
saveTestFile = os.path.join(direc, 'testing_list.pkl')
with open(saveCombFile, "wb") as fp:
    pickle.dump(newCombinedScoresList, fp)

with open(saveTestFile, "wb") as fp:
    pickle.dump(newTestingScoresListFinal, fp)


#Adding details to output text file
mulCount = 0
assignCount = 0
unCount = 0
for i in range(len(newTestingScoresListFinal)):
    f.write('LINE ' + str(i + 1) + ':\n')
    if 'y' in astro or 'Y' in astro:
        f.write('Velocity Shifted Frequency: ' + str(actualFrequencies[i]))
    else:
        f.write('Frequency: ' + str(actualFrequencies[i]))
    f.write('\n')
    f.write('Categorization: ')
    indivCarriers = []
    newTest = newTestingScoresListFinal[i]
    newComb = newCombinedScoresList[i]
    allMols = [z[2] for z in newTest]
    allSmiles = [z[0] for z in newTest]
    combMol = [z[0][1] for z in newComb]
    combScores = [z[1] for z in newComb]
    finalScores = []
    already = []
    for q in newTest:
        if q[2] not in already:
            indivScore = []
            indivScore.append(q[0])
            indivScore.append(q[2])
            indivScore.append(q[1])
            for e in range(len(combMol)):
                if combMol[e] == q[2]:
                    indivScore.append(combScores[e])
            already.append(q[2])
            finalScores.append(indivScore)

    if newTest[0][1] < globalThreshOriginal:
        category.append('Unidentified')
        f.write('Unidentified \n')
        unCount += 1
    elif newComb[0][1] < thresh:
        category.append('Several Possible Carriers')
        mulCount += 1
        f.write('Several Possible Carriers \n')
        f.write('Listed from highest to least ranked, these are: \n')
        for u in finalScores:
            if u[2] > globalThreshOriginal:
                f.write(u[1] + ' (' + u[0] + ')\n')

    else:
        category.append('Assigned')
        assignCount += 1
        allMols = [z[2] for z in newTest]
        allSmiles = [z[0] for z in newTest]
        f.write('Assigned to: ' + allMols[0] + ' (' + allSmiles[0] + ') \n')
    f.write('\n')
    f.write('All Candidate Transitions:')
    indivTest = allIndexTest[i]
    indivReport = allReports[i]
    for n in range(len(indivTest)):
        f.write('\n')
        f.write('Molecule: ' + str(indivTest[n][2]) + ' (' + str(indivTest[n][0]) + ')\n')
        f.write('Global Score: ' + str(indivTest[n][1]))
        f.write('\n')
        if len(indivReport[n]) == 0:
            f.write('No issues with this line \n')
        else:
            f.write('Issues with this line: ')
            for p in indivReport[n]:
                f.write(str(p))
            f.write('\n')

    f.write('\n')
    f.write('\n')
    f.write('------------------------------------------------------------\n')

f.write('\n')
f.write('Final Report\n')
f.write('Total number of lines: ' + str(len(newTestingScoresListFinal)))
f.write('\n')
f.write('Number of lines assigned to a single carrier: ' + str(assignCount))
f.write('\n')
f.write('Number of lines with more than one possible carrier: ' + str(mulCount))
f.write('\n')
f.write('Number of unassigned lines: ' + str(unCount))
f.write('\n')


analysisTock = time.perf_counter()
scrapeMins = (analysisTock-analysisTick)/60
scrapeMins2 = "{{:.{}f}}".format(2).format(scrapeMins)
print('')
print('Line assignment took ' + str(scrapeMins2) + ' minutes.')
print('')
print('Creating interactive output now.')


'''
The remainder of the code creates the interactive html output. It predominately uses Plotly figures
for interactivity. 
'''

data = molsim.file_handling.load_obs(specPath, type='txt')
ll0, ul0 = molsim.functions.find_limits(data.spectrum.frequency)
freq_arr = data.spectrum.frequency
int_arr = data.spectrum.Tb
resolution = data.spectrum.frequency[1] - data.spectrum.frequency[0]
ckm = (scipy.constants.c * 0.001)
min_separation = resolution * ckm / np.amax(freq_arr)
peak_indices = peak_indices_original
peak_freqs_new = data.spectrum.frequency[peak_indices]
peak_ints = abs(data.spectrum.Tb[peak_indices])
peak_ints_new = abs(data.spectrum.Tb[peak_indices])
peak_freqs_vlsr = []

for i in peak_freqs_new:
    off = i * vlsr_value / 299792
    newFreq = i + off
    peak_freqs_vlsr.append(newFreq)

peak_freqs_vlsr = np.array(peak_freqs_vlsr)
peak_freqs = np.array(peak_freqs_vlsr)

peak_freqs_og = peak_freqs

with open(os.path.join(direc, 'testing_list.pkl'), "rb") as fp:  # Unpickling
    newTestingScoresListFinal = pickle.load(fp)

with open(os.path.join(direc, 'combined_list.pkl'), "rb") as fp:  # Unpickling
    newCombinedScoresList = pickle.load(fp)

artFreqs = []
artInts = []
hoverArt = []

for i in range(len(peak_freqs)):
    if peak_freqs[i] in artifactFreqs or inAddedArt(peak_freqs[i], added_art) == True:
        artFreqs.append(peak_freqs_new[i])
        artInts.append(peak_ints_new[i])
        hoverArt.append(f'Artifact<br>Frequency: {peak_freqs[i]:.5f}<br>Intensity: {peak_ints[i]:.5f}')

peak_indices3 = []
peak_ints3 = []
peak_freqs3 = []

peak_indices3New = []
peak_ints3New = []
peak_freqs3New = []



for i in range(len(peak_indices)):
    if peak_freqs[i] not in artifactFreqs and inAddedArt(peak_freqs[i], added_art) == False and peak_freqs[i] not in noCanFreq:
        peak_ints3.append(peak_ints[i])
        peak_indices3.append(peak_indices[i])
        peak_freqs3.append(peak_freqs[i])

        peak_ints3New.append(peak_ints_new[i])
        peak_indices3New.append(peak_indices[i])
        peak_freqs3New.append(peak_freqs_new[i])


peak_indices = peak_indices3
peak_ints = peak_ints3
peak_freqs = peak_freqs3


peak_freqs_new = peak_freqs3New
peak_ints_new = peak_ints3New


combPeaks = [(peak_freqs[i], peak_ints[i]) for i in range(len(peak_freqs))]
sortedCombPeaks = sortTupleArray(combPeaks)
sortedCombPeaks.reverse()
spectrum_freqs = np.array([i[0] for i in sortedCombPeaks])
spectrum_ints = np.array([i[1] for i in sortedCombPeaks])

combPeaksNew = [(peak_freqs_new[i], peak_ints[i]) for i in range(len(peak_freqs_new))]
sortedCombNew = sortTupleArray(combPeaksNew)
sortedCombNew.reverse()
spectrum_freqs_new = np.array([i[0] for i in sortedCombNew])
spectrum_ints_new = np.array([i[1] for i in sortedCombNew])



unFreqs = []
unInts = []
hoverTextUn = []

mulFreqs = []
mulInts = []
hoverTextMul = []

assignFreqs = []
assignInts = []
hoverTextAssign = []

dfAssign = []
addLater = []
dfConf = []

allAssigned = {}
allFound = {}
maxRanks = {}
allAssignedInts = {}

for i in range(len(newTestingScoresListFinal)):
    indivCarriers = []
    newTest = newTestingScoresListFinal[i]
    newComb = newCombinedScoresList[i]
    allMols = [z[2] for z in newTest]
    allSmiles = [z[0] for z in newTest]
    combMol = [z[0][1] for z in newComb]
    combScores = [z[1] for z in newComb]
    finalScores = []
    already = []
    for q in newTest:
        if q[2] not in already:
            indivScore = []
            indivScore.append(q[0])
            indivScore.append(q[2])
            indivScore.append(q[1])
            for e in range(len(combMol)):
                if combMol[e] == q[2]:
                    indivScore.append(combScores[e])
            already.append(q[2])
            finalScores.append(indivScore)

    if newTest[0][1] < globalThreshOriginal:
        unFreqs.append(spectrum_freqs_new[i])
        unInts.append(spectrum_ints_new[i])
        dfAssign.append('Unassigned')
        dfConf.append('NA')
        hoverTextUn.append(f'Unassigned<br>Frequency: {spectrum_freqs_new[i]:.5f}<br>Intensity: {spectrum_ints_new[i]:.5f}')
    elif newComb[0][1] < thresh:
        mulFreqs.append(spectrum_freqs_new[i])
        mulInts.append(spectrum_ints_new[i])
        dfAssignText = 'Multiple Possible Carriers: '
        dfConf.append('NA')

        mulIndivText = f'Multiple possible carriers! These are: <br>'
        alreadyListed = []
        for d in range(len(newTest)):
            if newTest[d][1] > globalThreshOriginal and (newTest[d][0], newTest[d][2]) not in alreadyListed:
                mulIndivText = mulIndivText + newTest[d][2] + ' (' + newTest[d][0] + f') <br>'
                dfAssignText += newTest[d][2] + ' (' + newTest[d][0] + '), '
                alreadyListed.append((newTest[d][0], newTest[d][2]))
                addLater.append([newTest[d][2], spectrum_freqs[i], spectrum_ints[i]])

        dfAssignText = dfAssignText[:-2]

        dfAssign.append(dfAssignText)

        mulIndivText = mulIndivText + f'Frequency: {spectrum_freqs_new[i]:.5f}<br>Intensity: {spectrum_ints_new[i]:.5f}'
        hoverTextMul.append(mulIndivText)
    else:
        assignFreqs.append(spectrum_freqs_new[i])
        assignInts.append(spectrum_ints_new[i])
        topMol = newTest[0]
        dfAssign.append(topMol[2] + ' (' + topMol[0] + ')')
        dfConf.append(f'{topMol[1]:.3f}')
        textIndiv = 'Assigned to ' + topMol[2] + ' (' + topMol[
            0] + f')<br>Frequency: {spectrum_freqs_new[i]:.5f}<br>Intensity: {spectrum_ints_new[i]:.5f}'

        topMolForm = topMol[2]
        if topMolForm in totalForms_local:
            localIdx = totalForms_local.index(topMolForm)
            indivList = 'local'
            indivTag = totalTags_local[localIdx]

        else:
            localIdx = totalForms.index(topMolForm)
            indivList = 'JPL'
            indivTag = totalTags[localIdx]

        maxInt2, molRank2, closestFreq2 = checkIntensityFinal3(indivTag, indivList, topMolForm, spectrum_ints[i],
                                                         spectrum_freqs[i])
        hoverTextAssign.append(textIndiv)
        if topMol[2] not in allAssigned:
            allAssigned[topMol[2]] = [spectrum_freqs[i]]
            allAssignedInts[topMol[2]] = [spectrum_ints[i]]
            allFound[topMol[2]] = [closestFreq2]
            maxRanks[topMol[2]] = molRank2

        else:
            updatedList = allAssigned[topMol[2]]
            updatedList.append(spectrum_freqs[i])
            allAssigned[topMol[2]] = updatedList

            updatedListInts = allAssignedInts[topMol[2]]
            updatedListInts.append(spectrum_ints[i])
            allAssignedInts[topMol[2]] = updatedListInts

            updatedFound = allFound[topMol[2]]
            updatedFound.append(closestFreq2)
            allFound[topMol[2]] = updatedFound

            if molRank2 > maxRanks[topMol[2]]:
                maxRanks[topMol[2]] = molRank2

allAssignFreqs = assignFreqs + mulFreqs + artFreqs

unLineFreqs = []
unLineInts = []

count100 = 0
for u in freq_arr:
    nearby = False
    for q in allAssignFreqs:
        if abs(u - q) <= 1:
            nearby = True

    if nearby == True:
        unLineInts.append(0)
    else:
        unLineInts.append(int_arr[count100])

    count100 += 1

figUn = go.Figure()
figUn.add_trace(go.Scatter(x=freq_arr, y=unLineInts, mode='lines', name='Spectrum'))
figUn.update_layout(title='Inputted Spectrum with Assigned Lines and Artifacts Removed', xaxis_title='Frequency (MHz)',
                    yaxis_title='Intensity (arb.)', height=500)

dfFinalList = pd.DataFrame()
dfFinalList['Peak Frequency'] = spectrum_freqs_new
dfFinalList['Peak Intensity'] = spectrum_ints_new
dfFinalList['Assignment'] = dfAssign
dfFinalList['Confidence Score'] = dfConf
dfFinalList.to_csv(os.path.join(direc, 'final_assignment_table.csv'))

unIdxList = []
for v in range(len(dfAssign)):
    if 'Unassigned' in dfAssign[v]:
        unIdxList.append(v)

dfFinalUn = dfFinalList.iloc[unIdxList]

# Creating Plotly figures for output
fig = go.Figure()
fig.add_trace(go.Scatter(x=freq_arr, y=int_arr, mode='lines', name='Spectrum'))
peak_labels = [f'Peak {peak}' for peak in peak_indices]
fig.add_trace(go.Scatter(x=artFreqs, y=artInts, mode='markers', marker=dict(color='orange', size=8), hovertext=hoverArt,
                         hoverinfo='text', name='Artifacts'))
fig.add_trace(go.Scatter(x=unFreqs, y=unInts, mode='markers', marker=dict(color='red', size=8), hovertext=hoverTextUn,
                         hoverinfo='text', name='Unassigned lines'))
fig.add_trace(
    go.Scatter(x=mulFreqs, y=mulInts, mode='markers', marker=dict(color='yellow', size=8), hovertext=hoverTextMul,
               hoverinfo='text', name='Lines with multiple possible carriers'))
fig.add_trace(go.Scatter(x=assignFreqs, y=assignInts, mode='markers', marker=dict(color='green', size=8),
                         hovertext=hoverTextAssign, hoverinfo='text', name='Uniquely assigned lines'))

fig.update_layout(
    title='Spectrum with assignments (green for uniquely assigned, red for unassigned, yellow for multiple possible candidates)',
    xaxis_title='Frequency (MHz)', yaxis_title='Intensity (arb.)', height=500)
startingText = 'Initial Precursor Molecules: '
for p in startingMols:
    if p != startingMols[-1]:
        startingText = startingText + p + ', '
    else:
        startingText = startingText + p


text_content = """
<h2>AMASE</h2>

"""

text_content = text_content + '<p>' + startingText + '</p>'

items = [fig, dfFinalList, figUn, dfFinalUn]

html_content = text_content

html_content += f'<div style="padding: 10px;">{fig.to_html()}</div>'

html_content += f'<h2>Table of Line Assignments</h2>'

fig = go.Figure(data=[go.Table(
    header=dict(values=list(dfFinalList.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[dfFinalList[col] for col in dfFinalList.columns],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(
    title='',
    autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
    height=500
)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    updatemenus=None
)

html_content += f'<div style="padding: 10px; width: 100%;">{fig.to_html()}</div>'  # Set width to 100%
html_content += f'<div style="padding: 10px;">{figUn.to_html()}</div>'
html_content += f'<h2>Table of Unassigned Lines</h2>'

fig = go.Figure(data=[go.Table(
    header=dict(values=list(dfFinalUn.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[dfFinalUn[col] for col in dfFinalUn.columns],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(
    title='',
    autosize=True,
    margin=dict(l=0, r=0, t=0, b=0),
    height=500
)

fig.update_layout(
    xaxis_rangeslider_visible=False,
    updatemenus=None
)

html_content += f'<div style="padding: 10px; width: 100%;">{fig.to_html()}</div>'  # Set width to 100%

html_content += f'<h2>All Assigned Molecules</h2>'
html_content += f'<p> In the following plots, the simulated spectrum of the molecule is shown in red. All lines are scaled linearly such that the simulated intensity matches the observed value for the strongest observed line. Lines with green markers on top are those that were assigned to the displayed molecule by the algorithm. </p>'

int_arr_noArt = []

count100 = 0
for u in freq_arr:
    nearby = False
    for q in peak_freqs_og:
        if abs(u - q) <= 0.75:
            if q in artifactFreqs or inAddedArt(q, added_art) == True:
                nearby = True

    if nearby == True:
        int_arr_noArt.append(0)
    else:
        int_arr_noArt.append(int_arr[count100])

    count100 += 1

int_arr_noArt = np.array(int_arr_noArt)

for q in addLater:
    if q[0] in allAssigned:
        updatedList = allAssigned[q[0]]
        updatedInts = allAssignedInts[q[0]]
        updatedList.append(q[1])
        updatedInts.append(q[2])
        allAssigned[q[0]] = updatedList
        allAssignedInts[q[0]] = updatedInts


for i in allAssigned:
    stringValue = 'All observed lines of this molecule (MHz): '
    for z in allAssigned[i]:
        if z != allAssigned[i][-1]:
            stringValue = stringValue + f'{z:.2f}, '
        else:
            stringValue = stringValue + f'{z:.2f}'
    html_content = html_content + '<h2>' + str(i) + '</h2>'
    html_content = html_content + '<p>' + stringValue + '</p2>'
    topObservedLine = allAssigned[i][0]
    topObservedLineInt = allAssignedInts[i][0]

    listOfAssignedPeaks = allAssigned[i]
    listOfAssignedInts = allAssignedInts[i]
    hoverTextFinal = []

    topMolForm = i
    if topMolForm in totalForms_local:
        localIdx = totalForms_local.index(topMolForm)
        indivList = 'local'
        indivTag = totalTags_local[localIdx]

    else:
        localIdx = totalForms.index(topMolForm)
        indivList = 'JPL'
        indivTag = totalTags[localIdx]

    scaledFreqs, scaledInts, stickFreqs, stickInts = checkIntensityOutput(indivTag, indivList, topMolForm,
                                                                          topObservedLineInt, topObservedLine, ll0[0],
                                                                          ul0[-1])


    for h in range(len(listOfAssignedInts)):
        hoverTextFinal.append(f'Frequency: {listOfAssignedPeaks[h]:.5f}<br>Intensity: {listOfAssignedInts[h]:.5f}')
    figNew = go.Figure()
    figNew.add_trace(go.Scatter(x=freq_arr, y=int_arr_noArt, mode='lines', name='Spectrum'))
    figNew.add_trace(go.Scatter(x=stickFreqs, y=stickInts, mode='lines', name=i))
    figNew.add_trace(
        go.Scatter(x=listOfAssignedPeaks, y=listOfAssignedInts, mode='markers', marker=dict(color='green', size=8),
                   hovertext=hoverTextFinal,
                   hoverinfo='text', name='Lines Assigned to ' + str(i)))

    figNew.update_layout(xaxis_title='Frequency (MHz)', yaxis_title='Intensity (arb.)', height=500)
    html_content += f'<div style="padding: 10px;">{figNew.to_html()}</div>'

# Write and save HTML file
with open(os.path.join(direc, 'interactive_output.html'), 'w') as f:
    f.write(html_content)



print('Thank you for using this software! An interactive output (titled interactive_output.html) and a detailed line-by-line output (titled output_report.txt) are saved to your requested directory. Please send any questions/bugs to zfried@mit.edu')
