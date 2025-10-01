"""
Miscellaneous utilities for AMASE algorithm.
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # turn off RDKit warning message
import numpy as np
import math
import warnings
from fastdist import fastdist
from mol2vec import features
from gensim.models import word2vec
from mol2vec.features import sentences2vec
warnings.filterwarnings("ignore")
import numpy as np
from datetime import datetime



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def find_nearest(arr,val):
	idx = np.searchsorted(arr, val, side="left")
	if idx > 0 and (idx == len(arr) or math.fabs(val - arr[idx-1]) \
		 < math.fabs(val - arr[idx])):
		return idx-1
	else:
		return idx


def closest(lst, K):
    '''
    Function to find the closest value to an input value (K) is a list (lst)
    '''
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx, lst[idx]

def near_whole(number):
    # Check if the absolute difference between the number and its rounded value is less than or equal to 0.05
    return abs(number - round(number)) <= 0.05


def sortTupleArray(tup):
    '''
    Function to sort arrays of tuples
    '''
    tup.sort(key=lambda x: x[1])
    return tup



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



def euclidean_distance(vec1, vec2):
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        float: Euclidean distance
    """
    return np.linalg.norm(vec1 - vec2)



def gaussian(x, amp, mu, sigma):
    '''
    Define a Gaussian function
    '''
    return amp * np.exp(-0.5 * ((x - mu)/sigma)**2)





def deleteDuplicates(lst):
    '''
    Function to delete duplicate entries in a list
    '''
    seen = {}
    pos = 0
    for item in lst:
        if item not in seen:
            seen[item] = True
            lst[pos] = item
            pos += 1
    del lst[pos:]

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

def getCountDictionary(direc):
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

def getFeatureVector(direc, smile):
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

def addToGraph(addSmile, edges, smiles, countDict, allVectors, vectorSmiles, direc):
    '''
    Function that adds a molecule to the graph.
    '''
    indivVector = np.array(getFeatureVector(direc, addSmile))
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


def initial_banner():
    # Banner ASCII (left-aligned)
    banner = [
        "╔══════════════════════════════════════════════════════════╗",
        "║                         AMASE                            ║",
        "║                      Version 3.1                         ║",
        "╠══════════════════════════════════════════════════════════╣",
        "║                                                          ║",
        "║                        Greetings!                        ║",
        "║         This code will help you automatically            ║",
        "║             identify molecules in mixtures               ║",
        "║          measured with rotational spectroscopy           ║",
        "║                                                          ║", 
        "║     If you have any questions, issues, or feedback       ║",
        "║            please email zfried@mit.edu                   ║",
        "║                                                          ║",
       f"║             Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                 ║",
        "╚══════════════════════════════════════════════════════════╝",
    ]

    # Star ASCII (will be centered on banner)
    star = [
        "             .        *        .",
        "   *                 .             *",
        "        .    *    ✦    *    .",
        "   *                 .             *",
        "             .        *        .",
    ]

    # Banner width (assumes all lines are same length)
    banner_width = len(banner[0])

    # Print stars centered on banner
    for line in star:
        padding = max((banner_width - len(line)) // 2, 0)
        print(" " * padding + line)

    print("\n")

    # Print left-aligned banner
    for line in banner:
        print(line)

