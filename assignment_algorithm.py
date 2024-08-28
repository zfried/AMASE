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
from astroquery.splatalogue import Splatalogue
from astropy import units as u
from astropy import constants as c
import requests
import lxml.html as html
from lxml import etree
import pubchempy as pcp
from fastdist import fastdist
from astroquery.linelists.cdms import CDMS
from astroquery.jplspec import JPLSpec
import mol2vec
from mol2vec import features
from mol2vec import helpers
import gensim
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg
import pickle
warnings.filterwarnings("ignore")
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
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



def checkIntensityForward(tag, linelist, molForm, line_int, molFreq, loopIter):
    '''
    Function that checks whether the simulated line intensity is reasonable enough for assignment.
    '''

    thick = False

    if linelist == 'local':
        formIdx = totalForms_local.index(molForm)
        indexStr = str(formIdx)

        individualDF = pd.read_csv(os.path.join(pathLocal, indexStr + '.csv'))
        freqs = list(individualDF['frequencies'])
        peak_ints = np.array(list(individualDF['intensities']))
    else:
        if linelist == "CDMS" or linelist == "JPL":
            formIdx = totalTags.index(int(tag))
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


    thick = False

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

    return maxInt, molRank, closestFreq, intValue, thick


def checkIntensity(tag, linelist, molForm, line_int, molFreq):
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
               '36S', '(36)S', 'S36']
    for iso in isoList:
        if iso in mol:
            isotopologue += 1

    if "C13C" in mol:
        isotopologue = isotopologue - 1

    return isotopologue


def Get_Catalog(StartFrequency, StopFrequency):
    '''
    Function to query Splatalogue for all transitions in a certain range.
    '''
    Startf = StartFrequency * u.MHz
    Stopf = StopFrequency * u.MHz
    StopL = c.c / Startf
    StartL = c.c / Stopf

    URL = "https://splatalogue.online/splata-slap/slap?REQUEST=queryData&WAVELENGTH=%.8f/%.8f" % (
    StopL.to("m").value, StartL.to("m").value)

    page = requests.get(URL)
    Catalog = etree.fromstring(page.content)
    Table = Catalog[0].find("TABLE")
    Fields = []
    DataTypes = []
    FieldList = Table.findall("FIELD")
    for element in FieldList:
        Fields.append(element.attrib["ID"])
        DataTypes.append(element.attrib["datatype"])

    LineTree = Table.find("DATA").find("TABLEDATA").findall("TR")
    LineData = []
    for i, element in enumerate(LineTree):
        DataTree = element.findall("TD")
        CurrentData = []
        for j, subelement in enumerate(DataTree):
            if (DataTypes[j] == "double"):
                if (subelement.text is None):
                    CurrentData.append(np.nan)
                else:
                    CurrentData.append(float(subelement.text))
            elif (DataTypes[j] == "int"):
                if (subelement.text is None):
                    CurrentData.append(np.nan)
                else:
                    CurrentData.append(int(subelement.text))
            elif (DataTypes[j] == "boolean"):
                if (subelement.text is None):
                    CurrentData.append(np.nan)
                else:
                    CurrentData.append(bool(subelement.text))
            else:
                CurrentData.append(subelement.text)
        LineData.append(CurrentData)

    DataFrame = pd.DataFrame(LineData, columns=Fields)
    return DataFrame


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

def getCatCDMS(tag, saveNum):
    '''
    This function scrapes a .cat file from CDMS,
    simulates it at the experimental temperature, and
    stores the simulated peak frequencies and intensities in csv file.
    '''

    try:
        tag = int(tag)
        strTag = str(tag)
        strSave = str(saveNum)
        if tag >= 100000:
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c{tag}.cat"
        if tag >= 0 and tag < 100000:
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c0{tag}.cat"
        if tag < 0 and tag > -100000:
            tag2 = -1 * tag
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c0{tag2}.cat"
        if tag <= -100000:
            tag2 = -1 * tag
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c{tag2}.cat"
        response = requests.get(api_url)
        # print(response.text)
        isValid = False
        if 'Not Found' not in response.text:
            isValid = True
            with open(os.path.join(pathSplatCat, strSave + '.cat'), "w+") as f:
                f.write(response.text)

            dfFreq = pd.DataFrame()
            q = os.path.join(pathSplatCat, strSave + '.cat')
            mol = molsim.file_handling.load_mol(q, type='SPCAT')
            minFreq = ll0
            maxFreq = ul0
            if 'y' in astro or 'Y' in astro:
                observatory1 = molsim.classes.Observatory(dish=dishSize)
                observation1 = molsim.classes.Observation(observatory=observatory1)
                src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=dv_value)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014, observation=observation1)
            else:
                src = molsim.classes.Source(Tex=temp, column=1.E9)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014)
            peak_freqs2 = sim.spectrum.frequency
            peak_ints2 = sim.spectrum.Tb
            if peak_ints is not None:
                freqs = list(peak_freqs2)
                ints = list(peak_ints2)
                dfFreq['frequencies'] = freqs
                dfFreq['intensities'] = ints
                saveName = os.path.join(pathSplat, strSave + '.csv')
                dfFreq.to_csv(saveName)
        return isValid
    except:
        return False



def getCatCDMSCheck(tag, saveNum):
    '''
    This function scrapes a .cat file from CDMS,
    simulates it at the experimental temperature, and
    stores the simulated peak frequencies and intensities in csv file.
    '''

    try:
        tag = int(tag)
        strTag = str(tag)
        strSave = str(saveNum)
        if tag >= 100000:
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c{tag}.cat"
        if tag >= 0 and tag < 100000:
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c0{tag}.cat"
        if tag < 0 and tag > -100000:
            tag2 = -1 * tag
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c0{tag2}.cat"
        if tag <= -100000:
            tag2 = -1 * tag
            api_url = f"https://cdms.astro.uni-koeln.de/classic/entries/c{tag2}.cat"
        response = requests.get(api_url)
        # print(response.text)
        isValid = False
        if 'Not Found' not in response.text:
            isValid = True
            with open(os.path.join(pathSplatCat, strSave + '.cat'), "w+") as f:
                f.write(response.text)

            dfFreq = pd.DataFrame()
            q = os.path.join(pathSplatCat, strSave + '.cat')
            mol = molsim.file_handling.load_mol(q, type='SPCAT')
            minFreq = ll0
            maxFreq = ul0
            if 'y' in astro or 'Y' in astro:
                observatory1 = molsim.classes.Observatory(dish=dishSize)
                observation1 = molsim.classes.Observation(observatory=observatory1)
                src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=dv_value)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014, observation=observation1)
            else:
                src = molsim.classes.Source(Tex=temp, column=1.E9)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014)
            peak_freqs2 = sim.spectrum.frequency
            peak_ints2 = sim.spectrum.Tb
            if peak_ints is not None:
                freqs = list(peak_freqs2)
                ints = list(peak_ints2)
            else:
                freqs = []
                ints = []
        return freqs, ints

    except:
        freqs = []
        ints = []
        return freqs, ints


def getCatJPLCheck(tag, saveNum):
    try:
        tag = int(tag)
        strTag = str(tag)
        strSave = str(saveNum)
        if tag >= 100000:
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c' + strTag + '.cat'
        if tag >= 0 and tag < 100000:
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c0' + strTag + '.cat'
        if tag < 0 and tag > -100000:
            tag = -1 * tag
            strTag = str(tag)
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c0' + strTag + '.cat'
        if tag <= -100000:
            tag = -1 * tag
            strTag = str(tag)
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c' + strTag + '.cat'

        response1 = requests.get(url)
        responseText1 = str(response1.text)
        isValid = False
        if 'Not Found' not in responseText1:
            isValid = True
            with open(os.path.join(pathSplatCat, strSave + '.cat'), "w+") as f:
                f.write(responseText1)

            dfFreq = pd.DataFrame()
            q = os.path.join(pathSplatCat, strSave + '.cat')
            mol = molsim.file_handling.load_mol(q, type='SPCAT')
            minFreq = ll0
            maxFreq = ul0
            if 'y' in astro or 'Y' in astro:
                observatory1 = molsim.classes.Observatory(dish=dishSize)
                observation1 = molsim.classes.Observation(observatory=observatory1)
                src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=dv_value)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014, observation=observation1)
            else:
                src = molsim.classes.Source(Tex=temp, column=1.E9)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014)
            peak_freqs2 = sim.spectrum.frequency
            peak_ints2 = sim.spectrum.Tb
            if peak_ints is not None:
                freqs = list(peak_freqs2)
                ints = list(peak_ints2)
            else:
                freqs = []
                ints = []

        return freqs, ints

    except:
        freqs = []
        ints = []
        return freqs, ints





def getCatJPL(tag, saveNum):
    '''
    This function scrapes a .cat file from JPL database,
    simulates it at the experimental temperature, and
    stores the simulated peak frequencies and intensities in csv file.
    '''
    try:
        tag = int(tag)
        strTag = str(tag)
        strSave = str(saveNum)
        if tag >= 100000:
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c' + strTag + '.cat'
        if tag >= 0 and tag < 100000:
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c0' + strTag + '.cat'
        if tag < 0 and tag > -100000:
            tag = -1 * tag
            strTag = str(tag)
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c0' + strTag + '.cat'
        if tag <= -100000:
            tag = -1 * tag
            strTag = str(tag)
            url = 'https://spec.jpl.nasa.gov/ftp/pub/catalog/c' + strTag + '.cat'

        response1 = requests.get(url)
        responseText1 = str(response1.text)
        isValid = False
        if 'Not Found' not in responseText1:
            isValid = True
            with open(os.path.join(pathSplatCat, strSave + '.cat'), "w+") as f:
                f.write(responseText1)

            dfFreq = pd.DataFrame()
            q = os.path.join(pathSplatCat, strSave + '.cat')
            mol = molsim.file_handling.load_mol(q, type='SPCAT')
            minFreq = ll0
            maxFreq = ul0
            if 'y' in astro or 'Y' in astro:
                observatory1 = molsim.classes.Observatory(dish=dishSize)
                observation1 = molsim.classes.Observation(observatory=observatory1)
                src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=dv_value)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014, observation=observation1)
            else:
                src = molsim.classes.Source(Tex=temp, column=1.E9)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014)
            peak_freqs2 = sim.spectrum.frequency
            peak_ints2 = sim.spectrum.Tb
            if peak_ints is not None:
                freqs = list(peak_freqs2)
                ints = list(peak_ints2)
                dfFreq['frequencies'] = freqs
                dfFreq['intensities'] = ints
                saveName = os.path.join(pathSplat, strSave + '.csv')
                dfFreq.to_csv(saveName)
        return isValid
    except:
        return False


def checkAllLines(linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms):
    '''
    This function checks whether at least half of the predicted 10 sigma lines
    of the molecular candidate are present in the spectrum. If at least half arent
    present, rule_out = True is returned
    '''
    if linelist == 'local':
        catIdx = totalForms_local.index(formula)
        idxStr = str(catIdx)
        df = pd.read_csv(os.path.join(pathLocal, idxStr + '.csv'))

    elif linelist == "CDMS" or linelist == "JPL":
        catIdx = totalTags.index(int(tag))
        idxStr = str(totalIndices[catIdx])
        df = pd.read_csv(os.path.join(pathSplat, idxStr + '.csv'))

    else:
        catIdx = totalForms.index(formula)
        idxStr = str(totalIndices[catIdx])
        df = pd.read_csv(pathSplat, idxStr + '.csv')

    closestActualIdx, closestActualFreq = closest(peak_freqs_full, freq)
    line_int_full = peak_ints_full[closestActualIdx]

    sim_freqs = list(df['frequencies'])
    sim_ints = np.array(list(df['intensities']))
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

    '''
    filteredCombList = []
    rms_comb_list = []
    for comb1 in combList:
        clo = find_closest(rms_dict_values, comb1[0])
        rms_val = rms_dict[rms_dict_values[clo]]
        if comb1[1] > 10*rms_val:
            filteredCombList.append(comb1)
    '''

    filteredCombList = [i for i in combList if i[1] > 10 * rms]

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


    return rule_out



def checkIso(formula, linelist, tag, line_int_value):
    print(formula,linelist, tag)
    hasCandidate = False

    for p in range(len(mol_par)):
        if mol_par[p] == formula and list_par[p] == linelist:
            par = mol_parent[p]
            li = list_parent[p]
            ta = tag_parent[p]
            break

    print(par, li, ta)
    print(line_int_value)
    fullCats = pd.read_csv(os.path.join(pathSplatCat, 'catalog_list.csv'))
    fullCatForms = list(fullCats['formula'])
    fullCatLists = list(fullCats['linelist'])
    fullCatIdx = list(fullCats['idx'])
    foundPar = False

    for p in range(len(fullCatForms)):
        if fullCatForms[p] == par and fullCatLists[p] == li:
            foundPar = True
            idx = fullCatIdx[p]
            break

    if foundPar == True:
        par_spec_csv = pd.read_csv(os.path.join(pathSplat, str(idx) + '.csv'))
        par_freq = list(par_spec_csv['frequencies'])
        par_int = list(par_spec_csv['intensities'])

        for p in range(len(par_int)):
            if par_int[p] >= line_int_value/10:
                print(par_int[p])
                print(par_freq[p])
                hasCandidate = True

    else:
        if li == 'CDMS':
            par_freq, par_int = getCatCDMSCheck(ta, par[0:2])
        elif li == 'JPL':
            par_freq,par_int = getCatJPLCheck(ta, par[0:2])

        for p in range(len(par_int)):

            if par_int[p] >= line_int_value/10:
                print(par_int[p])
                print(par_freq[p])
                hasCandidate = True

    print('')
    return hasCandidate


def runPageRankInit2(smiles, detectedSmiles, testSmiles, testSmilesIso, testFrequencies, correctFreq,
                     edges, countDict, oldHighestIntensities, intensity, forms, linelists, tags,
                     previous_best, quantum_nums, oldHighestSmiles, newCalc, sorted_dict_last, loopIter):
    '''
    This function runs the graph-based ranking system given the detected smiles.
    It then takes the molecular candidates for a given line and checks the frequency and
    intensity match (by calling several of the other functions). It then combines all of the
    scores (i.e. the graph, intensity, and frequency scores) and returns the sorted results for
    each  molecular candidate.
    '''

    # running graph calculation.
    reportListForward = []
    if newCalc == True and len(detectedSmiles) > 0:
        nodeDictInit = {}
        nodeDict = {}
        for smile in smiles:
            # initializing all weight on the detected molecules
            if smile in detectedSmiles:
                nodeDict[smile] = 10 * detectedSmiles[smile]
                nodeDictInit[smile] = 10 * detectedSmiles[smile]

            else:
                nodeDict[smile] = 0
                nodeDictInit[smile] = 0
        maxIt = 5000  # maximum number of possible loop interations
        for i in range(maxIt):
            #intermediateDict = {}
            intermediateDict = nodeDictInit.copy()
            # looping through the edges and updating the node weights
            for edge in edges:
                updateNode = edge[0]
                partner = edge[1]
                partnerCount = countDict[partner]
                if partnerCount != 0:
                    addedValue = nodeDict[partner] / (1.5 * partnerCount)
                else:
                    addedValue = 0
                    
                intermediateDict[updateNode] = intermediateDict[updateNode] + addedValue

            # checking if the scores have converged
            converged = True
            for z in intermediateDict:
                if abs(intermediateDict[z] - nodeDict[z]) > 1e-5:
                    converged = False
                    break

            nodeDict = intermediateDict

            if converged == True:
                break

        sorted_dict = sorted(nodeDict.items(), key=lambda x: x[1])
        sorted_dict.reverse()
    else:
        sorted_dict = sorted_dict_last



    # storing graph scores
    newSmiles = [z[0] for z in sorted_dict]
    newValues = [z[1] for z in sorted_dict]

    testingScores = []
    testingScoresFreq_Updated = []

    # looping through the molecular candidates for each line
    for idx in range(len(testSmiles)):
        newReport = []
        smile = testSmiles[idx]
        formula = forms[idx]
        linelist = linelists[idx]
        tag = tags[idx]
        iso = testSmilesIso[idx]
        freq = testFrequencies[idx]
        qn = quantum_nums[idx]

        # checking the intensity match
        maxInt, molRank, closestFreq, line_int_value, thick = checkIntensityForward(tag, linelist, formula, intensity, freq, loopIter)
        rule_out_val = checkAllLines(linelist, formula, tag, freq, peak_freqs_full, peak_ints_full, rms)


        #rule_out_val = False

        if correctFreq in sigmaDict:
            sigmaList = sigmaDict[correctFreq]
            sigmaList.append((formula,freq,rule_out_val))
            sigmaDict[correctFreq] = sigmaList
        else:
            sigmaDict[correctFreq] = [(formula,freq,rule_out_val)]


        #print(sigmaDict[correctFreq])


        hasInvalid = False
        mol = Chem.MolFromSmiles(smile)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in validAtoms:
                hasInvalid = True

        if len(detectedSmiles) > 0:
            if testSmiles[idx] in newSmiles:
                newIdx = newSmiles.index(testSmiles[idx])
                value = newValues[newIdx]
                per = stats.percentileofscore(newValues, value)
            else:
                value = 0
                per = 0.1
        else:
            value = 10
            per = 100

        tu = [smile, per, formula, qn, value, iso]

        if per < 93:
            newReport.append('Structural relevance score not great. ')

        # scaling the score based on the frequency match
        offset = freq - correctFreq
        scale = 1 - abs(offset / 5)
        scaledPer = scale * per
        if scale < 0.93:
            newReport.append('Frequency match not great.')

        # scaling the score based on the intensity match
        if maxInt > 6 * maxObservedInt:
            scaledPer = 0.5 * scaledPer
            newReport.append(
                'Intensity suggests that there should be unreasonably strong lines of this molecule in the spectrum.')

        if rule_out_val == True:
            scaledPer = 0.5 * scaledPer
            newReport.append('Too many of the simulated 10 sigma lines of this molecule are not present.')


        if molRank > 25 and formula not in previousBest:
            scaledPer = 0.5 * scaledPer
            newReport.append(
                'This is the strongest observed line of this molecule in the spectrum, but it is simulated to be the  number ' + str(
                    molRank) + ' strongest transition.')

        

        if formula in oldHighestIntensities:
            if maxInt > 5 * oldHighestIntensities[formula]:
                scaledPer = 0.5 * scaledPer
                newReport.append('The simulated relative intensities do not match with what is observed.')

        else:
            if maxInt > 5 * intensity:
                scaledPer = 0.5 * scaledPer
                newReport.append(
                    'This is the strongest observed line of this molecule in spectrum but is simulated to be too weak.')

        # scaling the score if the molecule is a rare isotopologue with an unrealistic intensity.
        if iso != 0:
            if smile not in oldHighestSmiles or intensity > (0.08 ** iso) * oldHighestSmiles[smile]:
                scaledPer = 0.5 * scaledPer
                newReport.append('Isotopologue is too strong.')

        # scaling score if the molecule contains an invalid atom
        if hasInvalid == True:
            scaledPer = 0.5 * scaledPer
            newReport.append('Contains an invalid atom.')

        tu2 = [smile, scaledPer, formula, thick, qn, value, iso]
        testingScoresFreq_Updated.append(tu2)

        testingScores.append(tu)

    # the next several lines combine and sort the resulting scores of the molecules
    testingScoresSort = sortTupleArray(testingScores)
    testingScoresSort.reverse()

    testingScoresFreqSort = sortTupleArray(testingScoresFreq_Updated)
    testingScoresFreqSort.reverse()

    percentiles = [z[1] for z in testingScoresFreqSort]
    soft = list(softmax(percentiles))

    for i in range(len(testingScoresFreqSort)):
        testingScoresFreqSort[i].append(soft[i])

    testingScoresDict = {}
    for e in testingScoresFreqSort:
        if (e[0], e[2]) not in testingScoresDict:
            testingScoresDict[(e[0], e[2])] = e[-1]
        else:
            currentValue = testingScoresDict[(e[0], e[2])]
            newValue = e[-1] + currentValue
            testingScoresDict[(e[0], e[2])] = newValue

    keys = list(testingScoresDict.keys())
    values = list(testingScoresDict.values())

    tuplesCombined = [(keys[i], values[i]) for i in range(len(keys))]
    sortedTuplesCombined = sortTupleArray(tuplesCombined)
    sortedTuplesCombined.reverse()

    testingScoresSmiles = [i[0] for i in testingScoresFreqSort]
    softScores = [i[-1] for i in testingScoresFreqSort]
    globalScores = [i[1] for i in testingScoresFreqSort]

    ranking = 0
    return testingScoresFreqSort, testingScoresSmiles, softScores, ranking, testingScoresSort, sorted_dict, globalScores, sortedTuplesCombined





def scaleScoreReverse(smile, validAtoms, subReport, sorted_smiles, sorted_values, freq, correctFreq, molRank, maxInt,
                      form, newPreviousBest, newHighestIntensities, intensityReverse, iso, qn, newHighestSmiles,
                      rule_out_reverse, line_int_value, linelist, tag):
    '''
    This function scales the score of a molecule based on its frequency and intensity match along with
    the molecular composition (i.e. invalid atoms or isotopologues).
    '''

    subReport = []

    # Checking if there is an invalid atom
    hasInvalid = False
    mol = Chem.MolFromSmiles(smile)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in validAtoms:
            hasInvalid = True
    if len(detectedSmiles) > 0:
        if smile in sorted_smiles:
            idx2 = sorted_smiles.index(smile)
            value = sorted_values[idx2]
            per = stats.percentileofscore(sorted_values, value)
        else:
            value = 0.1
            per = 0

    else:
        value = 10
        per = 100

    if per < 93:
        subReport.append('Structural relevance score not great. ')

    # scaling score based on frequency match
    offset = freq - correctFreq
    scale = 1 - abs(offset / 5)
    scaledPer = scale * per


    if scale < 0.93:
        subReport.append('Frequency match not great. ')

    # next several lines scale the score based on intensity match
    if maxInt > 6 * maxObservedInt:
        subReport.append(
            'Intensity suggests that there should be unreasonably strong lines of this molecule in the spectrum. ')
        scaledPer = 0.5 * scaledPer

    if rule_out_reverse == True:
        subReport.append('Too many of the simulated 10 sigma lines of this molecule are not present. ')
        scaledPer = 0.5 * scaledPer


    if molRank > 25 and form not in newPreviousBest:
        subReport.append(
            'This is the strongest observed line of this molecule in the spectrum, but it is simulated to be the number ' + str(
                molRank) + ' strongest transition. ')
        scaledPer = 0.5 * scaledPer
    
    
    if form in newHighestIntensities:
        if maxInt > 5 * newHighestIntensities[form]:
            subReport.append('The simulated relative intensities do not match with what is observed. ')
            scaledPer = 0.5 * scaledPer
    else:
        if maxInt > 5 * intensityReverse:
            subReport.append(
                'This is the strongest observed line of this molecule in spectrum but is simulated to be too weak. ')
            scaledPer = 0.5 * scaledPer

    # scaling score if unreasonably strong isotopologue
    if iso != 0:
        if smile not in newHighestSmiles or intensityReverse > (0.08 ** iso) * newHighestSmiles[smile]:
            subReport.append('Isotopologue is too strong.')
            scaledPer = 0.5 * scaledPer



    # scaling score if there's an invalid atom.
    if hasInvalid == True:
        subReport.append('Contains an invalid atom.')
        scaledPer = 0.5 * scaledPer

    tu2 = [smile, scaledPer, form, qn, value, iso]

    return tu2, subReport, scaledPer, offset, value, per


def getTopScores(newIndexTest):
    '''
    This function takes the scores for each molecular candidate for a given line and performs
    the necesary sorting and combining to give the final results.
    '''
    sortedNewTest = sortTupleArray(newIndexTest)
    sortedNewTest.reverse()

    percentiles = [z[1] for z in sortedNewTest]
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

    tuplesCombinedReverse = [(keys[i], values[i]) for i in range(len(keys))]
    sortedTuplesCombinedReverse = sortTupleArray(tuplesCombinedReverse)
    sortedTuplesCombinedReverse.reverse()

    bestSmile = sortedNewTest[0][0]
    bestGlobalScore = sortedNewTest[0][1]

    # storing the top scores and molecules
    for co in sortedTuplesCombinedReverse:
        if co[0][0] == bestSmile:
            bestMol = co[0][1]
            bestScore = co[1]
            break

    topGlobalScoreSecond = sortedNewTest[0][1]

    best = sortedTuplesCombinedReverse[0]

    return sortedNewTest, best, bestSmile, bestMol, bestScore, bestGlobalScore, testingScoresDictReverse, percentiles, soft, tuplesCombinedReverse, sortedTuplesCombinedReverse, topGlobalScoreSecond


def runPageRankInit2_Final(smiles, detectedSmiles, edges, countDict):
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


def updateOverride(bestGlobalScore, globalThresh, bestMol, override):
    '''
    This function updates the override counter. This ultimately provides the algorithm a way
    to override the scoring if there is compelling enough evidence for a molecule being present.
    For example, if a molecule is ranked fairly highly but below the required thresholds enough
    times, the algorithm will override its calculation and list the molecule as detected.

    '''
    if bestGlobalScore > 50 and bestGlobalScore < globalThresh:
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
        if sorted_test[1] > globalThresh:
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

    topSmile = testingScoresFinal[0][0]
    topGlobalScore = testingScoresFinal[0][1]

    for co in sortedTuplesCombined:
        if co[0][0] == topSmile:
            topScore = co[1]
            break

    for sorted_test in testingScoresFinal:
        if sorted_test[1] > globalThresh:
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

opticalScales = [1,5,10,20,30,40,50,60,70,80,90,100,150]
opticallyThick = []
sigmaDict = {}
# The following lines are required for querying CDMS efficiently
cdmsMolsFull = ['003501 HD', '004501 H2D+', '005501 HD2+', '005502 HeH+', '012501 C-atom', '012502 BH', '012503 C+',
                '013501 C-13', '013502 CH', '013503 CH+', '013504 CH+, v=1-0', '013505 CH+, v=2-0', '013506 C-13-+',
                '014501 CH2', '014502 C-13-H+', '014503 CD+', '014504 C-13-H+, v=1-0', '014505 CD+, v=1-0', '014506 N+',
                '015501 NH', '015502 C-13-D+', '015503 C-14-H+', '015504 CT+', '016501 NH2', '016502 ND',
                '016503 CH2D+', '016504 O-atom', '016505 N-15-H', '017501 OH+', '017502 OH-', '017503 CH3D',
                '017504 N-15-H2', '017505 CHD2+', '017506 NH3-wHFS', '017507 OH, v=0', '017508 OH, v=1',
                '017509 OH, v=2', '018501 NH2D', '018502 OD-', '018503 N-15-H3', '018504 C-13-H3D', '018505 H2O+',
                '018506 OD+', '018507 O-18-atom', '019501 NHD2', '019502 O-18-H-', '019503 H3O+', '019504 N-15-H2D',
                '019505 O-18-H+', '019506 NH3D+', '020501 ND3', '020502 D2O', '020503 H2DO+', '020504 N-15-HD2',
                '021501 NeH+', '021502 H2F+', '022501 NeD+', '023501 Ne-22-H+', '024501 NaH', '024502 LiOH',
                '025501 CCH, v=0', '025502 MgH', '025503 CCH, v2=1', '025504 C2H-', '025505 CCH, v2=2',
                '025506 CCH, v3=1', '025507 CCH, nu2', '025508 CCH, nu3', '025509 CCH, nu2+nu3', '025510 CCH, 5nu2',
                '026501 CCD', '026502 C-13-CH', '026503 CC-13-H', '026504 CN, v=0,1', '026505 C2H2, nu5-nu4',
                '026506 CN-', '026507 B-10-O', '026508 CN+,v=0', '026509 CN+,v=1-0', '027501 HCN, v=0',
                '027502 HNC, v=0', '027503 HCN, v2=1', '027504 HNC, v2=1', '027505 C-13-N', '027506 CN-15',
                '027507 HCN, v2=2', '027508 HCN, v2=3', '027509 HCN, v3=1', '027510 HCN, v1=1', '027511 HCCD',
                '027512 HB-10-O', '027513 BO', '027514 C2H3+', '027515 Al-atom', '028501 HC-13-N, v=0', '028502 H2CN',
                '028503 CO, v=0', '028504 HCNH+', '028505 C-13-N-15', '028506 HCN-15, v=0', '028507 HCN-15, v2=1',
                '028508 DNC', '028509 DCN, v=0', '028510 DCN, v2=1', '028511 HC-13-N, v2=1', '028512 CO, v=1-3',
                '028513 CO+, v=0', '028514 HBO', '028515 HNC-13', '028516 AlH', '028517 Si-atom', '028518 Si+',
                '028519 CO+, v=1', '028520 HC-13-N, v2=2', '028521 HC-13-N, v2=3', '028522 HC-13-N, v3=1',
                '028523 HC-13-N, v1=1', '028524 HCN-15, v2=2', '028525 HCN-15, v2=3', '028526 HCN-15, v3=1',
                '028527 HCN-15, v1=1', '028528 H2NC', '029501 C-13-O', '029502 HCND+', '029503 CO-17',
                '029504 HOC+, v2=0', '029505 HOC+, v2=1', '029506 N2H+, v=0; recommended', '029507 HCO+, v=0',
                '029508 HCO+, v2=1', '029509 N2H+, v2=1', '029510 DC-13-N', '029511 DCN-15', '029512 HC-13-N-15, v=0',
                '029513 HC-13-N-15, v2=1', '029514 HC-13-NH+', '029515 HCO+, nu2', '029516 N2H+, nu2', '029517 SiH',
                '029518 H2CNH', '029519 C-13-O+', '029520 C2H3D', '029521 SiH+, v=0', '029522 SiH+, v=1-0',
                '030501 H2CO', '030502 CO-18', '030503 C-13-O-17', '030504 HC-13-O+', '030505 HCO-17+', '030506 DOC+',
                '030507 N-15-NH+', '030508 NN-15-H+', '030509 N2D+', '030510 DCO+', '030511 DC-13-N-15',
                '030512 NO+, v=0,1', '030513 CO-18-+', '030514 H2C-13-NH', '030515 H2CND', '030516 H2CN-15-H',
                '030517 NO, v=0', '030518 NO, v=1', '030519 H2CNH2+', '030520 NO+, v=1-0, 2-1', '030521 NO+, v=2-0',
                '031501 HDCO', '031502 C-13-O-18', '031503 H2C-13-O', '031504 H2COH+', '031505 CF', '031506 HCO-18+',
                '031507 CF+, v=0,1', '031508 DC-13-O+', '031509 N-15-ND+', '031510 NN-15-D+', '031511 C-13-O-18+',
                '031512 N-15-O', '031513 NO-17', '031514 H2CO-17', '031515 H2COH', '031516 N-15-N-15-H+', '032501 PH',
                '032502 D2CO', '032503 H2CO-18', '032504 CH3OH, vt=0-2', '032505 DCO-18+', '032506 H2C-13-OH+',
                '032507 HDC-13-O', '032508 O2-X, v=0', '032509 C-13-F+, v=0,1', '032510 O2-a', '032511 S-atom',
                '032512 LiCCH', '032513 NO-18', '032514 N-15-O-17', '032515 N-15-N-15-D+', '033501 PH2',
                '033502 C-13-H3OH, vt=0,1', '033503 NH2OH', '033504 SH-', '033505 SH+', '033506 D2C-13-O',
                '033507 N-15-O-18', '033508 SH, v=0', '033509 SH, v=1', '033510 HDCO-18', '034501 PH3', '034502 H2S',
                '034503 OO-18', '034504 CH3O-18-H,v=0-2', '034505 D2CO-18', '034506 CHD2OH, vt=0', '035501 NaC',
                '035502 HDS', '035503 H2S-33', '035504 S-34-H+', '035505 CD3OH, vt=0', '035506 CD3OH, vt=1',
                '036501 NaCH', '036502 C3, nu2', '036503 D2S', '036504 H2S-34', '036505 (O-18)2', '036506 CD3OD, vt=0',
                '036507 CD3OD, vt=1', '037501 C3H, v4=0,1mS', '037502 Ar-36-H+', '037503 HDS-34', '037504 H2Cl+',
                '037505 C3H+', '038501 l-C3H2', '038502 C2N', '038503 C3D, v4=0,1mS', '038504 13C-CCH, v4=0,1mS',
                '038505 C-13C-CH, v4=0,1mS', '038506 CC-13C-H, v4=0,1mS', '038507 D2S-34', '038508 c-C3H2',
                '039501 HCCN', '039502 l-C-13-CCH2', '039503 l-CC-13-CH2', '039504 l-CCC-13-H2', '039505 H2C3H',
                '039506 l-C3HD', '039507 H2Cl-37-+', '039508 c-C3HD', '039509 c-C-13-C2H2', '039510 c-CCC-13-H2',
                '039511 Ar-38-H+', '039512 H2C3H+', '040501 SiC, v=0', '040502 CH3CCH', '040503 KH',
                '040504 CH3CCH, v10=1', '040505 H2CCN', '040506 c-C3H4', '040507 CH3CCH, nu10', '040508 CH3CCH, nu9',
                '040509 NaOH', '040510 HC-13-CN', '040511 HCC-13-N', '040512 DCCN', '040513 HCCN-15', '040514 c-C3D2',
                '040515 l-C3D2', '040516 c-C3H2, 13C1D', '040517 c-C3H2, 13C2D', '040518 c-C3H2, 13C3D',
                '040519 MgO, v=0-2', '040520 c-C3H2D+', '041501 CH3CCD', '041502 CH2DCCH', '041503 H2CCNH',
                '041504 Ar-40-H+', '041505 CH3CN, v=0', '041506 HCCO', '041507 HNCN', '041508 MgOH',
                '041509 CH3CN, v8=1', '041510 CH3CN, v8=2', '041511 CH3CN, nu8', '041512 CH3CN, 2nu8-nu8',
                '041513 CH3CN, 2nu8', '041514 CH3NC', '041515 CH3CC-13-H', '041516 CH3C-13-CH', '041517 C-13-H3CCH',
                '041518 CaH+', '041519 Mg-25-O, v=0,1', '042501 H2CCO', '042502 NaF, v=0,1', '042503 NCO',
                '042504 CH3CNH+', '042505 SiN', '042506 HNCNH', '042507 Ar-40-D+', '042508 C-13-H3CN, v=0',
                '042509 CH3C-13-N, v=0', '042510 CH3CN-15, v=0', '042511 CH2DCN', '042512 NCO-',
                '042513 C-13-H3CN, v8=1', '042514 CH3C-13-N, v8=1', '042515 CH3CN-15, v8=1', '042516 Propene',
                '042517 H2CSi', '042518 H2NNC', '042519 DCCO', '042520 Al-26-O, v=0', '042521 Al-26-O, v=1,2',
                '042522 CH3C-13-C-13-H', '042523 C-13-H3CC-13-H', '042524 C-13-H3C-13-CH', '042525 CH3CC-13-D',
                '042526 CH3C-13-CD', '042527 C-13-H3CCD', '042528 CH2DCC-13-H', '042529 CH2DC-13-CH',
                '042530 C-13-H2DCCH', '042531 CH2DCCD', '042532 CHD2CCH', '042533 Mg-26-O, v=0,1', '043501 CP',
                '043502 Ethylenimine', '043503 MgF', '043504 C2H3NH2, w.in 0+,-', '043505 H2CC-13-O',
                '043506 H2C-13-CO', '043507 HDC2O', '043508 C2H3NH2, 0-&lt;-0+', '043509 HCNO', '043510 HOCN',
                '043511 HNCO', '043512 HONC', '043513 C-13-H3C-13-N', '043514 CHD2CN', '043515 H2NC-13-N',
                '043516 HNSi', '043517 HDNCN', '043518 H2N-15-CN', '043519 H2NCN-15', '043520 AlO, v=0',
                '043521 AlO, v=1,2', '043522 AlO, v=3-5', '044501 CS, v=0-4', '044502 HCP, v=0', '044503 HCP, v2=1',
                '044504 Ethylene oxide', '044505 SiO, v=0-10', '044506 s-H2C=CHOH', '044507 a-H2C=CHOH',
                '044508 H2C2O-18', '044509 D2C2O', '044510 CS, v=1-0,2-1', '044511 CS, v=2-0', '044512 CS+',
                '044513 DCNO', '044514 HC-13-NO', '044515 HCN-15-O', '044516 H2NCO+', '044517 H2CSiH2',
                '044518 AlO-17, v=0', '044519 AlO-17, v=1,2', '044520 CD3CN', '045501 C-13-S, v=0,1',
                '045502 CS-33, v=0,1', '045503 DCP', '045504 Si-29-O, v=0-6', '045505 H2CP', '045506 HCS+',
                '045507 HCS', '045508 HSC', '045509 C-13-S, v=1-0', '045510 HC-13-P', '045511 PN, v=0-5',
                '045512 HC(O)NH2, v=0', '045513 c-CC-13-H4O', '045514 HSiO', '045515 ethylamine, anti-conformer',
                '045516 HC(O)NH2, v12=1', '045517 t-HOCO', '045518 c-HOCO', '045519 SiO-17, v=0-4', '045520 HON2+',
                '045521 HN2O+', '045522 HOCO+', '045523 Al-26-F, v=0-2', '045524 CH3CDO, vt=0,1',
                '045525 CH2DCHO, vt=0', '045526 AlO-18, v=0', '045527 AlO-18, v=1,2', '045528 CH3C-13-HO,vt=0,1',
                '045529 C-13-H3CHO,vt=0,1', '045530 H2CNOH', '045531 c-C2H3DO', '045532 CH3NHCH3',
                '046501 CS-34, v=0,1', '046502 Si-30-O, v=0-6', '046503 SiO-18, v=0-5', '046504 HC-13-S+',
                '046505 DCS+', '046506 t-HCOOH', '046507 c-HCOOH', '046508 C-13-S-33', '046509 H2CS',
                '046510 CS-34, v=1-0', '046511 PN-15, v=0-1', '046512 HC-13-(O)NH2', '046513 HC(O)N-15-H2',
                '046514 CH3OCH3, v=0', '046515 NS, v=0', '046516 NS, v=1', '046517 NS, v=1-0', '046518 c-C2H4O-18',
                '046519 H2SiO', '046520 DC(O)NH2', '046521 cis-HC(O)NHD', '046522 trans-HC(O)NHD',
                '046523 Si-29-O-17, v=0-2', '046524 C2H5OH,v=0', '046525 DOCO+', '046526 HOC-13-O+', '046527 NS+',
                '046528 AlF, v=0-5', '046529 DSC', '046530 HSC-13', '046531 DCS', '046532 HC-13-S', '046533 c-CD2CH2O',
                '046534 CHD2CHO, vt=0', '046535 H2CS, v4=1', '046536 H2CS, v6=1', '046537 H2CS, v3=1',
                '046538 H2CS, v2=1', '046539 H2CS, nu4', '046540 H2CS, nu6', '046541 H2CS, nu3', '046542 H2CS, nu2',
                '047501 C-13-S-34', '047502 HCS-34+', '047503 t-HC-13-OOH', '047504 HDCS', '047505 H2C-13-S',
                '047506 H2CS-33', '047507 PO, v=0', '047508 HC(O-18)NH2', '047509 NS-33', '047510 N-15-S',
                '047511 a-CH3C-13-H2OH', '047512 a-C-13-H3CH2OH', '047513 Si-29-O-18, v=0-3',
                '047514 Si-30-O-17, v=0-2', '047515 a-CH3CH2OD', '047516 a-CH3CHDOH', '047517 a-a-CH3DCH2OH',
                '047518 a-s-CH2DCH2OH', '047519 PO, v=1,2', '047520 PO, v=3-5', '047521 CH3ONH2', '047522 C-13-H3OCH3',
                '047523 CCl+, v=0', '047524 SiF+, v=0,1', '047525 SiF+, v=1-0,2-1', '047526 SiF+, v=2-0',
                '047527 trans-HONO', '047528 cis-HONO', '048501 SO, v=0', '048502 SO, v=1', '048503 CS-36',
                '048504 SO-sgl-D, v0,1', '048505 NaCCH', '048506 HPO', '048507 D2CS', '048508 H2CS-34', '048509 NS-34',
                '048510 CH3SH,v=0-2', '048511 Si-30-O-18, v=0-3', '048512 MgC2', '048513 C-13-Cl+, v=0',
                '048514 Si-29-F+, v=0', '049501 S-33-O        ', '049502 SO-17     ', '049503 C4H, v=0',
                '049504 C4H, v7=1', '049505 C4H, v7=2^0', '049506 C4H, v7=2^2', '049507 MgCCH', '049508 C-13-S-36',
                '049509 C4H-', '049510 NaCN', '049511 N-15-S-34', '049512 HSO', '049513 H2C-13-S-34',
                '049514 C4H, v6=1', '049515 C4H, v5=1', '049516 C4H, v6=v7=1', '049517 CH3SD,v=0-2',
                '049518 C-13-H3SH,v=0-2', '049519 Mg-25-C2', '049520 MgCC-13', '049521 CCl-37+, v=0',
                '049522 Si-30-F+, v=0', '050501 S-34-O     ', '050502 SO-18', '050503 l-C4H2', '050504 MgNC, v=0',
                '050505 MgNC, v2=1', '050506 C4D', '050507 C-13-CCCH', '050508 CC-13-CCH', '050509 CCC-13-CH',
                '050510 CCCC-13-H', '050511 C3N, v=0', '050512 C3N, v5=1', '050513 NaC-13-N', '050514 C3N-',
                '050515 C4D-', '050516 NS-36', '050517 H2CS-36', '050518 CH3S-34-H,v=0-2', '050519 Mg-26-C2',
                '051501 HC3N, v=0', '051502 HC3N, v7=1', '051503 HC3N, v7=2', '051504 HC3N, v6=1',
                '051505 HC3N, v6=v7=1', '051506 HC3N, v4=1', '051507 HC3N, v4=v7=1', '051508 HC3N, v5=1/v7=3',
                '051509 HC3N, v4=1,v7=2/v5=2^0', '051510 KC', '051511 C-13-CCN', '051512 CC-13-CN', '051513 CCC-13-N',
                '051514 C3N-15', '051515 l-C-13-C3H2', '051516 l-CC-13-C2H2', '051517 l-C2C-13-CH2',
                '051518 l-C3C-13-H2', '051519 HC3N, v3=1', '051520 HC3N, v2=1', '051521 l-C4HD',
                '051522 HC3N, nu3 band', '051523 HC3N, nu2 band', '051524 HC3N, nu1 band', '051525 HC3N, v7=4/v5=v7=1',
                '051526 HC3N, v6=2', '051527 HMgNC', '051528 HNC3', '051529 AlC2', '051530 CH2DCl-35',
                '052501 CCCO, v=0', '052502 S-36-O', '052503 HC3NH+', '052504 KCH', '052505 CaC',
                '052506 C4H4: butenyne', '052507 (c-C3H2)CH2', '052508 DC3N, v=0', '052509 HC-13-CCN, v=0',
                '052510 HCC-13-CN, v=0', '052511 HCCC-13-N, v=0', '052512 HCCCN-15, v=0', '052513 DC3N, v7=1',
                '052514 HC-13-CCN, v7=1', '052515 HCC-13-CN, v7=1', '052516 HCCC-13-N, v7=1', '052517 HCCCN-15, v7=1',
                '052518 HC-13-CCN, v7=2', '052519 HCC-13-CN, v7=2', '052520 HCCC-13-N, v7=2', '052521 HC-13-CCN, v6=1',
                '052522 HCC-13-CN, v6=1', '052523 HCCC-13-N, v6=1', '052524 HC-13-CCN, v5=1/v7=3',
                '052525 HCC-13-CN, v5=1/v7=3', '052526 HCCC-13-N, v5=1/v7=3', '052527 SiC2, v=0', '052528 SiC2, v3=1',
                '052529 SiC2, v3=2', '052530 C3O, v5=1', '052531 HC-13-CCN, v6=v7=1', '052532 HCC-13-CN, v6=v7=1',
                '052533 HCCC-13-N, v6=v7=1', '052534 HCCC-13-N, v4=1', '052535 HCCC-13-N, v4=v7=1', '052536 AlCCH',
                '052537 SiC2, v3=3', '052538 SiC2, v3=4', '052539 HCCNCH+', '053501 AlNC', '053502 SiCCH',
                '053503 HCC-13-C-13-N, v=0', '053504 HC-13-CC-13-N, v=0', '053505 AlCN', '053506 HCCC-13-N-15',
                '053507 t-HC3O', '053508 HC-13-C-13-CN, v=0', '053509 SiC-13-C', '053510 Si-29-C2', '053511 DC-13-CCN',
                '053512 DCC-13-CN', '053513 DCCC-13-N', '053514 DC3N-15', '053515 C2H3CN, v=0', '053516 C-13-CCO',
                '053517 CC-13-CO', '053518 CCC-13-O', '053519 NCCNH+', '053520 CrH', '053521 CH2DCl-37', '053522 HC3O+',
                '053523 Z-propynimine', '053524 E-propynimine', '054501 SiCN', '054502 SiNC', '054503 Propadienone',
                '054504 Cyclopropenone', '054505 Si-30-C2', '054506 H2C-13-CHCN, v=0', '054507 H2CC-13-HCN, v=0',
                '054508 H2CCHC-13-N, v=0', '054509 H2CCHCN-15', '054510 Propynal', '054511 C3O-18', '054512 E-HNCHCN',
                '054513 Z-HNCHCN', '054514 H2CNCN', '054515 NCCND+', '054516 NCC-13-NH+', '054517 NC-13-CNH+',
                '054518 CH2CHCNH+', '054519 C2H5CCH', '054520 c-H2C2Si', '054521 H2C2Si', '054522 HSiCCH',
                '055501 HCOCN', '055502 C2H5CN, v=0', '055503 CCP', '055504 C3F', '055505 HSiCN', '055506 HSiNC',
                '055507 C2H5NC', '055508 NaS', '055509 HCCCH2NH2', '055511 Cyclopropenone-13C1',
                '055512 Cyclopropenone-13C2', '055513 C2H5CN, v20=1-A', '055514 C2H5CN, v12=1-A', '055515 DCCCHO',
                '055516 HCCCDO', '055517 HCCC-13-HO', '055518 HCC-13-CHO', '055519 HC-13-CCHO', '055520 Allylimine-Ta',
                '056501 HCCP', '056502 CCS', '056503 ONCN', '056504 C2H5C-13-N, v=0', '056505 CH3C-13-H2CN, v=0',
                '056506 C-13-H3CH2CN, v=0', '056507 H2NCH2CN, v=0', '056508 C2H5CN-15', '056509 CH3CHDCN',
                '056510 CH2D(ip)CH2CN', '056511 CH2D(oop)CH2CN', '056512 C-13-CP', '056513 CC-13-P', '056514 KOH',
                '056515 CaO, v=0,1', '056516 Fe-atom', '056517 Fe+', '056518 SiH3CCH', '056519 trans-propenal, v=0',
                '056520 H2NCH2CN, v11+v18=1', '056521 H2NCH2CN, v17=1', '056522 NaSH', '056523 c-H2C3O-18',
                '056524 c-D2C3O', '056525 MgS, v=0,1', '056526 Isobutene', '057501 CaOH', '057502 HDNCH2CN',
                '057503 SiH3CN', '057504 PCN', '057505 CH3NCO, vb=0', '057506 CH3NCO, vb=1',
                '057507 CH3C-13-H2C-13-N, v=0', '057508 C-13-H3CH2C-13-N, v=0', '057509 C-13-H3C-13-H2CN, v=0',
                '057510 CH3CNO', '057511 CH3OCN', '057512 HOCH2CN', '057513 H2NCH2C-13-N', '057514 H2NC-13-H2CN',
                '057515 HCCS', '057516 MgSH', '057517 HCCS+', '057518 Mg-25-S, v=0', '057519 E-1-propanimine',
                '057520 Z-1-propanimine', '057521 CH3C(NH)CH3', '058501 H2C2S', '058502 NaCl, v=0-15',
                '058503 KF, v=0,1', '058504 NCS', '058505 s-Propanal, v=0', '058506 K-41-OH', '058507 CH3CP',
                '058508 D2NCH2CN', '058509 Si2H2, dibridged', '058510 Si2H2, monobridged', '058511 SiH3C-13-N',
                '058512 Si-29-H3CN', '058513 c-C2H2O2', '058514 Propylene oxide', '058515 Oxetane',
                '058516 C-13-H3C-13-H2C-13-N', '058517 s-Propanal, v24=1', '058518 s-Propanal, v23=1',
                '058519 g-Propanal, v=0', '058520 HCCSH', '058521 Mg-26-S, v=0', '058522 MgS-34, v=0', '059501 MgCl',
                '059502 CaF', '059503 HNCS, a-type', '059504 HNCS, b-type', '059505 HSCN', '059506 C2Cl', '059507 AlS',
                '059508 Si-30-H3CN', '059509 H2PCN', '059510 HCNS', '059511 HSNC', '059512 NaCl-36, v=0-5',
                '059513 H2CC-13-S', '059514 H2-13-CCS', '059515 HDC2S', '059516 C-13-H3C(O)CH3', '059517 HNSiO',
                '060501 CH2(OH)CHO, v=0', '060502 NaCl-37, v=0-15', '060503 OCS, v=0', '060504 OCS, v2=1',
                '060505 Ga-n-C3H7OH', '060506 SiS, v=0-20', '060507 SiS, v=1-0,2-1', '060508 SiS, v=2-0',
                '060509 Ethylene sulfide', '060510 DNCS, a-type', '060511 DNCS, b-type', '060512 HNC-13-S, a-type',
                '060513 HN-15-CS, a-type', '060514 HSC-13-N', '060515 DSCN', '060516 HSCN-15', '060517 urea, v=0',
                '060518 g-i-C3H7OH', '060519 a-i-C3H7OH', '060520 AlSH', '060521 s-C2H3SH', '060522 a-C2H3SH',
                '060523 CH3COOH, vt=0', '060524 CH3COOH, vt=1', '060525 CH3COOH, vt=2', '060526 CH3COOH, Dvt&lt;&gt;0',
                '060527 Ti-44-O, v=0', '060528 Ti-44-O, v=1', '060529 CH2(OH)CHO, v18=1', '060530 CH2(OH)CHO, v12=1',
                '060531 CH2(OH)CHO, v17=1', '060532 H2CCS-34', '060533 urea, v1', '060534 urea, v2 &amp; v3',
                '060535 urea, v4 &amp; v5', '060536 HPCO', '061501 PNO', '061502 OC-13-S', '061503 OCS-33',
                '061504 O-17-CS', '061505 C5H', '061506 Si-29-S, v=0-12', '061507 Si-29-S, v=1-0',
                '061508 SiS-33, v=0-9', '061509 HSCO+', '061510 HOCS+', '061511 C2Cl-37', '061512 HSiS',
                '061513 CH2(OH)C-13-HO', '061514 C-13-H2(OH)CHO', '061515 CH3OC-13-HO, vt=0,1', '061516 CH2(OD)CHO',
                '061517 CHD(OH)CHO', '061518 CH2(OH)CDO', '061519 HNCS-34, a-type', '061520 HS-34-CN', '061521 ScO',
                '061522 Al-26-Cl, v=0-2', '061523 HC(S)NH2', '061524 c-C5H', '061525 HCOOCH2D', '061526 DCOOCH3',
                '062501 l-C5H2', '062502 TiN', "062503 aGg' glycol", "062504 gGg' glycol", '062505 OCS-34',
                '062506 O-18-CS', '062507 OC-13-S-33', '062508 SiS-34, v=0-12', '062509 SiS-34, v=1-0',
                '062510 Si-30-S, v=0-12', '062511 Si-30-S, v=1-0', '062512 Si-29-S-33', '062513 H2SiS', '062514 C4N',
                '062515 t-HC(O)SH', '062516 c-HC(O)SH', '062517 C4C-13-H', '062518 C3C-13-CH', '062519 C2C-13-C2H',
                '062520 CC-13-C3H', '062521 C-13-C4H', '062522 C5D', '062523 g-C2H5SH', '062524 a-C2H5SH',
                '062525 DSCO+', '062526 AlCl, v=0-10', '062527 CH3OCH2OH', '062528 Ti-46-O, v=0', '062529 Ti-46-O, v=1',
                '062530 Ti-46-O, v=2', '062531 Ti-46-O, v=3', '062532 c-C3HCCH', '062533 HCOOCHD2', '063501 l-HC4N',
                '063502 OC-13-S-34', '063503 O-18-C-13-S', '063504 Si-29-S-34, v=0,1', '063505 Si-30-S-33',
                '063506 c-C3HCN', '063507 Al-26-Cl-37, v=0-2', '063508 AlCl-36, v=0-2', '063509 Ti-47-O, v=0',
                '063510 Ti-47-O, v=1', '063511 Ti-47-O, v=2', '063512 Ti-47-O, v=3', '063513 SiCl+, v=0,1',
                '063514 HNSO', '064501 p-c-SiC3', '064502 SO2, v=0', '064503 SO2, v2=1', '064504 TiO, v=0',
                '064505 l-SiC3', '064506 CuH', '064507 CH3C4H', '064508 ScF', '064509 H2C4N', '064510 OCS-36',
                '064511 O-18-CS-34', '064512 SO2, nu2', '064513 Si-30-S-34, v=0,1', '064514 SiS-36', '064515 KCCH',
                '064516 o-c-SiC3', '064517 g-C2H5S-34-H', '064518 CrC', '064519 AlCl-37, v=0-10',
                '064520 1,4-Pentadiyne', '064521 TiO, v=1', '064522 TiO, v=2', '064523 TiO, v=3', '064524 TiO, v=4',
                '064525 TiO, v=5', '064526 H2C3HCCH', '064527 HC3HCN', '065501 S-33-O2', '065502 SOO-17',
                '065503 CH3C3N', '065504 ZnH', '065505 CH3CCNC', '065506 H2CCCHCN', '065507 Si-29-S-36', '065508 KCN',
                '065509 HS2', '065510 cis-HOSO+', '065511 CaCCH', '065512 SiC3H', '065513 HC4O', '065514 HCCCH2CN',
                '065515 Ti-49-O, v=0', '065516 Ti-49-O, v=1', '065517 Ti-49-O, v=2', '065518 Ti-49-O, v=3',
                '066501 S-34-O2', '066502 SOO-18', '066503 CaNC', '066504 Cu-65-H', '066505 Si-30-S-36',
                '066506 H2C(CN)2', '066507 H2S2', '066508 SiC2N', '066509 H2C4O', '066510 DS2', '066511 CrN',
                '066512 Ti-50-O, v=0', '066513 Ti-50-O, v=1', '066514 Ti-50-O, v=2', '066515 Ti-50-O, v=3',
                '066516 TiO-18, v=0', '066517 TiO-18, v=1', '066518 TiO-18, v=2', '066519 TiO-18, v=3', '066520 c-C5H6',
                '066521 CH2DC3N', '066522 H2NC3N', '066523 H2C3Si', '067501 Zn-66-H', '067502 NCHCCO',
                '067503 c-C3H5CN', '067504 Pyrrole ', '067505 HCCNSi', '067506 TiF', '067507 c-C5H6-13C1',
                '067508 c-C5H6-13C2', '067509 c-C5H6-13C5', '067510 c-C5H6-D1', '067511 c-C5H6-D2', '067512 c-C5H6-D5',
                '067513 E-CH3CHCHCN', '067514 s-CHCHCH2CN', '067515 g-CHCHCH2CN', '068501 FeC', '068502 HC3P',
                '068503 C3S, v=0', '068504 C3O2, nu7', '068505 C3S, v5=1', '068506 H2CNCH2CN', '068507 Furan ',
                '068508 Si2C, v=0', '068509 NCNSi', '068510 CrO', '068511 NCCNO', '068512 HNC(CH3)CN', '069501 NC2P',
                '069502 CC-13-CS', '069503 C-13-CCS', '069504 Zn-68-H', '069505 n-C3H7CN, v=0', '069506 i-C3H7CN',
                '069507 CCC-13-S', '069508 g-n-C3H7CN, v30=1', '069509 a-n-C3H7CN, v30=1', '069510 g-n-C3H7CN, v29=1',
                '069511 a-n-C3H7CN, v18=1', '069512 g-n-C3H7CN, v30=2', '069513 a-n-C3H7CN, v30=2',
                '069514 g-n-C3H7CN, v28=1', '069515 a-n-C3H7CN, v29=1', '069516 i-C3H7CN, v30=1', '069517 HC3S',
                '069518 Cyanooxirane', '069519 HC3S+', '069520 t-C2H3NCO', '069521 cis-C2H3NCO', '070501 NiC',
                '070502 C3S-34', '070503 H2C3S', '070504 NCCONH2', '070505 CH3CH(NH2)CN', '070506 CC-13-C-13-S',
                '070507 C-13-CC-13-S', '070508 i-C3H7CN-13C1', '070509 i-C3H7CN-13C2', '070510 i-C3H7CN-13C3',
                '070511 Propynethial', '070512 FeN', '070513 NH2CH2CH2CN, conf. I', '070514 NH2CH2CH2CN, conf. II',
                '070601 Fe-54-O, Om=4', '070602 Fe-54-O, Om=3', '070603 Fe-54-O, Om=2', '071501 CoC', '071502 C3Cl',
                '071503 CCC-13-S-34', '071504 CC-13-CS-34', '071505 C-13-CCS-34', '071506 KS', '071507 CrF',
                '071508 C2H5NCO', '071509 HC(S)CN', '071510 syn-C2H3C(O)NH2, v=0', '071511 syn-C2H3C(O)NH2, v24=1',
                '071512 H2C2C-13-S', '071513 H2CC-13-CS', '071514 H2C-13-C2S', '071515 HDC3S', '072501 CaS, v=0,1',
                '072502 s-cis-Propenoic acid', '072503 s-trans-Propenoic acid', '072504 HOCHCHCHO',
                '072505 vinyl formate', '072506 KSH', '072507 H2C3S-34', '072508 2-Hydroxypropenal', '072601 FeO, Om=4',
                '072602 FeO, Om=3', '072603 FeO, Om=2', '073501 C6H, v=0', '073502 C6H-', '073503 HSCH2CN',
                '073504 CaSH', '073505 MgC4H', '073506 HC(S-34)CN', '073507 MgC4H+', '073508 NaC3N',
                '073509 C6H, v11=1mS', '073510 C6H, v11=1D', '074501 C5N', '074502 l-C6H2', '074503 aa-diethyl ether',
                '074504 NiO', '074505 KCl, v=0-15', '074506 C-13-C5H', '074507 CC-13-C4H', '074508 C2C-13-C3H',
                '074509 C3C-13-C2H', '074510 C4C-13-CH', '074511 C5C-13-H', '074512 C6D', '074513 C5N-',
                '074514 ethyl formate', '074515 ag-diethyl ether', '074516 glyoxylic acid', '074517 CH3OCH2CHO',
                '074518 MgC3N', '074519 CH3CHOHCHO', '074520 H2NCH2C(O)NH2', '074521 HC2HC4', '074522 MgC3N+',
                '075501 HC4NC', '075502 CaCl', '075503 HC5N, v=0', '075504 HC5N, v11=1', '075505 HC5N, v11=2',
                '075506 HC5N, v10=1', '075507 HC5N, v11=3', '075508 HC5N, v11=4', '075509 HC5N, v9=1',
                '075510 HC5N, nu7', '075511 Glycine, conf. I', '075512 Glycine, conf. II', '075513 HC5N, v10=v11=1',
                '075514 HC5N, v11=5', '075515 K-40-Cl, v=0-5', '075516 KCl-36, v=0-5', '075517 HOCH2C(O)NH2',
                "075518 g'Gg'-Alaninol", "075519 gG'g-Alaninol", '075520 HMgC3N', '076501 NC3NC', '076502 SiC4',
                '076503 Benzyne', '076504 KCl-37, v=0-15', '076505 K-41-Cl, v=0-15', '076506 DC5N', '076507 HC-13-C4N',
                '076508 HCC-13-C3N', '076509 HC2C-13-C2N', '076510 HC3C-13-CN', '076511 HC4C-13-N', '076512 HC5N-15',
                "076513 aG'g-1,2-Propanediol", '076514 glycolic acid', "076515 gG'a-1,2-Propanediol",
                "076516 g'G'g-1,2-Propanediol", '076517 OSiS', "076518 a'GG'g-1,3-Propanediol",
                "076519 gGG'g-1,3-Propanediol", "076520 g'Ga-1,2-Propanediol", "076521 gG'g'-1,2-Propanediol",
                "076522 aGg'-1,2-Propanediol", "076523 g'Gg-1,2-Propanediol", '076524 HC5NH+', '077501 Si-29-C4',
                '077502 SiC-13-C3', '077503 SiCC-13-C2', '077504 SiC2C-13-C', '077505 SiC3C-13', '077506 C2H3C3N',
                '077507 E-HC2CHCHCN', '077508 Z-HC2CHCHCN', '077509 ScS', '077510 AlC3N', '077511 SiC4H', '077512 HC5O',
                '077513 K-40-Cl-37, v=0-5', '077514 K-41-Cl-36, v=0-5', '077515 NC4NH+', '078501 Si-30-C4',
                '078502 Ti-46-O2', '078503 Ti-46-S', '078504 SiC3N', '078505 Pentadiynal', '078506 t-HC(S)SH',
                '078507 c-HC(S)SH', '078508 K-41-Cl-37, v=0-15', '078509 Fulvene', '078510 Benzvalene',
                '078511 Dewar benzene', '078512 Dimethylenecyclobutene', '078513 H2C4Si', '079501 Pyridine ',
                '079502 Z-CH2(CH)3CN', '079503 E-CH2(CH)3CN', '079504 a-CH2CCHCH2CN', '079505 s-CH2CCHCH2CN',
                '080501 C4S', '080502 ScCl', '080503 S2O, v=0', '080504 TiO2', '080505 TiS', '080506 OC(CN)2',
                '080507 c-C5H4O', '080508 1,3-c-C6H8', '081501 HC4S', '082501 ScCl-37', '082502 Ti-50-O2',
                '082503 Ti-50-S', '082504 H2C4S', '083501 AA-n-C4H9CN', '083502 GA-n-C4H9CN', '083503 AG-n-C4H9CN',
                '083504 t-C4H9CN', '083505 2-CAB, v=0', '083506 3-MABN, v=0', '083507 3-MGBN, v=0', '084501 FeCO',
                '084502 OC3S', '084503 Diketene', '085501 C7H', '086501 l-C7H2', '086502 NiCO', '087501 l-HC6N',
                '087502 CrCl', '088501 SiC5', '088502 CH3C6H', '088503 Allenyldiacetylene', '088504 Heptatetraen-6-yne',
                '089501 CH3C5N', '089502 a-Alanine, conf. I', '089503 a-Alanine, conf. II', '089504 HC6O',
                '089505 CrCl-37', '089506 Allenylcyanoacetylene', '090501 Glyceraldehyde',
                '090502 Ethenylidenecyclopentadiene', '092501 C5S', "092502 G'Gg'gg'-Glycerol",
                "092503 GGag'g'-Glycerol", '092504 Norbornadiene', '092505 Methylenecyclohexadiene',
                '092506 Spiroheptadiene', '093501 C-13-C4S', '093502 CC-13-C3S', '093503 C2C-13-C2S',
                '093504 C3C-13-CS', '093505 C4C-13-S', '093506 HC5S', '094501 Phenol', '094502 C5S-34',
                '094503 Pentadiynethial', '094504 H2C5S', '096501 cis-S2O2', '097501 C8H', '097502 C8H-',
                '097503 MgC6H', '097504 MgC6H+', '098501 l-C8H2', '098502 MgC5N', '098503 C7N-', '098504 MgC5N+',
                '099501 HC7N, v=0', '099502 HC7N, v15=1', '099503 HC7N, v15=2', '099504 HC6NC', '100501 SiC6',
                '100502 DC7N', '100503 HC-13-C6N', '100504 HCC-13-C5N', '100505 HC2C-13-C4N', '100506 HC3C-13-C3N',
                '100507 HC4C-13-C2N', '100508 HC5C-13-CN', '100509 HC6C-13-N', '100510 HC7N-15', '100511 HC7NH+',
                '101501 HC7O', '102501 Heptatriynal', '102502 c-C6H5CCH', '103501 c-C6H5CN', '104501 C6S', '105502 YO',
                '106501 Benzaldehyde', '108501 Anisole', '109501 C9H', '112501 CH3C8H', '113501 CH3C7N', '116501 C7S',
                '116502 c-C9H8', '117501 HC7S', '121501 C10H', '121502 YS', '121503 C10H-', '123501 HC9N, v=0',
                '124501 DC9N', '124502 HC-13-C8N', '124503 HCC-13-C7N', '124504 HC2C-13-C6N', '124505 HC3C-13-C5N',
                '124506 HC4C-13-C4N', '124507 HC5C-13-C3N', '124508 HC6C-13-C2N', '124509 HC7C-13-CN',
                '124510 HC8C-13-N', '124511 HC9N-15', '127501 C6H5C3N', '127502 1-CN-4-CCH-C6H4',
                '127503 1-CN-2-CCH-C6H4', '127504 1-CN-3-CCH-C6H4', '128501 c-C10H8', '137501 CH3C9N', '140501 C9S',
                '145501 C12H', '147501 HC11N', '152501 c-C12H8', '154501 c-C12H10', '166501 c-C13H10', '171501 HC13N']
cdmsMols = [i.split(' ')[1] for i in cdmsMolsFull]
cdmsTagsFull = [i.split(' ')[0] for i in cdmsMolsFull]

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
rms = molsim.stats.get_rms(int_arr)

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

# peak_indices3 = [peak_indices[i] for i in range(len(peak_indices)) if near_whole(peak_freqs[i]) == False]
# peak_ints3 = [peak_ints[i] for i in range(len(peak_ints)) if near_whole(peak_freqs[i]) == False]
# peak_freqs3 = [peak_freqs[i] for i in range(len(peak_freqs)) if near_whole(peak_freqs[i]) == False]

peak_indices = peak_indices3
peak_ints = peak_ints3
peak_freqs = peak_freqs3

print('')
print('Number of peaks at ' + str(sig) + ' sigma significance in the spectrum: ' + str(len(peak_freqs)))
print('')

#print('')
# sorting peaks by intensity
combPeaks = [(peak_freqs[i], peak_ints[i]) for i in range(len(peak_freqs))]
sortedCombPeaks = sortTupleArray(combPeaks)
sortedCombPeaks.reverse()
spectrum_freqs = [i[0] for i in sortedCombPeaks]
spectrum_ints = [i[1] for i in sortedCombPeaks]
firstLine = ['obs frequency', 'obs intensity']

# Making the dataset file.
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


'''
Finding all candidate lines from .cat files in local directory.
'''
localYN = True
if localYN == True:
    for filename in os.listdir(localDirec):
        q = os.path.join(localDirec, filename)
        # checking if it is a file
        if os.path.isfile(q) and q.endswith(
                '.cat') and 'super' not in q and '.DS_Store' not in q and 'hc6n_rc' not in q and 'OC_CN_2' not in q and 'allene_u01' not in q and 'c4h4o_isomer4' not in q:
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
                src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=dv_value)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014, observation=observation1)
            else:
                src = molsim.classes.Source(Tex=temp, column=1.E9)
                sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                                res=0.0014)
            peak_freqs2 = sim.spectrum.frequency
            peak_ints2 = sim.spectrum.Tb
            if peak_ints2 is not None:
                freqs = list(peak_freqs2)
                ints = list(peak_ints2)
                dfFreq['frequencies'] = freqs
                dfFreq['intensities'] = ints
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

print('done with local catalog scraping!')
print('')
print('querying CDMS/JPL')
print('')


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
The following loop combines queries of Splatalogue, CDMS, and JPL to get all candidate molecules
for all of the lines in the spectrum along with the required information. For all candidates,
the spectrum is simulated at the inputted experimental temperature and saved in the 
splatalogue_catalogs directory.
'''

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
                    #jplMolsAlready.append(forms[z])
                    #rowComb.append((cdmsNames[i], cdmsForms[i], freqs[q]))


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
            src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=0)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014, observation=observation1)
        else:
            src = molsim.classes.Source(Tex=temp, column=1.E9)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014)

        peak_freqs2 = sim.spectrum.frequency
        peak_ints2 = sim.spectrum.Tb
        if peak_ints2 is not None:
            freqs = list(peak_freqs2)
            ints = list(peak_ints2)
            dfFreq['frequencies'] = freqs
            dfFreq['intensities'] = ints
            saveName = os.path.join(pathSplat, str(catCount) + '.csv')
            dfFreq.to_csv(saveName)

            savedCatIndices.append(catCount)
            # catalogNames.append(jplNames[i])
            savedForms.append(cdmsForms[i])
            savedTags.append(cdmsTags[i])
            savedList.append('CDMS')

        catCount += 1


jplDirec = direc  + 'jpl_cats/'
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
                    # jplMolsAlready.append(forms[z])
                    #rowComb.append((cdmsNames[i], cdmsForms[i], freqs[q]))

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
            src = molsim.classes.Source(Tex=temp, column=1.E9, size=sourceSize, dV=0)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014, observation=observation1)
        else:
            src = molsim.classes.Source(Tex=temp, column=1.E9)
            sim = molsim.classes.Simulation(mol=mol, ll=minFreq, ul=maxFreq, source=src, line_profile='Gaussian',
                                            res=0.0014)

        peak_freqs2 = sim.spectrum.frequency
        peak_ints2 = sim.spectrum.Tb
        if peak_ints2 is not None:
            freqs = list(peak_freqs2)
            ints = list(peak_ints2)
            dfFreq['frequencies'] = freqs
            dfFreq['intensities'] = ints
            saveName = os.path.join(pathSplat, str(catCount) + '.csv')
            dfFreq.to_csv(saveName)

            savedCatIndices.append(catCount)
            # catalogNames.append(jplNames[i])
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


#print('saving things')
'''
keysList = list(countDict.keys())
valueList = list(countDict.values())

countDF = pd.DataFrame()
countDF['smiles'] = keysList
countDF['count'] = valueList

countDF.to_csv(direc + 'counts_update.csv')
del countDF

edgeDF = pd.DataFrame()
edgeDF['edges'] = listToString(edges)
edgeDF.to_csv(direc + 'edges_update.csv')
del edgeDF

smilesDF = pd.DataFrame()
smilesDF['smiles'] = smiles
smilesDF.to_csv(direc + 'smiles_update.csv')
del smilesDF

vectorDF = pd.DataFrame()
vectorDF['smiles'] = vectorSmiles
listVectors = list(allVectors)
#print(len(listVectors))
#print(len(listVectors[0]))
#print(type(listVectors))
#print(type(listVectors[0]))
vectorDF['vectors'] = listToString(listVectors)
vectorDF.to_csv(direc + 'vectors_update.csv')

'''



print('')
print('Ok, thanks! Now running the assignment algorithm.')
print('')
scrapeMins = (tockScrape-tickScrape)/60
scrapeMins2 = "{{:.{}f}}".format(2).format(scrapeMins)
print('Catalog scraping took ' + str(scrapeMins2) + ' minutes.')

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
#print(maxObservedInt)


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
'''
parent_csv = pd.read_csv(os.path.join(direc, 'parent_list.csv'))
mol_par = list(parent_csv['mol'])
list_par = list(parent_csv['linelist'])
mol_parent = list(parent_csv['parent'])
list_parent = list(parent_csv['parent list'])
tag_parent = list(parent_csv['parent tag'])
'''
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

        inputValues = [smiles, detectedSmiles, testSmiles, testIso, testFrequencies, correctFreq, edges, countDict,
                       oldHighestIntensities, intensityValue, forms, linelists, tags, previousBest, qns,
                       oldHighestSmiles, newDetSmiles, sorted_dict_last]

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



        testingScoresFinal, testingScoresSmiles, softScores, ranking, testingScores, sorted_dict, globalScores, sortedTuplesCombined = runPageRankInit2(
            smiles, detectedSmiles, testSmiles, testIso, testFrequencies, correctFreq, edges,
            countDict, oldHighestIntensities, intensityValue, forms, linelists, tags, previousBest,
            qns, oldHighestSmiles, newCalc, sorted_dict_last, loopIter)
        
        sorted_dict_last = sorted_dict

        sorted_smiles = [q[0] for q in sorted_dict]
        sorted_values = [q[1] for q in sorted_dict]

    indicesBefore = list(range(i))

    newTestingScoresListFinal = []
    newDetectedSmiles = {}
    newPreviousBest = []
    newCombinedScoresList = []
    newBestGlobalScoresFull = []

    topReverseSmiles = []
    newHighestIntensities = {}
    newHighestSmiles = {}

    # getting information on top ranked molecule for the line in question
    topSmile = testingScoresFinal[0][0]
    topGlobalScore = testingScoresFinal[0][1]
    #print('top score tau')
    #print(testingScoresFinal[0][3])
    #if testingScoresFinal[0][3] == True and topGlobalScore > globalThresh:
    #    opticallyThick.append(topSmile)
    #    print('optically thick! ' + topSmile)
    #    print(opticallyThick)

    for testingSmilesIdx in range(len(sortedTuplesCombined)):
        if sortedTuplesCombined[testingSmilesIdx][0][0] == topSmile:
            topMol = sortedTuplesCombined[testingSmilesIdx][0][1]
            topScore = sortedTuplesCombined[testingSmilesIdx][1]
            break

    override = {}



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


                    foundSig = False
                    for sig in sigmaListReverse:
                        if sig[0] == form and sig[1] == freq:
                            rule_out_reverse = sig[2]
                            foundSig = True

                    if foundSig == False:
                        rule_out_reverse = True
                        print('not present in rule out reverse')
                        


                    #rule_out_reverse = False


                    #rule_out_reverse = checkAllLines(linelist, form, tag, freq, peak_freqs_full, peak_ints_full, rms)
                    tu2, subReport, scaledPer, offset, value, per = scaleScoreReverse(smile, validAtoms, subReport,
                                                                                      sorted_smiles, sorted_values,
                                                                                      freq, correctFreq, molRank,
                                                                                      maxInt, form, newPreviousBest,
                                                                                      newHighestIntensities,
                                                                                      intensityReverse, iso, qn,
                                                                                      newHighestSmiles,
                                                                                      rule_out_reverse, line_int_value, linelist, tag)
                    newIndexTest.append(tu2)
                    report.append(subReport)

                # compiling top scores
                sortedNewTest, best, bestSmile, bestMol, bestScore, bestGlobalScore, testingScoresDictReverse, percentiles, soft, tuplesCombinedReverse, sortedTuplesCombinedReverse, topGlobalScoreSecond = getTopScores(
                    newIndexTest)

                newTestingScoresListFinal.append(sortedNewTest)
                newCombinedScoresList.append(sortedTuplesCombinedReverse)
                newBestGlobalScoresFull.append(bestGlobalScore)
                topReverseSmiles.append(bestSmile)

                # updating the override counter
                override = updateOverride(bestGlobalScore, globalThresh, bestMol, override)

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

        # next
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
        oldTestingScoresList = newBestGlobalScoresFull

        oldCombinedTestingScoresList = newCombinedScoresList
        oldHighestIntensities = newHighestIntensities
        oldHighestSmiles = newHighestSmiles
    else:
        for teVal in testingScoresFinal:
            if teVal[1] > globalThresh:
                if teVal[2] not in oldHighestIntensities:
                    oldHighestIntensities[teVal[2]] = intensityValue
                if teVal[0] not in oldHighestSmiles:
                    oldHighestSmiles[teVal[0]] = intensityValue
                if teVal[2] not in previousBest:
                    previousBest.append(teVal[2])

        oldTestingScoresListFull.append(testingScoresFinal)
        oldTestingScoresList.append(topGlobalScore)
        oldCombinedTestingScoresList.append(sortedTuplesCombined)
        newDetSmiles = False
        if topScore > thresh and topGlobalScore > globalThresh:
            if topSmile not in detectedSmiles:
                detectedSmiles[topSmile] = 1
                newDetSmiles = True

        overNew = {}
        for o in range(len(oldTestingScoresListFull)):
            if oldTestingScoresListFull[o][0][1] > 50 and oldTestingScoresListFull[o][0][1] < globalThresh:
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
sorted_dict, sorted_smiles, sorted_values = runPageRankInit2_Final(smiles, detectedSmiles, edges, countDict)

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
            


        #rule_out_reverse = False

        # scaling scores based on intensity and frequency match
        maxInt, molRank, closestFreq, line_int_value = checkIntensity(tag, linelist, form, intensityReverse, freq)
        #rule_out_reverse = checkAllLines(linelist, form, tag, freq, peak_freqs_full, peak_ints_full, rms)
        tu2, subReport, scaledPer, offset, value, per = scaleScoreReverse(smile, validAtoms, subReport,
                                                                          sorted_smiles, sorted_values,
                                                                          freq, correctFreq, molRank,
                                                                          maxInt, form, newPreviousBest,
                                                                          newHighestIntensities,
                                                                          intensityReverse, iso, qn,
                                                                          newHighestSmiles,
                                                                          rule_out_reverse, line_int_value, linelist, tag)

        newIndexTestOrd.append(tu2)
        newIndexTest.append(tu2)
        report.append(subReport)
    allIndexTest.append(newIndexTestOrd)
    allReports.append(report)

    # obtaining and storing scores
    sortedNewTest, best, bestSmile, bestMol, bestScore, bestGlobalScore, testingScoresDictReverse, percentiles, soft, tuplesCombinedReverse, sortedTuplesCombinedReverse, topGlobalScoreSecond = getTopScores(
        newIndexTest)

    newTestingScoresListFinal.append(sortedNewTest)
    newCombinedScoresList.append(sortedTuplesCombinedReverse)
    newBestGlobalScoresFull.append(bestGlobalScore)
    topReverseSmiles.append(bestSmile)

    # updating override counter
    override = updateOverride(bestGlobalScore, globalThresh, bestMol, override)

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
#peak_indices = molsim.analysis.find_peaks(freq_arr, int_arr, res=resolution, min_sep=min_separation, sigma=sig)
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
    if peak_freqs[i] not in artifactFreqs and inAddedArt(peak_freqs[i], added_art) == False:
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
        if abs(u - q) <= 0.7:
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

print('Thank you for using this software! An interactive output (titled interactive_output.html) and a detailed line-by-line output (titled output.txt) are saved to your requested directory. Please send any questions/bugs to zfried@mit.edu')
