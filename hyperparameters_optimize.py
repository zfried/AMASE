import random
import numpy as np
import pandas as pd
import time
import statistics
import os
import scipy
from scipy import stats


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
        fullList.append(newList)

    return fullList

def getCountDictionary(edgeNum):
    '''
    Format to convert edge connection counts into format required for algorithm.
    '''
    countDict = {}
    full = pd.read_csv(os.path.join(direc, 'counts_' + str(edgeNum) + '.csv'))
    smiles = list(full['smiles'])
    edgeCount = list(full['count'])

    for i in range(len(smiles)):
        countDict[smiles[i]] = edgeCount[i]
    return countDict, edgeCount



'''
The next three functions are necessary to convert the graph information into the required format.
'''



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


def runPageRankInit2_Final(smiles, detectedSmiles, edges, countDict):
    '''
    This function runs the graph based ranking system given the detected molecules.
    It returns a score for all molecules in the graph.
    '''
    tick = time.perf_counter()
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
    tock = time.perf_counter()
    print('converged on iteration:')
    print(i)

    # sorting the results
    sorted_dict = sorted(nodeDict.items(), key=lambda x: x[1])
    sorted_dict.reverse()
    sorted_smiles = [q[0] for q in sorted_dict]
    sorted_values = [q[1] for q in sorted_dict]

    time_taken = tock - tick

    return sorted_dict, sorted_smiles, sorted_values, time_taken



def divide_list(lst, num_chunks=5, seed=42):
    '''
    Function to make data splits for five-fold cross validation
    '''

    # Shuffle the list with a fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(lst)

    # Calculate the length of each chunk
    avg_len = len(lst) / float(num_chunks)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg_len)])
        last += avg_len

    return out


direc = os.getcwd()

#edge distance thresholds to test
edgeDistances = [9,9.5,10,10.25,10.5,10.75,11,12]

#creating dataset chunks for cross validation on the o2/benzene dataset
df = pd.read_csv(os.path.join(direc, 'o2_benzene_dataset_final.csv'))
fullSmiles = list(df['original smiles'])
fullSmiles2 = []
for i in fullSmiles:
    if i not in fullSmiles2:
        fullSmiles2.append(i)

chunks = divide_list(fullSmiles2)

#looping through all distance thresholds to determine optimal hyperparamters
for dis in edgeDistances:

    #uploading datasets
    edge = pd.read_csv(os.path.join(direc, 'edges_' + str(dis) + '.csv'))
    edges = edgeStringToList(list(edge['edges']))

    full = pd.read_csv(os.path.join(direc, 'all_smiles.csv'))
    smiles = list(full['smiles'])

    countDict, edgeCount = getCountDictionary(dis)

    vectorDF = pd.read_csv(os.path.join(direc, 'all_vectors.csv'))
    vectorSmiles = list(vectorDF['smiles'])
    allVectors = np.array(stringToList(list(vectorDF['vectors'])))

    del vectorDF
    del full
    del edge

    #storing the average number of edge connections per node
    meanConnections = statistics.mean(edgeCount)

    times = []
    totalPers = []
    totalScores = []
    for c in range(len(chunks)):

        detectedSmiles = {}
        validation = chunks[c]

        for q in range(len(chunks)):
            if q != c:
                for o in chunks[q]:
                    detectedSmiles[o] = 1

        print(detectedSmiles)

        #running th egraph calculation
        sorted_dict, sorted_smiles, sorted_values, time_taken = runPageRankInit2_Final(smiles, detectedSmiles, edges, countDict)

        newSmiles = [z[0] for z in sorted_dict]
        newValues = [z[1] for z in sorted_dict]

        valPers = []
        valScores = []

        #storing the results on the validation set
        for z in validation:
            newIdx = newSmiles.index(z)
            value = newValues[newIdx]
            per = stats.percentileofscore(newValues, value)
            valScores.append(newIdx-len(chunks[c]))
            valPers.append(per)


        for u in range(len(valPers)):
            totalPers.append(valPers[u])
            totalScores.append(valScores[u])

        times.append(time_taken)

    #printing results
    print('Report for a distance threshold of ' + str(dis))
    print('Percentile ranking of all molecules in the various validation sets')
    print(totalPers)
    print('')
    print('Median percentile ranking')
    print(statistics.median(totalPers))
    print('Mean percentile ranking')
    print(statistics.mean(totalPers))
    print('')
    print('')
    print('Actual ranking of all molecules in the various validation sets')
    print(totalScores)
    print('')
    print('Median actual ranking')
    print(statistics.median(totalScores))
    print('Mean actual ranking')
    print(statistics.mean(totalScores))
    print()
    print('Median time per calcualtion')
    print(statistics.median(times))
    print('Mean time per calculation')
    print(statistics.mean(times))
    print('')
    print('')
    print('')
    print('------------------------------------')



`
    





