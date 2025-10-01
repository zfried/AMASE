"""
Output text file creation for AMASE.
Generates a summary report of line assignments and categorizations.
"""


import os 
from config import SCORE_THRESHOLD, GLOBAL_THRESHOLD_ORIGINAL


def create_output(direc, startingMols, newTestingScoresListFinal, newCombinedScoresList, actualFrequencies, allIndexTest, allReports):
    '''
    
    Function to create output text file summarizing results of line assignments.
    
    '''

    print('')
    print('Creating output text file...')
    print('')
    f = open(os.path.join(direc, 'output_report.txt'), "w")
    f.write('Initial Detected Molecules: ')
    if len(startingMols) == 0:
        f.write('Nothing inputted')
    else:
        for u in startingMols:
            f.write(str(u))
            f.write(' ')
    f.write('\n')
    f.write('--------------------------------------\n')
    category = []

    mulCount = 0
    assignCount = 0
    unCount = 0
    for i in range(len(newTestingScoresListFinal)):
        f.write('LINE ' + str(i + 1) + ':\n')
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

        if newTest[0][1] < GLOBAL_THRESHOLD_ORIGINAL:
            category.append('Unidentified')
            f.write('Unidentified \n')
            unCount += 1
        elif newComb[0][1] < SCORE_THRESHOLD:
            category.append('Several Possible Carriers')
            mulCount += 1
            f.write('Several Possible Carriers \n')
            f.write('Listed from highest to least ranked, these are: \n')
            for u in finalScores:
                if u[2] > GLOBAL_THRESHOLD_ORIGINAL:
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
