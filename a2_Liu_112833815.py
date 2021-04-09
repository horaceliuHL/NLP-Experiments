#!/usr/bin/python3
# Horace Liu 112833815
# CSE354, Spring 2021
##########################################################
## a2_Liu_112833815.py
## Word Sense Disambiguation

import sys
import re #regular expressions
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  #pytorch

sys.stdout = open('a2_Liu_112833815_OUTPUT.txt', 'w', encoding="utf-8")

##########################################################
## Part 1: Word Sense Disambiguation (WSD) with One-Hot Neighbors

## 1.1 (Reading and Processing Data)

#Tokenization method where I'm listing down the indices of the target word and removing the <head></head> tags
#and tokenizing the words by removing the lemma and POS

def tokenize(str):
  headMatch=re.compile(r'<head>([^<]+)</head>') #matches contents of head  
  tokens = str.split() #get the tokens

  headIndex = -1 #will be set to the index of the target word

  for i in range(len(tokens)):
    m = headMatch.match(tokens[i])
    if m: #a match: we are at the target token
      tokens[i] = m.groups()[0]
      headIndex = i

    split = tokens[i].split("/")
    tokens[i] = split[0].lower()
  # print(tokens)

  indicesOfHead.append(headIndex)

  return tokens


#Main readData method which reads from a .tsv file, converts their unique senses into numbers, and tokenizes

def readData(filename):
  readFile = pd.read_csv(filename, sep ='\t', names=["lemma.POS.id", "sense", "context"])
  temp = ['process%1:09:00::', 'process%1:04:00::', 'process%1:03:00::', 'process%1:10:00::', 'process%1:08:00::', 'process%1:09:01::']
  temp1 = ['machine%1:06:00::', 'machine%1:18:00::', 'machine%1:14:01::', 'machine%1:06:02::', 'machine%1:14:00::', 'machine%1:06:01::']
  temp2 = ['language%1:10:03::', 'language%1:10:01::', 'language%1:10:02::', 'language%1:09:00::', 'language%1:10:00::', 'language%1:09:01::']
  for i in range(len(readFile["sense"])):
    word = readFile["sense"][i]
    if readFile["lemma.POS.id"][i][0] == 'p':
      if word in temp:
        readFile["sense"][i] = temp.index(word)
      else:
        temp.append(word)
        readFile["sense"][i] = temp.index(word)
    elif readFile["lemma.POS.id"][i][0] == 'm':
      if word in temp1:
        readFile["sense"][i] = temp1.index(word)
      else:
        temp1.append(word)
        readFile["sense"][i] = temp1.index(word)
    elif readFile["lemma.POS.id"][i][0] == 'l':
      if word in temp2:
        readFile["sense"][i] = temp2.index(word)
      else:
        temp2.append(word)
        readFile["sense"][i] = temp2.index(word)

  for i in range(len(readFile["context"])):
    readFile["context"][i] = tokenize(readFile["context"][i])

  return readFile


## 1.2 (Adding One-Hot Feature Encodings)

#Getting the one-hot encodings of the word before the target word and using the top 2000 most frequent words

def getOneHotBefore(dictionary, contextList, word, actualIndex):
  result = [0]*2000
  placeOfWord = actualIndex
  # for i in range(len(contextList)):
  #   if word in contextList[i]:
  #     placeOfWord = i
  #     break
  if placeOfWord != 0 and contextList[placeOfWord - 1] in dictionary:
    result[dictionary[contextList[placeOfWord - 1]]] = 1
  
  return result


#Getting the one-hot encodings of the word after the target word and using the top 2000 most frequent words

def getOneHotAfter(dictionary, contextList, word, actualIndex):
  result = [0]*2000
  placeOfWord = actualIndex
  # for i in range(len(contextList)):
  #   if word in contextList[i]:
  #     placeOfWord = i
  #     break
  if placeOfWord != len(contextList) - 1 and contextList[placeOfWord + 1] in dictionary:
    result[dictionary[contextList[placeOfWord + 1]]] = 1

  return result


## 1.3 (Implementing the Logistic Regression class)

#The Logistic Regression classifier can handle multiple outcome classes with cross entropy loss

class LogReg(torch.nn.Module):
  def __init__(self, num_feats, senses, learn_rate = 0.01, device = torch.device("cpu") ):
      super(LogReg, self).__init__()
      self.linear = torch.nn.Linear(num_feats+1, senses) #add 1 to features for intercept

  def forward(self, X):
      #This is where the model itself is defined.
      #For logistic regression the model takes in X and returns
      #a probability (a value between 0 and 1)

      newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
      return self.linear(newX) #logistic function on the linear output


###################################################################################
## Part 2: Extracting PCA-Based Word Embeddings

## 2.1 (Converting the corpus into a co-occurrence matrix of the top 2000 words and OOV)

#Takes the training corpus and the top 2000 words and creates a co-occurrence matrix of size 2001x2001 with OOV

def convertCorpusIntoCoocurrenceMatrix(top2kWords, dataset):
  resultMatrix = np.zeros((2001, 2001), dtype=int)     
  for context in range(len(dataset)):
    lengthOfContext = dataset["context"][context]

    contextDict = {}
    for i in range(len(lengthOfContext)):
      if lengthOfContext[i] in contextDict:
        contextDict[lengthOfContext[i]] += 1
      else:
        contextDict[lengthOfContext[i]] = 1
    
    for i in contextDict:
      flag = False
      for j in contextDict:
        if flag == True:
          smaller = contextDict[i] if contextDict[i] < contextDict[j] else contextDict[j]
          if i in top2kWords and j in top2kWords:
            resultMatrix[top2kWords[i]][top2kWords[j]] += smaller
            resultMatrix[top2kWords[j]][top2kWords[i]] += smaller
          elif i not in top2kWords and j in top2kWords:
            resultMatrix[2000][top2kWords[j]] += smaller
            resultMatrix[top2kWords[j]][2000] += smaller
          elif i in top2kWords and j not in top2kWords:
            resultMatrix[top2kWords[i]][2000] += smaller
            resultMatrix[2000][top2kWords[i]] += smaller
          else:
            resultMatrix[2000][2000] += smaller
        elif i == j:
          if i in top2kWords:
            resultMatrix[top2kWords[i]][top2kWords[i]] += int(contextDict[i]/2)
          else:
            resultMatrix[2000][2000] += int(contextDict[i]/2)
          flag = True
        
    # for i in range(len(lengthOfContext) - 1):
    #   for j in range(i + 1, len(lengthOfContext)):
    #     if lengthOfContext[i] in top2kWords and lengthOfContext[j] in top2kWords:
    #       resultMatrix[top2kWords[lengthOfContext[i]]][top2kWords[lengthOfContext[j]]] += 1
    #       resultMatrix[top2kWords[lengthOfContext[j]]][top2kWords[lengthOfContext[i]]] += 1
    #     elif lengthOfContext[i] not in top2kWords and lengthOfContext[j] in top2kWords:
    #       resultMatrix[2000][top2kWords[lengthOfContext[j]]] += 1
    #       resultMatrix[top2kWords[lengthOfContext[j]]][2000] += 1
    #     elif lengthOfContext[i] in top2kWords and lengthOfContext[j] not in top2kWords:
    #       resultMatrix[top2kWords[lengthOfContext[i]]][2000] += 1
    #       resultMatrix[2000][top2kWords[lengthOfContext[i]]] += 1
    #     else:
    #       resultMatrix[2000][2000] += 1

  return resultMatrix


## 2.2 (Running PCA and extracting embeddings)

#Takes the co-occurrence matrix, conducts dimensionality reduction, and returns the resulting U matrix with 
#static, 50 dimensional embeddings

def runPCAAndExtractEmbeddings(data, mostCommonWordsList):
  data = (data - np.mean(data)) / np.std(data)
  data = torch.from_numpy(data)
  u, s, v = torch.svd(data)
  resultEmbedding = {}
  for i in range(2000):
    resultEmbedding[mostCommonWordsList[i]] = u[i][0:50]
  resultEmbedding['OOV'] = u[2000][0:50]
  return resultEmbedding


## 2.3 (Find Euclidian distance)

#Takes the embedding vectors of two words and calculates the euclidian distance between them to check the 
#distance between two words

def findEuclidianDistance(tensorVector, tensorVector1):
  tensorVector = tensorVector.numpy()
  tensorVector1 = tensorVector1.numpy()
  return np.linalg.norm(tensorVector - tensorVector1)


###################################################################################
## Part 3: WSD with Embeddings

## 3.1 (Extract embedding features)

#Extracting and concatenating 4 sets of embeddings which represent two words before, one word before, one word
#after, and two words after respectively for each context

def extractEmbeddingFeatures(embeddings, dataset, actualIndex):
  entireEmbedVector = []
  for i in range(len(dataset)):
    context = dataset[i]
    wordIndex = actualIndex[i]
    embedVector = []
    if wordIndex - 2 >= 0:
      if context[wordIndex - 2] in embeddings:
        embedVector += embeddings[context[wordIndex - 2]].tolist()
      else:
        embedVector += embeddings['OOV'].tolist()
    else:
      embedVector += [0]*50
    if wordIndex - 1 >= 0:
      if context[wordIndex - 1] in embeddings:
        embedVector += embeddings[context[wordIndex - 1]].tolist()
      else:
        embedVector += embeddings['OOV'].tolist()
    else:
      embedVector += [0]*50
    if wordIndex + 1 < len(context):
      if context[wordIndex + 1] in embeddings:
        embedVector += embeddings[context[wordIndex + 1]].tolist()
      else:
        embedVector += embeddings['OOV'].tolist()
    else:
      embedVector += [0]*50
    if wordIndex + 2 < len(context):
      if context[wordIndex + 2] in embeddings:
        embedVector += embeddings[context[wordIndex + 2]].tolist()
      else:
        embedVector += embeddings['OOV'].tolist()
    else:
      embedVector += [0]*50
    entireEmbedVector.append(embedVector)
  return entireEmbedVector


###################################################################################
## MAIN

if __name__ == "__main__":

  ## Part 1: Word Sense Disambiguation (WSD) with One-Hot Neighbors

  #1.1 (Reading the data)
  indicesOfHead = []
  val = readData("onesec_train.tsv")

  #Grabbing the most common words
  commonWords = {}
  for i in range(len(val["context"])):
    listOfWords = val["context"][i]
    for j in range(len(listOfWords)):
      listOfWords[j] = listOfWords[j].lower()
      if listOfWords[j] in commonWords:
        commonWords[listOfWords[j]] += 1
      else:
        commonWords[listOfWords[j]] = 1
  
  sorted_commonWords = dict(sorted(commonWords.items(), key=lambda item: (item[1], item[0]), reverse=True))

  keys = list(sorted_commonWords.keys())
  top2kCommon = keys[0: 2000]

  # print(list(sorted_commonWords.items())[0:2000])

  top2kDict = {}
  for i in range(len(top2kCommon)):
    top2kDict[top2kCommon[i]] = i


  #1.2 (Adding the one-hot feature encoding for each target word)
  contextForProcess = []
  contextForMachine = []
  contextForLanguage = []

  for i in range(len(val)):
    if val["lemma.POS.id"][i][0] == 'p':
      contextForProcess.append(val["context"][i])
    elif val["lemma.POS.id"][i][0] == 'm':
      contextForMachine.append(val["context"][i])
    elif val["lemma.POS.id"][i][0] == 'l':
      contextForLanguage.append(val["context"][i])


  #get One Hot for Process
  oneHotProcess = []
  indicesOfProcess = indicesOfHead[0: len(contextForProcess)]
  countIndexProcess = 0
  for context in contextForProcess:
    oneHotProcess.append(getOneHotBefore(top2kDict, context, "process", indicesOfProcess[countIndexProcess]) + getOneHotAfter(top2kDict, context, "process", indicesOfProcess[countIndexProcess]))
    countIndexProcess = countIndexProcess + 1


  #get One Hot for Machine
  oneHotMachine = []
  indicesOfMachine = indicesOfHead[len(contextForProcess): (len(contextForProcess) + len(contextForMachine))]
  countIndexMachine = 0
  for context in contextForMachine:
    oneHotMachine.append(getOneHotBefore(top2kDict, context, "machine", indicesOfMachine[countIndexMachine]) + getOneHotAfter(top2kDict, context, "machine", indicesOfMachine[countIndexMachine]))
    countIndexMachine = countIndexMachine + 1


  #get One Hot for Language
  oneHotLanguage = []
  indicesOfLanguage = indicesOfHead[(len(contextForProcess) + len(contextForMachine)): ]
  countIndexLanguage = 0
  for context in contextForLanguage:
    oneHotLanguage.append(getOneHotBefore(top2kDict, context, "language", indicesOfLanguage[countIndexLanguage]) + getOneHotAfter(top2kDict, context, "language", indicesOfLanguage[countIndexLanguage]))
    countIndexLanguage = countIndexLanguage + 1



  #1.3 (Training the logistic regression classifiers)

###Training for Process

  process1DTensor = torch.tensor(val["sense"][0: len(contextForProcess)])

  learning_rate, epochs = 1, 30
  modelProcess = LogReg(4000, 6)
  sgdProcess = torch.optim.SGD(modelProcess.parameters(), lr=learning_rate)
  loss_func_Process = torch.nn.CrossEntropyLoss() #includes log

  #training loop:
  for i in range(epochs):
    modelProcess.train()
    sgdProcess.zero_grad()
    #forward pass:
    ypred = modelProcess(torch.tensor(oneHotProcess))
    loss = loss_func_Process(ypred, process1DTensor)
    #backward: /(applies gradient descent)
    loss.backward()
    sgdProcess.step()


 ###Training for Machine

  machine1DTensor = val["sense"][len(contextForProcess): len(contextForProcess) + len(contextForMachine)].reset_index()
  machine1DTensor = torch.tensor(machine1DTensor["sense"])

  learning_rate, epochs = 0.1, 30
  modelMachine = LogReg(4000, 6)
  sgdMachine = torch.optim.SGD(modelMachine.parameters(), lr=learning_rate, weight_decay=0.01)
  loss_func_Machine = torch.nn.CrossEntropyLoss() #includes log

  #training loop:
  for i in range(epochs):
    modelMachine.train()
    sgdMachine.zero_grad()
    #forward pass:
    ypred = modelMachine(torch.tensor(oneHotMachine))
    loss = loss_func_Machine(ypred, machine1DTensor)
    #backward: /(applies gradient descent)
    loss.backward()
    sgdMachine.step()

  
  #Training for Language

  language1DTensor = val["sense"][len(contextForProcess) + len(contextForMachine): ].reset_index()
  language1DTensor = torch.tensor(language1DTensor["sense"])

  learning_rate, epochs = 1.0, 30
  modelLanguage = LogReg(4000, 6)
  sgdLanguage = torch.optim.SGD(modelLanguage.parameters(), lr=learning_rate)
  loss_func_Language = torch.nn.CrossEntropyLoss() #includes log

  #training loop:
  for i in range(epochs):
    modelLanguage.train()
    sgdLanguage.zero_grad()
    #forward pass:
    ypred = modelLanguage(torch.tensor(oneHotLanguage))
    loss = loss_func_Language(ypred, language1DTensor)
    #backward: /(applies gradient descent)
    loss.backward()
    sgdLanguage.step()


  #1.4 (Testing each classifier model on the test set)

  #Reading the data from the test set
  indicesOfHead = []
  val1 = readData("onesec_test.tsv")

  #Getting most common words from the test set
  commonWords = {}
  for i in range(len(val1["context"])):
    listOfWords = val1["context"][i]
    for j in range(len(listOfWords)):
      listOfWords[j] = listOfWords[j].lower()
      if listOfWords[j] in commonWords:
        commonWords[listOfWords[j]] += 1
      else:
        commonWords[listOfWords[j]] = 1

  sorted_commonWords = dict(sorted(commonWords.items(), key=lambda item: (item[1], item[0]), reverse=True))

  keys = list(sorted_commonWords.keys())
  top2kCommon1 = keys[0: 2000]

  top2kDict1 = {}
  for i in range(len(top2kCommon1)):
    top2kDict1[top2kCommon1[i]] = i


  contextForProcess = []
  contextForMachine = []
  contextForLanguage = []

  for i in range(len(val1)):
    if val1["lemma.POS.id"][i][0] == 'p':
      contextForProcess.append(val1["context"][i])
    elif val1["lemma.POS.id"][i][0] == 'm':
      contextForMachine.append(val1["context"][i])
    elif val1["lemma.POS.id"][i][0] == 'l':
      contextForLanguage.append(val1["context"][i])


  #get One Hot for Process
  oneHotProcess = []
  indicesOfProcess = indicesOfHead[0: len(contextForProcess)]
  countIndexProcess = 0
  for context in contextForProcess:
    oneHotProcess.append(getOneHotBefore(top2kDict1, context, "process", indicesOfProcess[countIndexProcess]) + getOneHotAfter(top2kDict1, context, "process", indicesOfProcess[countIndexProcess]))
    countIndexProcess = countIndexProcess + 1

  #get One Hot for Machine
  oneHotMachine = []
  indicesOfMachine = indicesOfHead[len(contextForProcess): (len(contextForProcess) + len(contextForMachine))]
  countIndexMachine = 0
  for context in contextForMachine:
    oneHotMachine.append(getOneHotBefore(top2kDict1, context, "machine", indicesOfMachine[countIndexMachine]) + getOneHotAfter(top2kDict1, context, "machine", indicesOfMachine[countIndexMachine]))
    countIndexMachine = countIndexMachine + 1

  #get One Hot for Language
  oneHotLanguage = []
  indicesOfLanguage = indicesOfHead[(len(contextForProcess) + len(contextForMachine)): ]
  countIndexLanguage = 0
  for context in contextForLanguage:
    oneHotLanguage.append(getOneHotBefore(top2kDict1, context, "language", indicesOfLanguage[countIndexLanguage]) + getOneHotAfter(top2kDict1, context, "language", indicesOfLanguage[countIndexLanguage]))
    countIndexLanguage = countIndexLanguage + 1

  #Predicting the test data and comparing it against the actual senses
  print("[TESTING WSD MODEL WITH ONE-HOT NEIGHBORS]")

  with torch.no_grad():
    ytestpred_prob = modelProcess(torch.tensor(oneHotProcess))
    ytestpred_prob = ytestpred_prob.cpu().detach().numpy()
    print("process")
    print("\tpredictions for process.NOUN.000018: " + str(ytestpred_prob[3][0:]))
    print("\tpredictions for process.NOUN.000024: " + str(ytestpred_prob[4][0:]))
    calculatedTestPred = []
    for i in range(len(ytestpred_prob)):
      largest = 0
      indexOfLargest = 0
      for j in range(len(ytestpred_prob[i])):
        if ytestpred_prob[i][j] > largest:
          largest = ytestpred_prob[i][j]
          indexOfLargest = j
      calculatedTestPred.append(indexOfLargest)
    count = 0
    for i in range(len(calculatedTestPred)):
      if calculatedTestPred[i] == val1["sense"][i]:
        count = count + 1
    print("\tcorrect: " + str(count) + " out of " + str(len(calculatedTestPred)))



  with torch.no_grad():
    ytestpred_prob = modelMachine(torch.tensor(oneHotMachine))
    ytestpred_prob = ytestpred_prob.cpu().detach().numpy()
    print("machine")
    print("\tpredictions for machine.NOUN.000004: " + str(ytestpred_prob[0][0:]))
    print("\tpredictions for machine.NOUN.000008: " + str(ytestpred_prob[1][0:]))
    calculatedTestPred = []
    for i in range(len(ytestpred_prob)):
      largest = 0
      indexOfLargest = 0
      for j in range(len(ytestpred_prob[i])):
        if ytestpred_prob[i][j] > largest:
          largest = ytestpred_prob[i][j]
          indexOfLargest = j
      calculatedTestPred.append(indexOfLargest)
    count = 0
    resetIndex = val1["sense"][len(contextForProcess): (len(contextForProcess) + len(contextForMachine))].reset_index()
    resetIndex = resetIndex["sense"]
    for i in range(len(calculatedTestPred)):
      if calculatedTestPred[i] == resetIndex[i]:
        count = count + 1
    print("\tcorrect: " + str(count) + " out of " + str(len(calculatedTestPred)))


  with torch.no_grad():
    ytestpred_prob = modelLanguage(torch.tensor(oneHotLanguage))
    ytestpred_prob = ytestpred_prob.cpu().detach().numpy()
    print("language")
    print("\tpredictions for language.NOUN.000008: " + str(ytestpred_prob[1][0:]))
    print("\tpredictions for language.NOUN.000014: " + str(ytestpred_prob[2][0:]))
    calculatedTestPred = []
    for i in range(len(ytestpred_prob)):
      largest = 0
      indexOfLargest = 0
      for j in range(len(ytestpred_prob[i])):
        if ytestpred_prob[i][j] > largest:
          largest = ytestpred_prob[i][j]
          indexOfLargest = j
      calculatedTestPred.append(indexOfLargest)
    count = 0
    resetIndex = val1["sense"][(len(contextForProcess) + len(contextForMachine)): ].reset_index()
    resetIndex = resetIndex["sense"]
    for i in range(len(calculatedTestPred)):
      if calculatedTestPred[i] == resetIndex[i]:
        count = count + 1
    print("\tcorrect: " + str(count) + " out of " + str(len(calculatedTestPred)))

#############################################################################
  ## Part 2: Extracting PCA-Based Word Embeddings

  #2.1 (Converting the corpus into a co-occurrence matrix)
  coOccurMatrix = convertCorpusIntoCoocurrenceMatrix(top2kDict, val)
  # print(coOccurMatrix)

  # #takes the co-occurrence matrix from a saved file to optimize time (should not be used for actual testing)
  # file = open("a2_temp1.txt", "r")
  # coOccurMatrix = file.read()
  # coOccurMatrix = coOccurMatrix.replace('\n', '').split(' ')
  # actualMatrix = []
  # tempInsert = []
  # for i in range(len(coOccurMatrix)):
  #   if ']' in coOccurMatrix[i]:
  #     tempNum = coOccurMatrix[i].replace(']', '')
  #     if tempNum != '':
  #       tempInsert.append(int(tempNum))
  #     actualMatrix.append(tempInsert)
  #     tempInsert = []
  #   elif '[' not in coOccurMatrix[i] and coOccurMatrix[i] != '':
  #     tempInsert.append(int(coOccurMatrix[i]))

  actualMatrix = np.array(coOccurMatrix, dtype=float)

  #2.2 (Running PCA and extracting embeddings)
  embeddingDict = runPCAAndExtractEmbeddings(actualMatrix, top2kCommon)

  #2.3 (Finding the euclidian distance between certain words)
  distBetweenLanguageProcess = findEuclidianDistance(embeddingDict['language'], embeddingDict['process'])
  distBetweenMachineProcess = findEuclidianDistance(embeddingDict['machine'], embeddingDict['process'])
  distBetweenLanguageSpeak = findEuclidianDistance(embeddingDict['language'], embeddingDict['speak'])
  distBetweenWordWords = findEuclidianDistance(embeddingDict['word'], embeddingDict['words'])
  distBetweenWordThe = findEuclidianDistance(embeddingDict['word'], embeddingDict['the'])

  print("\n[EUCLIDIAN DISTANCE BETWEEN PCA-BASED WORD EMBEDDINGS]")
  print("\t('language', 'process') : " + str(distBetweenLanguageProcess))
  print("\t('machine', 'process') : " + str(distBetweenMachineProcess))
  print("\t('language', 'speak') : " + str(distBetweenLanguageSpeak))
  print("\t('word', 'words') : " + str(distBetweenWordWords))
  print("\t('word', 'the') : " + str(distBetweenWordThe))

###########################################################################
  ## Part 3: WSD with Embeddings

  #3.1 (Extract embedding features)
  indicesOfHead = []
  val = readData("onesec_train.tsv")

  processContextList = []
  machineContextList = []
  languageContextList = []

  for i in range(len(val)):
    if val["lemma.POS.id"][i][0] == 'p':
      processContextList.append(val["context"][i])
    elif val["lemma.POS.id"][i][0] == 'm':
      machineContextList.append(val["context"][i])
    elif val["lemma.POS.id"][i][0] == 'l':
      languageContextList.append(val["context"][i])

  embeddingsForProcess = extractEmbeddingFeatures(embeddingDict, processContextList, indicesOfHead)

  indicesOfHeadForMachine = indicesOfHead[len(processContextList): (len(processContextList) + len(machineContextList))]
  embeddingsForMachine = extractEmbeddingFeatures(embeddingDict, machineContextList, indicesOfHeadForMachine)

  indicesOfHeadForLanguage = indicesOfHead[(len(processContextList) + len(machineContextList)): ]
  embeddingsForLanguage = extractEmbeddingFeatures(embeddingDict, languageContextList, indicesOfHeadForLanguage)


  #3.2 (Rerun logistic regression training using the word embeddings)

  ###Training for Process

  process1DTensor = torch.tensor(val["sense"][0: len(processContextList)])

  learning_rate, epochs = 1, 100
  modelProcess = LogReg(200, 6)
  sgd = torch.optim.SGD(modelProcess.parameters(), lr=learning_rate)
  loss_func = nn.CrossEntropyLoss() #includes log

  #training loop:
  for i in range(epochs):
    modelProcess.train()
    sgd.zero_grad()
    #forward pass:
    ypred = modelProcess(torch.tensor(embeddingsForProcess))
    loss = loss_func(ypred, process1DTensor)
    #backward: /(applies gradient descent)
    loss.backward()
    sgd.step()


  ###Training for Machine

  machine1DTensor = val["sense"][len(processContextList): (len(processContextList) + len(machineContextList))].reset_index()
  machine1DTensor = torch.tensor(machine1DTensor["sense"])

  learning_rate, epochs = 1, 100
  modelMachine = LogReg(200, 6)
  sgdMachine1 = torch.optim.SGD(modelMachine.parameters(), lr=learning_rate, weight_decay=0.05)
  loss_func = nn.CrossEntropyLoss() #includes log

  #training loop:
  for i in range(epochs):
    modelMachine.train()
    sgdMachine1.zero_grad()
    #forward pass:
    ypred = modelMachine(torch.tensor(embeddingsForMachine))
    loss = loss_func(ypred, machine1DTensor)
    #backward: /(applies gradient descent)
    loss.backward()
    sgdMachine1.step()

  
  ###Training for Language

  language1DTensor = val["sense"][(len(processContextList) + len(machineContextList)): ].reset_index()
  language1DTensor = torch.tensor(language1DTensor["sense"])

  learning_rate, epochs = 1, 300
  modelLanguage = LogReg(200, 6)
  sgdLanguage1 = torch.optim.SGD(modelLanguage.parameters(), lr=learning_rate)
  loss_func_Language1 = nn.CrossEntropyLoss() #includes log

  #training loop:
  for i in range(epochs):
    modelLanguage.train()
    sgdLanguage1.zero_grad()
    #forward pass:
    ypred = modelLanguage(torch.tensor(embeddingsForLanguage))
    loss = loss_func_Language1(ypred, language1DTensor)
    #backward: /(applies gradient descent)
    loss.backward()
    sgdLanguage1.step()


  #3.3 (Testing the new logistic regression classifier on the test data)

  indicesOfHead = []
  val1 = readData("onesec_test.tsv")

  coOccurMatrixTest = convertCorpusIntoCoocurrenceMatrix(top2kDict1, val1)
  coOccurMatrixTest = np.array(coOccurMatrixTest, dtype=float)

  embeddingDict = runPCAAndExtractEmbeddings(coOccurMatrixTest, top2kCommon1)

  processContextList = []
  machineContextList = []
  languageContextList = []

  for i in range(len(val1)):
    if val1["lemma.POS.id"][i][0] == 'p':
      processContextList.append(val1["context"][i])
    elif val1["lemma.POS.id"][i][0] == 'm':
      machineContextList.append(val1["context"][i])
    elif val1["lemma.POS.id"][i][0] == 'l':
      languageContextList.append(val1["context"][i])

  embeddingsForProcess = extractEmbeddingFeatures(embeddingDict, processContextList, indicesOfHead)

  indicesOfHeadForMachine = indicesOfHead[len(processContextList): (len(processContextList) + len(machineContextList))]
  embeddingsForMachine = extractEmbeddingFeatures(embeddingDict, machineContextList, indicesOfHeadForMachine)

  indicesOfHeadForLanguage = indicesOfHead[(len(processContextList) + len(machineContextList)): ]
  embeddingsForLanguage = extractEmbeddingFeatures(embeddingDict, languageContextList, indicesOfHeadForLanguage)


print("\n[TESTING WSD MODEL WITH EMBEDDINGS]")

with torch.no_grad():
  ytestpred_prob = modelProcess(torch.tensor(embeddingsForProcess))
  ytestpred_prob = ytestpred_prob.cpu().detach().numpy()
  print("process")
  print("\tpredictions for process.NOUN.000018: " + str(ytestpred_prob[3][0:]))
  print("\tpredictions for process.NOUN.000024: " + str(ytestpred_prob[4][0:]))
  calculatedTestPred = []
  for i in range(len(ytestpred_prob)):
    largest = 0
    indexOfLargest = 0
    for j in range(len(ytestpred_prob[i])):
      if ytestpred_prob[i][j] > largest:
        largest = ytestpred_prob[i][j]
        indexOfLargest = j
    calculatedTestPred.append(indexOfLargest)
  count = 0
  for i in range(len(calculatedTestPred)):
    if calculatedTestPred[i] == val1["sense"][i]:
      count = count + 1
  print("\tcorrect: " + str(count) + " out of " + str(len(calculatedTestPred)))


with torch.no_grad():
  ytestpred_prob = modelMachine(torch.tensor(embeddingsForMachine))
  ytestpred_prob = ytestpred_prob.cpu().detach().numpy()
  print("machine")
  print("\tpredictions for machine.NOUN.000004: " + str(ytestpred_prob[0][0:]))
  print("\tpredictions for machine.NOUN.000008: " + str(ytestpred_prob[1][0:]))
  calculatedTestPred = []
  for i in range(len(ytestpred_prob)):
    largest = 0
    indexOfLargest = 0
    for j in range(len(ytestpred_prob[i])):
      if ytestpred_prob[i][j] > largest:
        largest = ytestpred_prob[i][j]
        indexOfLargest = j
    calculatedTestPred.append(indexOfLargest)
  count = 0
  resetIndex = val1["sense"][len(processContextList): (len(processContextList) + len(machineContextList))].reset_index()
  resetIndex = resetIndex["sense"]
  for i in range(len(calculatedTestPred)):
    if calculatedTestPred[i] == resetIndex[i]:
      count = count + 1
  print("\tcorrect: " + str(count) + " out of " + str(len(calculatedTestPred)))


with torch.no_grad():
  ytestpred_prob = modelLanguage(torch.tensor(embeddingsForLanguage))
  ytestpred_prob = ytestpred_prob.cpu().detach().numpy()
  print("language")
  print("\tpredictions for language.NOUN.000008: " + str(ytestpred_prob[1][0:]))
  print("\tpredictions for language.NOUN.000014: " + str(ytestpred_prob[2][0:]))
  calculatedTestPred = []
  for i in range(len(ytestpred_prob)):
    largest = 0
    indexOfLargest = 0
    for j in range(len(ytestpred_prob[i])):
      if ytestpred_prob[i][j] > largest:
        largest = ytestpred_prob[i][j]
        indexOfLargest = j
    calculatedTestPred.append(indexOfLargest)
  count = 0
  resetIndex = val1["sense"][(len(processContextList) + len(machineContextList)): ].reset_index()
  resetIndex = resetIndex["sense"]
  for i in range(len(calculatedTestPred)):
    if calculatedTestPred[i] == resetIndex[i]:
      count = count + 1
  print("\tcorrect: " + str(count) + " out of " + str(len(calculatedTestPred)))
