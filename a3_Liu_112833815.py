#!/usr/bin/python3
# Horace Liu 112833815
# CSE354, Spring 2021
##########################################################
## a3_Liu_112833815.py
## Word Sense Disambiguation

import sys
import re #regular expressions
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  #pytorch

sys.stdout = open('a3_Liu_112833815_OUTPUT.txt', 'w', encoding="utf-8")

##########################################################
## Part 2: Create a probabilistic language model

## 2.1: Preparing corpus and vocabulary

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
  # temp = ['process%1:09:00::', 'process%1:04:00::', 'process%1:03:00::', 'process%1:10:00::', 'process%1:08:00::', 'process%1:09:01::']
  # temp1 = ['machine%1:06:00::', 'machine%1:18:00::', 'machine%1:14:01::', 'machine%1:06:02::', 'machine%1:14:00::', 'machine%1:06:01::']
  # temp2 = ['language%1:10:03::', 'language%1:10:01::', 'language%1:10:02::', 'language%1:09:00::', 'language%1:10:00::', 'language%1:09:01::']
  # for i in range(len(readFile["sense"])):
  #   word = readFile["sense"][i]
  #   if readFile["lemma.POS.id"][i][0] == 'p':
  #     if word in temp:
  #       readFile["sense"][i] = temp.index(word)
  #     else:
  #       temp.append(word)
  #       readFile["sense"][i] = temp.index(word)
  #   elif readFile["lemma.POS.id"][i][0] == 'm':
  #     if word in temp1:
  #       readFile["sense"][i] = temp1.index(word)
  #     else:
  #       temp1.append(word)
  #       readFile["sense"][i] = temp1.index(word)
  #   elif readFile["lemma.POS.id"][i][0] == 'l':
  #     if word in temp2:
  #       readFile["sense"][i] = temp2.index(word)
  #     else:
  #       temp2.append(word)
  #       readFile["sense"][i] = temp2.index(word)

  for i in range(len(readFile["context"])):
    readFile["context"][i] = tokenize(readFile["context"][i])
    readFile["context"][i].insert(0, "<s>")
    readFile["context"][i].append("</s>")

  return readFile


def getTop5k(val):
  commonWords = {}
  for i in range(len(val["context"])):
    listOfWords = val["context"][i]
    for j in range(len(listOfWords)):
      listOfWords[j] = listOfWords[j].lower()
      if listOfWords[j] == '<s>' or listOfWords[j] == '</s>':
        continue
      if listOfWords[j] in commonWords:
        commonWords[listOfWords[j]] += 1
      else:
        commonWords[listOfWords[j]] = 1
  
  sorted_commonWords = dict(sorted(commonWords.items(), key=lambda item: (item[1], item[0]), reverse=True)[:5000])
  # sorted_commonWords = dict(sorted(commonWords.items(), key=lambda item: (-item[1], item[0]))[:5000])
  # sorted_commonWords = dict(sorted(commonWords.items(), key=lambda item: (item[1]), reverse=True)[:5000])

  # keys = list(sorted_commonWords.keys())
  # top5kCommon = keys[0: 5000]
  # return top5kCommon
  return sorted_commonWords


## 2.2: Extract unigram, bigram, and trigram counts

def getUnigrams(dict5K, val):
  resultUnigram = {}
  for i in range(len(val["context"])):
    word = val["context"][i]
    for j in range(len(word)):
      if word[j] == '<s>' or word[j] == '</s>':
        continue
      if word[j] in dict5K:
        if word[j] in resultUnigram:
          resultUnigram[word[j]] += 1
        else:
          resultUnigram[word[j]] = 1
      else:
        if 'OOV' in resultUnigram:
          resultUnigram['OOV'] += 1
        else:
          resultUnigram['OOV'] = 1
  # sorted_commonWords = dict(sorted(resultUnigram.items(), key=lambda item: (item[1]), reverse=True))
  # print(sorted_commonWords)
  return resultUnigram


def getBigrams(dict5K, val):
  resultBigram = {}
  for i in range(len(val["context"])):
    word = val["context"][i]
    for j in range(len(word) - 1):
      if word[j] == '<s>' or word[j] == '</s>':
        continue
      if word[j] in dict5K:
        if word[j] in resultBigram:
          if word[j+1] in dict5K:
            if word[j+1] in resultBigram[word[j]]:
              resultBigram[word[j]][word[j+1]] += 1
            else:
              resultBigram[word[j]][word[j+1]] = 1
          else:
            if 'OOV' in resultBigram[word[j]]:
              resultBigram[word[j]]['OOV'] += 1
            else:
              resultBigram[word[j]]['OOV'] = 1
        else:
          resultBigram[word[j]] = {}
          if word[j+1] in dict5K:
            resultBigram[word[j]][word[j+1]] = 1
          else:
            resultBigram[word[j]]['OOV'] = 1
      else:
        if 'OOV' in resultBigram:
          if word[j+1] in dict5K:
            if word[j+1] in resultBigram['OOV']:
              resultBigram['OOV'][word[j+1]] += 1
            else:
              resultBigram['OOV'][word[j+1]] = 1
          else:
            if 'OOV' in resultBigram['OOV']:
              resultBigram['OOV']['OOV'] += 1
            else:
              resultBigram['OOV']['OOV'] = 1
        else:
          resultBigram['OOV'] = {}
          if word[j+1] in dict5K:
            resultBigram['OOV'][word[j+1]] = 1
          else:
            resultBigram['OOV']['OOV'] = 1
            
  return resultBigram


def getTrigrams(dict5K, val, bigramDict):
  resultTrigram = {}
  for i in range(len(val["context"])):
    word = val["context"][i]
    for j in range(len(word) - 2):
      if word[j] == '<s>' or word[j] == '</s>':
        continue
      if word[j] in bigramDict:
        if word[j+1] in bigramDict[word[j]]:
          if (word[j], word[j+1]) in resultTrigram:
            if word[j+2] in dict5K:
              if word[j+2] in resultTrigram[(word[j], word[j+1])]:
                resultTrigram[(word[j], word[j+1])][word[j+2]] += 1
              else:
                resultTrigram[(word[j], word[j+1])][word[j+2]] = 1
            else:
              if 'OOV' in resultTrigram[(word[j], word[j+1])]:
                resultTrigram[(word[j], word[j+1])]['OOV'] += 1
              else:
                resultTrigram[(word[j], word[j+1])]['OOV'] = 1
          else:
            resultTrigram[(word[j], word[j+1])] = {}
            if word[j+2] in dict5K:
              resultTrigram[(word[j], word[j+1])][word[j+2]] = 1
            else:
              resultTrigram[(word[j], word[j+1])]['OOV'] = 1
        else:
          if (word[j], 'OOV') in resultTrigram:
            if word[j+2] in dict5K:
              if word[j+2] in resultTrigram[(word[j], 'OOV')]:
                resultTrigram[(word[j], 'OOV')][word[j+2]] += 1
              else:
                resultTrigram[(word[j], 'OOV')][word[j+2]] = 1
            else:
              if 'OOV' in resultTrigram[(word[j], 'OOV')]:
                resultTrigram[(word[j], 'OOV')]['OOV'] += 1
              else:
                resultTrigram[(word[j], 'OOV')]['OOV'] = 1
          else:
            resultTrigram[(word[j], 'OOV')] = {}
            if word[j+2] in dict5K:
              resultTrigram[(word[j], 'OOV')][word[j+2]] = 1
            else:
              resultTrigram[(word[j], 'OOV')]['OOV'] = 1
      else:
        if word[j+1] in bigramDict['OOV']:
          if ('OOV', word[j+1]) in resultTrigram:
            if word[j+2] in dict5K:
              if word[j+2] in resultTrigram[('OOV', word[j+1])]:
                resultTrigram[('OOV', word[j+1])][word[j+2]] += 1
              else:
                resultTrigram[('OOV', word[j+1])][word[j+2]] = 1
            else:
              if 'OOV' in resultTrigram[('OOV', word[j+1])]:
                resultTrigram[('OOV', word[j+1])]['OOV'] += 1
              else:
                resultTrigram[('OOV', word[j+1])]['OOV'] = 1
          else:
            resultTrigram[('OOV', word[j+1])] = {}
            if word[j+2] in dict5K:
              resultTrigram[('OOV', word[j+1])][word[j+2]] = 1
            else:
              resultTrigram[('OOV', word[j+1])]['OOV'] = 1
        else:
          if ('OOV', 'OOV') in resultTrigram:
            if word[j+2] in dict5K:
              if word[j+2] in resultTrigram[('OOV', 'OOV')]:
                resultTrigram[('OOV', 'OOV')][word[j+2]] += 1
              else:
                resultTrigram[('OOV', 'OOV')][word[j+2]] = 1 
            else:
              if 'OOV' in resultTrigram[('OOV', 'OOV')]:
                resultTrigram[('OOV', 'OOV')]['OOV'] += 1
              else:
                resultTrigram[('OOV', 'OOV')]['OOV'] = 1
          else:
            resultTrigram[('OOV', 'OOV')] = {}
            if word[j+2] in dict5K:
              resultTrigram[('OOV', 'OOV')][word[j+2]] = 1
            else:
              resultTrigram[('OOV', 'OOV')]['OOV'] = 1

  return resultTrigram


## 2.3: Create a method that calculates language model probabilities

def getBigramProbs(uniDict, biDict):
  tempVocabDict = {}
  for i in biDict:
    for j in biDict[i]:
      if j not in tempVocabDict:
        tempVocabDict[j] = 1
  for i in biDict:
    for j in biDict[i]:
      checkCount = uniDict['OOV']
      if i in uniDict:
        checkCount = uniDict[i]
      tempProb = (biDict[i][j] + 1) / (checkCount + len(tempVocabDict))
      biDict[i][j] = tempProb
  return biDict

def getTrigramProbs(biDictProbs, biDictCounts, triDict):
  tempVocabDict = {}
  for i in triDict:
    for j in triDict[i]:
      if j not in tempVocabDict:
        tempVocabDict[j] = 1
  for i in triDict:
    for j in triDict[i]:
      firstWord = i[1]
      secondWord = j
      tempProb = biDictProbs[firstWord][secondWord]
      tempProb1 = (triDict[i][j] + 1) / (biDictCounts[firstWord][secondWord] + len(tempVocabDict))
      actualProb = (tempProb + tempProb1)/2
      # if i == ('to', 'process'):
      #   print((tempProb + (1/len(tempVocabDict)))/2)
      triDict[i][j] = actualProb
  return triDict






###################################################################################
## MAIN

if __name__ == "__main__":
  indicesOfHead = []
  val = readData("onesec_train.tsv")
  top5kDict = getTop5k(val)
  unigramCounts = getUnigrams(top5kDict, val)
  bigramCounts = getBigrams(top5kDict, val)
  trigramCounts = getTrigrams(top5kDict, val, bigramCounts)
  print(unigramCounts['language'])
  print(unigramCounts['the'])
  print(unigramCounts['formal'])
  print(bigramCounts['the']['language'])
  print(bigramCounts['OOV']['language'])
  print(bigramCounts['to']['process'])
  if ('specific', 'formal') in trigramCounts:
    if 'languages' in trigramCounts[('specific', 'formal')]:
      print(trigramCounts[('specific', 'formal')]['languages'])
    else:
      print(0)
  else:
    print(0)
  if ('to', 'process') in trigramCounts:
    if 'OOV' in trigramCounts[('to', 'process')]:
      print(trigramCounts[('to', 'process')]['OOV'])
    else:
      print(0)
  else:
    print(0)
  if ('specific', 'formal') in trigramCounts:
    if 'event' in trigramCounts[('specific', 'formal')]:
      print(trigramCounts[('specific', 'formal')]['event'])
    else:
      print(0)
  else:
    print(0)


  probsOfBigram = getBigramProbs(getUnigrams(top5kDict, val), getBigrams(top5kDict, val))
  print(probsOfBigram['the']['language'])
  print(probsOfBigram['OOV']['language'])
  print(probsOfBigram['to']['process'])
  probsOfTrigram = getTrigramProbs(getBigramProbs(getUnigrams(top5kDict, val), getBigrams(top5kDict, val)), getBigrams(top5kDict, val), getTrigrams(top5kDict, val, bigramCounts))
  if ('specific', 'formal') in probsOfTrigram:
    if 'languages' in probsOfTrigram[('specific', 'formal')]:
      print(probsOfTrigram[('specific', 'formal')]['languages'])
    else:
      print("NOT VALID Wi")
  else:
    print("NOT VALID Wi")
  if ('to', 'process') in probsOfTrigram:
    # print(probsOfTrigram[('to', 'process')])
    if 'OOV' in probsOfTrigram[('to', 'process')]:
      print(probsOfTrigram[('to', 'process')]['OOV'])
    else:
      print("NOT VALID Wi")
  else:
    print("NOT VALID Wi")
  if ('specific', 'formal') in probsOfTrigram:
    if 'event' in probsOfTrigram[('specific', 'formal')]:
      print(probsOfTrigram[('specific', 'formal')]['event'])
    else:
      print("NOT VALID Wi")
  else:
    print("NOT VALID Wi")




