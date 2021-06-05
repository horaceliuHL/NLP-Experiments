# -*- coding: utf-8 -*-
"""a4_Liu_112833815.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f4I11WXY2CdqCmcKE6JQA6hNPfPWtrsp
"""

import gensim.downloader as api
word_embs = api.load('glove-wiki-gigaword-50')

! wget -nc https://www3.cs.stonybrook.edu/~has/CSE354/music_QA_train.json
! wget -nc  https://www3.cs.stonybrook.edu/~has/CSE354/music_QA_dev.json

! pip install datasets transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
tokenizer0 = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer1 = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer2 = AutoTokenizer.from_pretrained("bert-base-uncased")
model0 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model1 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model2 = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

import sys
import numpy as np
import pandas as pd
import json
from gensim.utils import tokenize
import torch
import torch.nn as nn  #pytorch
import torch.nn.functional as F
from datasets import Dataset

sys.stdout = open('a4_Liu_112833815_OUTPUT.txt', 'w')
###############################################################################

# 1.1: Load the data
def loadData(filename):
  data = []
  with open(filename, 'r') as infile:
    data = json.load(infile)
  return data

# 1.2: Prepare to create word embeddings as input
def tokenizeData(dictData):
  for i in range(len(dictData)):
    entry = dictData[i]
    entry['question_toks'] = list(tokenize(entry['question'], lowercase=True))
    entry['passage_toks'] = list(tokenize(entry['passage'], lowercase=True))

def get_embed(word):
  if word in word_embs:
    return word_embs.wv[word]
  else:
    return word_embs.wv['unk']
  

# 1.3: Define and train the GRU-RNN in PyTorch
class DOC_RNN(nn.Module):
  def __init__(self, embedding_dim, gru_hidden_dim, number_of_labels):
    super(DOC_RNN, self).__init__()
    self.gru = nn.GRU(embedding_dim, gru_hidden_dim)
    self.linearClassifier = nn.Linear(gru_hidden_dim, number_of_labels)
  def forward(self, X):
    doc_vecs = []
    for doc in X:
      s, _ = self.gru(doc.unsqueeze(1))
      doc_vecs.append(s[-1])
    doc_vecs = torch.stack(doc_vecs).squeeze(1)
    doc_vecs = self.linearClassifier(doc_vecs)
    yprobs = F.softmax(doc_vecs)
    return yprobs


# 2.1: Create baseline transformers: question-only and passage-only

def removeKeyFromDict(dictData, key):
  for i in range(len(dictData)):
    del dictData[i][key]

# Taken from Hugging Face [SQuAD|Sequence Classification] Tutorial
def preprocess_function_passage(examples):
  return tokenizer0(examples['passage'], truncation=True)

# Taken from Hugging Face [SQuAD|Sequence Classification] Tutorial
def preprocess_function_question(examples):
  return tokenizer1(examples['question'], truncation=True)

# Taken from Hugging Face [SQuAD|Sequence Classification] Tutorial
def preprocess_function_full(examples):
  if 'passage' not in examples:
    return tokenizer2(examples['question'], truncation=True)
  return tokenizer2(examples['question'], examples['passage'], truncation=True)

def convertLabelBoolToInt(dictData):
  for i in range(len(dictData)):
    if dictData[i]['label'] == True:
      dictData[i]['label'] = 1
    else:
      dictData[i]['label'] = 0


# 2.3

# def concatEntire(dictData):
#   for i in range(len(dictData)):
#     entry = dictData[i]
#     entry['entire'] = '[CLS] ' + entry['question'] + ' [SEP] ' + entry['passage'] + ' [SEP]'


###############################################################################
if __name__ == "__main__":
  print("Key points about my music QA system: \n")
  print("1. Runs using BERT Transformer with question and passage \n")
  print("2. Attempted to use GPT as well as bert-large-uncased but ran into GPU issues \n")
  print("3. Uses max rounded value (integer) as predictions where <= 0 is 0 and >= 1 is 1 \n")
  print("4. Uses 5 epochs with 0.01 weight decay and 0.2 learning rate \n")
  # Part 1: Music QA with RNNs
  trainData = loadData('music_QA_train.json')
  tokenizeData(trainData)
  converted = []
  for i in range(len(trainData)):
    eachEntry = trainData[i]
    input = [get_embed(word) for word in list(eachEntry['passage_toks']) + list(eachEntry['question_toks'])]
    input = torch.FloatTensor(input)
    converted.append(input)
  
  resultToTestAgainst = []
  for i in range(len(trainData)):
    temp = 0
    temp1 = 1
    if trainData[i]['label'] == True:
      temp = 1
      temp1 = 0
    resultToTestAgainst.append([temp, temp1])

  resultToTestAgainst = torch.FloatTensor(resultToTestAgainst)

  learning_rate, epochs = 0.1, 10
  modelProcess = DOC_RNN(len(converted[0][0]), 50, 2)
  sgdProcess = torch.optim.SGD(modelProcess.parameters(), lr=learning_rate, weight_decay=0.001)
  loss_func_Process = torch.nn.BCELoss() #includes log

  #training loop:
  for i in range(epochs):
    modelProcess.train()
    sgdProcess.zero_grad()
    #forward pass:
    ypred = modelProcess(converted)
    loss = loss_func_Process(ypred, resultToTestAgainst)
    #backward: /(applies gradient descent)
    loss.backward()
    sgdProcess.step()

  ########################
  # Testing data

  testData = loadData('music_QA_dev.json')
  tokenizeData(testData)
  convertedTest = []
  for i in range(len(testData)):
    eachEntry = testData[i]
    input = [get_embed(word) for word in list(eachEntry['passage_toks']) + list(eachEntry['question_toks'])]
    input = torch.FloatTensor(input)
    convertedTest.append(input)
  
  actualResultToTestAgainst = []
  for i in range(len(testData)):
    temp = 0
    if testData[i]['label'] == True:
      temp = 1
    actualResultToTestAgainst.append(temp)

  with torch.no_grad():
    ytestpred_prob = modelProcess(convertedTest)
    ytestpred_prob = ytestpred_prob.cpu().detach().numpy()
    numCorrect = 0
    for i in range(len(ytestpred_prob)):
      high = round(np.max(ytestpred_prob[i]))
      high = 0 if high <= 0 else 1
      if high == actualResultToTestAgainst[i]:
        numCorrect += 1
    print("Accuracy for GRU is: " + str(numCorrect/len(actualResultToTestAgainst)))

  ##############################################################################

  # Part 2: Music QA with Transformers (Passage-only)
  trainDataPassage = loadData('music_QA_train.json')
  removeKeyFromDict(trainDataPassage, 'question')
  convertLabelBoolToInt(trainDataPassage)
  trainDataPassage = Dataset.from_dict({k: [d[k] for d in trainDataPassage] for k in trainDataPassage[0]})
  encoded_Passage = trainDataPassage.map(preprocess_function_passage, batched=True)

  valDataPassage = loadData('music_QA_dev.json')
  removeKeyFromDict(valDataPassage, 'question')
  convertLabelBoolToInt(valDataPassage)
  valDataPassage1 = Dataset.from_dict({k: [d[k] for d in valDataPassage] for k in valDataPassage[0]})
  encoded_Passage1 = valDataPassage1.map(preprocess_function_passage, batched=True)

  argsPassage = TrainingArguments(output_dir='temp',
                           evaluation_strategy = "epoch",
                           learning_rate=0.2,
                           num_train_epochs=3,
                            weight_decay=0.00001, load_best_model_at_end=True)

  trainerPassage = Trainer(
      model0, argsPassage, train_dataset=encoded_Passage, eval_dataset=encoded_Passage1,
      tokenizer=tokenizer0
  )
  trainerPassage.train()
  resultPassage = trainerPassage.predict(encoded_Passage1)
  predictionsPassage = resultPassage.predictions.tolist()
  # print(predictionsPassage)
  counter = 0
  for i in range(len(predictionsPassage)):
    tempPred = max(predictionsPassage[i][0], predictionsPassage[i][1])
    tempPred = 0 if round(tempPred) <= 0 else 1
    if valDataPassage1[i]['label'] == tempPred:
      counter += 1
  print("Accuracy for passage-only transformer is: " + str(counter/len(valDataPassage1)))

  # Part 2: Music QA with Transformers (Question-only)
  trainDataQuestion = loadData('music_QA_train.json')
  removeKeyFromDict(trainDataQuestion, 'passage')
  convertLabelBoolToInt(trainDataQuestion)
  trainDataQuestion = Dataset.from_dict({k: [d[k] for d in trainDataQuestion] for k in trainDataQuestion[0]})
  encoded_Question = trainDataQuestion.map(preprocess_function_question, batched=True)

  valDataQuestion = loadData('music_QA_dev.json')
  removeKeyFromDict(valDataQuestion, 'passage')
  convertLabelBoolToInt(valDataQuestion)
  valDataQuestion1 = Dataset.from_dict({k: [d[k] for d in valDataQuestion] for k in valDataQuestion[0]})
  encoded_Question1 = valDataQuestion1.map(preprocess_function_question, batched=True)

  argsQuestion = TrainingArguments(output_dir='temp',
                           evaluation_strategy = "epoch",
                           learning_rate=0.2,
                           num_train_epochs=3,
                            weight_decay=0.01, load_best_model_at_end=True)

  trainerQuestion = Trainer(
      model1, argsQuestion, train_dataset=encoded_Question, eval_dataset=encoded_Question1,
      tokenizer=tokenizer1
  )
  trainerQuestion.train()
  resultQuestion = trainerQuestion.predict(encoded_Question1)
  predictionsQuestion = resultQuestion.predictions.tolist()
  # print(predictionsPassage)
  counter = 0
  for i in range(len(predictionsQuestion)):
    tempPred = max(predictionsQuestion[i][0], predictionsQuestion[i][1])
    tempPred = 0 if round(tempPred) <= 0 else 1
    if valDataQuestion1[i]['label'] == tempPred:
      counter += 1
  print("Accuracy for question-only transformer is: " + str(counter/len(valDataQuestion1)))

  # Part 2: Music QA with Transformers (Passage and Question)
  # Final model after attempts to improve (tried mean/min/max and GPT)
  trainData = loadData('music_QA_train.json')
  # concatEntire(trainData)
  # removeKeyFromDict(trainData, 'passage')
  # removeKeyFromDict(trainData, 'question')
  convertLabelBoolToInt(trainData)
  trainData1 = Dataset.from_dict({k: [d[k] for d in trainData] for k in trainData[0]})
  encoded_dataset = trainData1.map(preprocess_function_full, batched=True)

  valData = loadData('music_QA_dev.json')
  # concatEntire(valData)
  # removeKeyFromDict(valData, 'passage')
  # removeKeyFromDict(valData, 'question')
  convertLabelBoolToInt(valData)
  valData1 = Dataset.from_dict({k: [d[k] for d in valData] for k in valData[0]})
  encoded_dataset1 = valData1.map(preprocess_function_full, batched=True)

  args = TrainingArguments(output_dir='temp',
                           evaluation_strategy = "epoch",
                           learning_rate=0.2,
                           num_train_epochs=5,
                            weight_decay=0.01, load_best_model_at_end=True)

  trainer = Trainer(
      model2, args, train_dataset=encoded_dataset, eval_dataset=encoded_dataset1,
      tokenizer=tokenizer2
  )

  trainer.train()
  result = trainer.predict(encoded_dataset1)
  predictions = result.predictions.tolist()
  # print(predictionsPassage)
  counter = 0
  for i in range(len(predictions)):
    tempPred = max(predictions[i][0], predictions[i][1])
    # tempPred = (predictions[i][0] + predictions[i][1])/2 # tried mean but worse
    tempPred = 0 if round(tempPred) <= 0 else 1
    if valData1[i]['label'] == tempPred:
      counter += 1
  print("Accuracy for question-passage transformer is: " + str(counter/len(valData)))

##################################################################

  # Part 2: Asking transformer model 3 yes/no questions
  print("\nThree Questions\n")
  threeQuestions = [{
      'question': "Was the Titanic produced in 1997?",
      'idx': 1
  }, {
      'question': "Is Jojo Rabbit based on a real person?",
      'idx': 2
  }, {
      'question': "Did Christopher Nolan's Inception get good reviews?",
      'idx': 3
  }]
  print(threeQuestions[0]['question'])
  print(threeQuestions[1]['question'])
  print(threeQuestions[2]['question'])
  threeQuestions1 = Dataset.from_dict({k: [d[k] for d in threeQuestions] for k in threeQuestions[0]})
  encoded_questions = threeQuestions1.map(preprocess_function_full, batched=True)

  questionAns = trainer.predict(encoded_questions)
  questionAns = questionAns.predictions
  for i in range(len(questionAns)):
    tempPred = max(questionAns[i][0], questionAns[i][1])
    tempPred = 'No' if round(tempPred) <= 0 else 'Yes'
    print(tempPred)

  # Part 2: Actually testing for Kaggle competition
  valDataTest = loadData('music_QA_test.json')
  idxList = []
  for i in range(len(valDataTest)):
    idxList.append(valDataTest[i]['idx'])
  # concatEntire(valDataTest)
  # removeKeyFromDict(valDataTest, 'passage')
  # removeKeyFromDict(valDataTest, 'question')
  # removeKeyFromDict(valDataTest, 'idx')
  valDataTest = Dataset.from_dict({k: [d[k] for d in valDataTest] for k in valDataTest[0]})
  encoded_dataset_test = valDataTest.map(preprocess_function_full, batched=True)

  test_predictions = trainer.predict(encoded_dataset_test)
  test_predictions = test_predictions.predictions
  actualGuesses = []
  for i in range(len(test_predictions)):
    tempPred = max(test_predictions[i][0], test_predictions[i][1])
    tempPred = 0 if round(tempPred) <= 0 else 1
    actualGuesses.append(tempPred)
  
  tempdf = pd.DataFrame({'idx': idxList, 'label': actualGuesses})

  tempdf.to_csv("354_Kaggle_Horace_Liu.csv", index=False)