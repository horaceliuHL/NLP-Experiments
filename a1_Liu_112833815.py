#!/usr/bin/python3
# Horace Liu 112833815
# CSE354, Spring 2021
##########################################################
## a1_Liu_112833815.py
## Sentiment Classifier - lexicon-based,
##                        logistic regression based

import sys
import gzip
import json #for reading json encoded files into opjects
import re #regular expressions
import numpy as np
import torch
import torch.nn as nn  #pytorch

# Comment this line out if you wish to see results on the console
sys.stdout = open('a1_Liu_112833815_OUTPUT.txt', 'w') # EDIT THIS

#########################################################
## Part 1. Read and tokenize data.

def loadData(filename, bin_thresh=3.5):
    #DONT EDIT
    #reads the data and returns text, score per
    #binarizes the scores based on being above or below bin_thresh
    f = gzip.open(filename, 'r')
    data = [] #tuple of reviewText, score
    for jsonline in f:
        record = json.loads(jsonline.decode('utf-8'))
        try:
            data.append([record['reviewText'].lower(),
                         1 if record['overall'] > bin_thresh else 0])
        except KeyError:#skip record, don't have a key
            pass
    return data

wordRE = re.compile(r"[\.,!?;\"]|[A-z0-9\-]+'[A-z0-9\-]{0,3}|[A-z0-9\-]+", re.UNICODE) # COMPLETE THIS
def tokenize(text):
    text = text.strip().replace('\n', ' ')
    # text = re.sub(r'\'([A-z0-9]{4,})', r' \1', text)
    text = wordRE.findall(text)
    
    return text
    #1.1 IMPLEMENT
    #takes in a single string
    #Returns a list tokens from the string according to
    #the restrictions provided in the instructions.
    #Remember to lowercase.



#########################################################
## Part 2. Logistic Regression Classification

#default lexicon
#DO NOT DELETE ANY WORDS BUT YOU MAY ADD WORDS
#lexicon is the top 30 most common words from the LIWC2007 lexicon
lexica = {
    'pos' : ['like', 'love', 'good', 'happy', 'great', 'best', 'better', 'fun', 'hope', 'thank',
             'awesome', 'ready', 'thanks', 'yay', 'play', 'ok', 'pretty', 'loves', 'free', 'playing',
             'care', 'cool', 'true', 'glad', 'sweet', 'win', 'super', 'hopefully', 'okay', 'hah'],
    'neg' : ['miss', 'bad', 'hate', 'lost', 'hell', 'sad', 'ugh', 'fuck', 'sorry', 'pain',
             'mad', 'alone', 'seriously', 'sucks', 'missing', 'war', 'boring', 'hates', 'cry', 'worst',
             'suck', 'crap', 'missed', 'fear', 'cut', 'lazy', 'lose', 'broke', 'serious', 'crying'],
    'verbs': ['is', 'be', 'have', 'are', 'go', 'was', 'love', 'get', 'do', "i'm",
             'will', 'can', 'got', 'know', 'has', 'had', 'am', 'see', "it's", "don't"]
    }

def lexicaScore(lexica, tokens):
    posCount = 0
    negCount = 0
    verbCount = 0
    for word in lexica['pos']:
        for text in tokens:
            if word == text:
                posCount+=1
    
    for word in lexica['neg']:
        for text in tokens:
            if word == text:
                negCount+=1
    
    for word in lexica['verbs']:
        for text in tokens:
            if word == text:
                verbCount+=1

    # print(posCount, " ", negCount, " ", verbCount, " ", len(tokens))

    result = {
        'neg': negCount / len(tokens),
        'pos': posCount / len(tokens),
        'verbs': verbCount / len(tokens),
    }

    return result

    #2.1 IMPLEMENT
    #given:
    #  lexica -- a dictionary of "category name" -> [list of words]
    #  tokens -- a list of tokens (to match with lexicon
    #Finds the relative frequeny (count of tokens in category / total tokens
    #return:
    #  a dictionary of frequencies per category (e.g. {'pos': 0.10, 'neg': 0.07} )

def posNegLexClassify(lexicaScores, posName='pos', negName='neg'):
    #DON'T EDIT EXCEPT thresh
    #Given:
    #  lexicaScores - a dict of frequencies per category
    #  posName, negName - category name for positive and negative category
    #                     (don't change)

    thresh = 0.0001 #2.2 IMPLEMENT (adjust this to improve classification)
    score = lexicaScores[posName] - lexicaScores[negName]
    return 1 if score >= thresh else 0



#########################################################
## Part 3. Logistic Regression Classification

def extractMultiHot(tokens, vocab):
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    multi_hot = [0]*len(vocab)
    for word in tokens:
        if word in vocab_dict:
            multi_hot[vocab_dict[word]] = 1
    
    return multi_hot

    ##3.1 IMPLEMENT
    #Given:
    # tokens -- a list of strings appearing in a given
    # vocab -- the vocabulary tokens to record in the multi hot
    #          (this is also as a list of strings)
    #Output: A single multi-hot encoding of length len(vocab)


def normalizedLogLoss(ypred, ytrue):
    log_loss = 0
    for count in range(len(ypred)):
        log_loss += (ytrue[count]*torch.log(ypred[count]) + ((1 - ytrue[count])*torch.log(1 - ypred[count])))
    log_loss /= len(ypred)
    log_loss *= -1
    return log_loss

    ##3.2 IMPLEMENT
    ##Given:
    #  ypred - a vector (torch 1-d tensor) of predictions from the model.
    #          these are probabilities (values between 0 and 1)
    #  ytrue - a vector (torch 1-d tensor) of the true labels
    #Output:
    #  the logloss

## The Logistic Regression Class (do not edit but worth studying)
class LogReg(nn.Module):
    def __init__(self, num_feats, learn_rate = 0.01, device = torch.device("cpu") ):
        #DONT EDIT
        super(LogReg, self).__init__()
        self.linear = nn.Linear(num_feats+1, 1) #add 1 to features for intercept

    def forward(self, X):
        #DONT EDIT
        #This is where the model itself is defined.
        #For logistic regression the model takes in X and returns
        #a probability (a value between 0 and 1)

        newX = torch.cat((X, torch.ones(X.shape[0], 1)), 1) #add intercept
        return 1/(1 + torch.exp(-self.linear(newX))) #logistic function on the linear output


###################################################################################
## MAIN - NOTHING BELOW SHOULD BE EDITED


if __name__ == "__main__":
    ##DONT EDIT

    ##RUN PART 1: loading data, tokenize:
    print("\nLOADING DATA...")
    if len(sys.argv) != 2:
        print("USAGE: python3 a1_lastname_id.py Magazine_Subscriptions_5.json.gz")
        sys.exit(1)
    filename = sys.argv[1]
    data = loadData(filename)
    print("DONE.\n%d records.\nFirst 3 records:\n%s"
          % (len(data), "\n".join([str(d) for d in data[:3]])))

    #tokenize
    print("\nTOKENIZING TEXT...")
    for record in data:
        record[0] = tokenize(record[0])
    if not data[0][0]:
        print("**  1.1 Not complete")
        sys.exit(1)
    print("DONE.\nFirst 3 records:\n%s"
          % "\n".join([str(d) for d in data[:3]]))


    ##RUN PART 2: lexicon classifier
    #first run lexicon and then ask for a score.
    #the lexicon scores and the classification result
    #is appended to the record:
    print("\nSCORING BY LEXICON...")
    for record in data:
        lexicaScores = lexicaScore(lexica, record[0])
        try:
            lexicaClass =  posNegLexClassify(lexicaScores)
        except TypeError:
            print("** 2.1 not complete")
            sys.exit(1)
        record.extend([lexicaScores, lexicaClass])
    print("DONE.\nFirst 20 lexicon predictions:")
    lexPreds = []
    ratings = []
    for d in data[:20]:
        print("  rating: %d,   lex pred: %d,   lex scores: %s" % (d[1], d[3], d[2]))
    for d in data:
        lexPreds.append(d[3])
        ratings.append(d[1])
    print(  "Lexicon Overall Accuracy:      %.3f" % ((np.array(ratings) == np.array(lexPreds)).sum() / len(data)))


    ##RUN PART 3: logistic regression
    #get vocabulary:
    print("\nEXTRACTING FEATURES...")
    allWordCounts = dict()
    for word in [w for d in data for w in d[0]]:
        try:
            allWordCounts[word] +=1
        except KeyError:
            allWordCounts[word] = 1
    vocab = [k for k, v in allWordCounts.items() if v > 5] #all words occuring more than 5 times
    print("Vocabulary Size: %d" % len(vocab))

    #extract features:
    X = np.array([extractMultiHot(d[0], vocab) for d in data])
    y = np.array([d[1] for d in data])

    #create training and test
    splitPoint = int(0.80 * len(X))
    Xtrain, Xtest = torch.from_numpy(X[:splitPoint].astype(np.float32)), \
                    torch.from_numpy(X[splitPoint:].astype(np.float32))
    ytrain, ytest = torch.from_numpy(y[:splitPoint].astype(np.float32)).view(splitPoint,1), \
                    y[splitPoint:]
    print("Done.\n Xtrain shape: ", Xtrain.shape, ",  ytrain shape: ", ytrain.shape)

    #Model setup:
    learning_rate, epochs = 1.0, 300
    print("\nTraining Logistic Regression...")
    model = LogReg(len(vocab))
    sgd = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #training loop:
    for i in range(epochs):
        model.train()
        sgd.zero_grad()

        #forward pass:
        try:
            ypred = model(Xtrain)
        except IndexError:
            print("** 3.1 not complete")
            sys.exit(1)
        loss = normalizedLogLoss(ypred, ytrain)
        #backward:
        try:
            loss.backward()
        except AttributeError:
            print("** 3.2 not complete")
            sys.exit(1)
        sgd.step()

        if i % 20 == 0:
            print("  epoch: %d, loss: %.5f" %(i, loss.item()))
    print("Done.\nFirst 20 test set predictions:")

    #calculate accuracy on test set:
    with torch.no_grad():
        ytestpred_prob = model(Xtest)
        ytestpred_class = ytestpred_prob.round().type(torch.int8).numpy().T[0]
        ytestpred_prob = ytestpred_prob.numpy().T[0]
        lexPreds = []
        for i in range(20):
            print("  rating: %d,   logreg pred: %d (prob: %.3f),  lex pred: %d" % (ytest[i], ytestpred_class[i], ytestpred_prob[i], data[splitPoint+i][3]))
            lexPreds.append(data[splitPoint+i][3])
        print("\nLogReg Model Test Set Accuracy: %.3f" % ((ytest == ytestpred_class).sum() / ytest.shape[0]))
        print(  "Lexicon Test Set Accuracy:      %.3f" % ((ytest == np.array(lexPreds).T[0]).sum() / ytest.shape[0]))