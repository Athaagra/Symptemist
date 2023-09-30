#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:39:30 2023

@author: Optimus
"""

import glob
import codecs 
from bs4 import BeautifulSoup as bs
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import os
import numpy as np
from itertools import chain
import sklearn_crfsuite
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
#import nltk
import time
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
#nltk.download('punkt')
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut
cess_sents = cess.tagged_sents()
# Train the unigram tagger
uni_tag = ut(cess_sents)

def modelCrf(X_train,y_train):
    import pycrfsuite
    trainer = pycrfsuite.Trainer(verbose=True)

    # Submit training data to the trainer
    for xseq, yseq in zip(X_train, y_train):
      trainer.append(xseq, yseq)

    # Set the parameters of the model
    trainer.set_params({
       # coefficient for L1 penalty
       'c1': 0.1,
  
       # coefficient for L2 penalty
       'c2': 0.1,  

       # maximum number of iterations
       'max_iterations': 1,

       # whether to include transitions that
       # are possible, but not observed
       #'feature.possible_transitions': True
    })
    # Provide a file name as a parameter to the train function, such that
    # the model will be saved to the file when training is finished
    trainer.train('crf.model')

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def documents(soup):
    sent = []
    sent1 = []
    wtt=[]
    for elem in soup.find_all("body"):
        t1=0
        for c in elem:
            line = c.splitlines()
            sent.append(list(line))
            print(sent[0])
        for i in range(len(sent[0])):
              line = re.sub(r'[^\w\s]','',sent[0][i])
              line = word_tokenize(line)
              lin=list(chain(*list(chain(*line))))
              for e in lin:
                  t2= t1+len(e)
                  wtt.append((e,t1,t2))
                  t1 += len(e)
              sent1.append(line)
    return sent1,wtt

def anottationtokenization(soup):
    documents = []
    sent1 = []
    for elem in soup.find_all("body"):
        for e in elem:
            print('This is the elem {}'.format(e))
            line = e.splitlines()
            documents.append(list(line))
            t1=0
            for i in range(len(documents[0])):
                  lineOne = re.sub(r'[^\w\s]','',documents[0][i])
                  lineOne = word_tokenize(lineOne)
                  #print(lineOne[0])
                  sent1.append(lineOne)
    return sent1

def token_annotation(annotations):
    termsl=[]
    for q in range(len(annotations)):
        terms = []
        for i in range(len(annotations[q])):
            if len(annotations[q][i]) >= 4:
                ent = np.arange(4,len(annotations[q][i]))
                for e in ent:
                    terms.append((annotations[q][i][0],annotations[q][i][1],annotations[q][i][2],annotations[q][i][3],annotations[q][i][e]))
        termsl.append(terms)
    return termsl

def read_xml(titles):
    tr = []
    for i in titles:
#/home/sakis/MEDOCAN/"xml_train/"+
        with open(i, "r") as infile:
            soup_train = bs(infile, "html5lib")
            tr.append(soup_train)
    return tr

def parse_data(data):
    dataa=[]
    for i in data:
        doc,wti = documents(i)
        dataa.append(doc)
    return dataa,wti

def parse_dataAnn(data):
    dataa=[]
    for i in data:
        doc = anottationtokenization(i)
        dataa.append(doc)
    return dataa

def tagging1(dataset):     
    docs = []
    for elem in dataset:
        texts = []
        # Loop through each child of the element under "textwithnamedentities"
        for i,c in enumerate (elem):
            if type(c) == list:
                #if str(c.attrs) == "{'type': 'HOSPITAL'}":
                #    label = "I-HOSP"  # part of a named entity
                #elif str(c.attrs) == "{'type': 'DOCTOR'}":
                #    label = "I-DOC"  # part of a named entity
                #elif str(c.attrs) == "{'type': 'ID'}":
                #    label = "ID"  # part of a named entity
                #elif str(c.attrs) == "{'type': 'DATE'}":
                #    label = "DT"  # part of a named entity
                if str(c) == "{'type': 'PATIENT'}":
                    label = "I-PAT"  # part of a named entity
                #elif str(c.attrs) == "{'type': 'LOCATION'}":
                #    label = "LOC"  # part of a named entity
                #else:
                #    label = "PHO"  # part of a named entity
            else:
                label = "O"  # irrelevant word
            for w in c:
                #if not w.text == None: 
                   # w = w.text.split(',')
                    if len(w) > 0 and w is not None:
                                texts.append((w, "O"))
            #    print(i)
        docs.append(texts)
    return docs

def clear_tags_t(titles_t):
    import xml.etree.ElementTree as ET
    from pprint import pprint
    from nltk import word_tokenize
    import nltk
    docum = []
    c= 0 
    for i in range(len(titles_t)):
        x = open(str(titles_t[i]))
        new = []
        token = nltk.word_tokenize(str(x.readlines()))
        for i in token:
          if type(i) == str:
              label = 'O'
              new.append([label,i])
          #c= c + 1
        docum.append(new)
        #print(nm,Id,Loc,date,age,other,cont)
    return docum

def clear_tags(titles):
    import xml.etree.ElementTree as ET
    from pprint import pprint
    docum = []
    nm = 0
    Id = 0
    Loc = 0
    date = 0
    age = 0
    other = 0
    cont = 0
    for i in titles:
#/home/sakis/MEDOCAN/"xml_train/"+
        x = open(str(i))
        print('This is the x {}'.format(x))
        tree = ET.parse(x)
        new = []
        for e in tree.findall('.//'):
            #for x in e.attrib.values():
            for key, value in e.attrib.items():
                if key == 'text':
                    #print(key,value)
                    if e.tag == 'NAME':
                        label = 'I-NAME'
                        nm= nm + 1
                    if e.tag == 'ID':
                        label = 'I-ID'
                        Id= Id + 1
                    if e.tag == 'LOCATION':
                        label = 'I-LOC'
                        Loc = Loc + 1
                    if e.tag == 'DATE':
                        label = 'I-DATE'
                        date = date + 1
                    if e.tag == 'AGE':
                        label = 'I-AGE'
                        age = age + 1
                    if e.tag == 'OTHER':
                        label = 'I-OTHER'
                        other = other + 1
                    if e.tag == 'CONTACT':
                        label = 'I-CONT'
                        cont = cont + 1
                    new.append([label,tokenize(re.sub(r'[^\w\s]','',value))])
        docum.append(new)
        print(nm,Id,Loc,date,age,other,cont)
    return docum

def datasets(label1, dataset): 
    newww=[]
    for i in range(len(label1)):
        neww =[]
        for x in dataset[i]:
            #print(x[0])
            for b in label1[i]:
                c = 0 
              #  print(b[1][0])
                if x[0] == b[1][0]:
                    x = (x[0],b[0])
                   # print(x)
                elif len(b[1])> 0:
                    while len(b[1])>c:
                        if x[0] == b[1][c]:
                            x=(x[0],b[0])
                            c = c +1
                        else:
                            c = c +1
                else:
                    x = (x)
            neww.append(x)
        newww.append(neww)
    return newww

def preproann(annotations_set): #,annotationSetC):
    tr_ann = read_xml(annotations_set)
    tr_an = parse_dataAnn(tr_ann)
    #trAnn = read_xml(annotationSetC)
    #trAnnC = parse_dataAnnC(trAnn)
    #print(tr_an)
    #print(trAnnC)
    #tr_AnnoC= mergeAnnot(tr_an,trAnnC)
    ann_tr = token_annotation(tr_an)
    return ann_tr

def prepro(xml):
    titles = []
    for file in xml:
        titles.append(file)
    data,termL = parse_data(read_xml(titles))
    dataPrepr=tagging1(data)
    #dataPrepr = pos(bios_tagging(datasets(clear_tags(titles),tagging1(data))))
    return dataPrepr,termL

def IndexAnnotationsAset(sett,tkannot):
    indexbt=[]
    indexbtCO=[]
    for i in range(len(sett)):
        c=0
        x=0
        while len(tkannot[i]) > c:
            #print(i,c)
            #print(sett[i][x][0].lower(),tkannot[i][c][4].lower())
            if sett[i][x][0].lower()==tkannot[i][c][4].lower():
                    indexbt.append((i,x))
                    indexbtCO.append((i, x, tkannot[i][c][0]))
            if x==len(sett[i])-1:
                    c= c+1
                    x=0
            else:
                    x= x+1

    return indexbt,indexbtCO

def tagging(IndexTags,Set):
    print(Set)
    for i in IndexTags:
        entity = list(Set[i[0]][i[1]])
        if entity[1] == "O":
            entity[1] = entity[1].replace("O", "SINTOMA")
            Set[i[0]][i[1]] = tuple(entity)
        else:
            Set[i[0]][i[1]] = Set[i[0]][i[1]]
    return Set

def bios_tagging(datast):
    docs =[]
    for i,doc in enumerate(datast):
        words=[]
        for j in range(len(datast[i])):
            previous_word = datast[i][j-2][1]
            #print(previous_word)
            next_word = datast[i][j][1]
            #print(next_word)
            token = datast[i][j][0]
            if previous_word == "O" and next_word == "I-SINTOMA":
                    #next_word = next_word.replace("I-MORFOLOGIA_NEOPLASIA",
                next_word="B-SINTOMA"
                #print(next_word)
            #elif previous_word == "I-SINTOMA" and next_word == "O":
                #next_word = next_word.replace("I-MORFOLOGIA_NEOPLASIA",
                #next_word = "E-MORFOLOGIA_NEOPLASIA"
                #print(next_word)
            else:
                next_word = datast[i][j][1]
            words.append((token, next_word))
        docs.append(words)
    return docs

def pos(sets):
    import nltk
    data = []
    for i, doc in enumerate(sets):
        # Obtain the list of tokens in the document
        tokens = [t for t, label in doc]
        # Perform POS tagging
        tagged = uni_tag.tag(tokens)#es.tag(tokens)
        # Take the word, POS tag, and its label
        data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    return data
def lastData(Ds,term):
    TrainSet=[]
    for i in range(len(Ds)):
        Set=[]
        for x in range(len(Ds[i])):
            Set.append((Ds[i][x][0],Ds[i][x][1],term[i][x][1],term[i][x][2],Ds[i][x][2]))
        TrainSet.append(Set)
    return TrainSet

# =============================================================================
# Train set 
# =============================================================================
train_txt_files = glob.glob('brat/*.txt')
train_ann_files = glob.glob('brat/*.ann')
train_set,termLenTrain = prepro(train_txt_files)
trann = preproann(train_ann_files)
indexOfEntities,inde = IndexAnnotationsAset(train_set,trann)
Dataset = tagging(indexOfEntities,train_set)
BiosData=bios_tagging(Dataset)
TrainSet=pos(BiosData)
# =============================================================================
# Test set
# =============================================================================
test_txt_files = glob.glob('brat-test/*.txt')
test_set,termLenTest = prepro(test_txt_files)
BioDat=bios_tagging(test_set)
TestSet=pos(BioDat)
#TrainSet=lastData(DatSet,termLenTrain)


def Sets(Daset):
    SeentWord = []
    SeentTags = []
    Worrds = []
    Taggs = []
    for a in range(len(Daset)):
        wordy=[]
        tagy=[]
        for b in range(len(Daset[a])):
            print(a,b)
            print(Daset[a][b][0])
            print(Daset[a][b][4])
            Worrds.append(Daset[a][b][0])
            Taggs.append(Daset[a][b][4])
            wordy.append(Daset[a][b][0])
            tagy.append(Daset[a][b][4])
        SeentWord.append(wordy)
        SeentTags.append(tagy)
    return SeentWord, SeentTags, Worrds, Taggs
def word2features0(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word[-1:]=' + word[-1:],
        'word[+1:]=' + word[+1:],
        'word[+2:]=' + word[+2:],
        'word[+1:]=' + word[+3:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + str(postag)
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        word2 = doc[i-2][0]
        postag1 = doc[i-1][1]
        postag2 = doc[i-2][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + str(postag1),
            '-2:word.lower=' + word2.lower(),
            '-2:word.istitle=%s' % word2.istitle(),
            '-2:word.isupper=%s' % word2.isupper(),
            '-2:word.isdigit=%s' % word2.isdigit(),
            '-2:postag=' + str(postag2)
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-2:
        word1 = doc[i+1][0]
        word2 = doc[i+2][0]
        postag1 = doc[i+1][1]
        postag2 = doc[i+2][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + str(postag1),
            '+2:word.lower=' + word2.lower(),
            '+2:word.istitle=%s' % word2.istitle(),
            '+2:word.isupper=%s' % word2.isupper(),
            '+2:word.isdigit=%s' % word2.isdigit(),
            '+2:postag=' + str(postag2)
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features
# A function for extracting features in documents
def extract_features(doc):
    return [word2features0(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]


X_train = [extract_features(doc) for doc in TrainSet]
y_train = [get_labels(doc) for doc in TrainSet]
#X_valid = [extract_features(doc) for doc in DatSetDv]
#y_valid = [get_labels(doc) for doc in DatSetDv]
X_test = [extract_features(doc) for doc in TestSet]
start_time_crf = time.time()
#X = [rpl.sent2features(s) for s in new]
#y = [rpl.sent2labels(s) for s in new]
# Conditional random fields
# define fixed parameters and parameters to search
rs = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),}
rs.fit(X_train, y_train)
labels = list(rs.classes_)
# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)
# search
#rs = RandomizedSearchCV(crf, params_space,
 #                       cv=3,
 #                       verbose=1,
 #                       n_jobs=-1,
 #                       n_iter=50,
 #                       scoring=f1_scorer)
#rs.fit(X_train, y_train)
# predictions on the X_test
y_pred = rs.predict(X_test)

np.save('CRFtestLabels.npy', y_pred)    # .npy extension is added if not given
np.save('CRFtestText.npy', test_set)
#y_pred_test = rs.predict(X_test)
metrics.flat_f1_score(y_train, y_pred,
                      average='weighted', labels=labels)
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
elapsed_time_crf = time.time() - start_time_crf
import time
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn_crfsuite.metrics import flat_classification_report
print("/n/n/n/n/n/n **** CRF **** /n/n/n/n",file=open("CRF.txt", "a"))
#print(flat_classification_report(y_train, y_pred,digits=3),file=open("CRF.txt", "a"))
print("F1-score: {:.1%}".format(f1_score(y_train, y_pred)),file=open("CRF.txt", "a"))
print(classification_report(y_train, y_pred),file=open("CRF.txt", "a"))
#print(trainer,file=open("CRF.txt", "a"))
#print(y_pred_test,file=open("CRFx.txt", "a"))
#print(X_test,file=open("CRFx.txt", "a"))


import gym
import tensorflow as tf
import tensorflow.keras as keras 
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import os
#import catch as Catch

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

class BytesIntEncoder:

    @staticmethod
    def encode(b: bytes) -> int:
        return int.from_bytes(b, byteorder='little')

    @staticmethod
    def decode(i: int) -> bytes:
        return i.to_bytes(((i.bit_length() + 5) // 8), byteorder='little')
    
class environment:
    def __init__(self,dset):
        self.envidata=dset
        self.stepenv=0
        self.reward=0
        self.done=False
    def reset(self,eps):
        stateenv=BytesIntEncoder.encode(self.envidata[eps][0][0].encode())#self.envidata['ddate'][self.stepenv],int(np.array(self.envidata['adsh'][self.stepenv][0:10]))
        stateenv=str(stateenv)
        stateenv=np.array(int(stateenv[0:5]))
        stateenv=np.reshape(stateenv,(1,1))
        self.stepenv=0
        return stateenv
    def step(self,action,eps):
        steps=self.stepenv
        #print(steps)
        done=self.done
        steps+=1
        self.stepenv=steps
        #print(self.envidata[eps][steps][0])
        new_stateenv=BytesIntEncoder.encode(self.envidata[eps][steps][0].encode())
        #state=new_state
        new_stateenv=str(new_stateenv)
        new_stateenv=np.array(int(new_stateenv[0:5]))
        new_stateenv=np.reshape(new_stateenv,(1,1))
        if action==1:
            reward=[0 if self.envidata[eps][steps][2] == 'O' else 1]
        #if action==2:
        #    reward=[1 if self.envidata['value'][steps] < 0 else 0]
        else:
            reward=[0 if self.envidata[eps][steps][2] == 'SINTOMA' else 1]
        done=[True if len(self.envidata[ep])-2==steps else False]
        #print(done)
        return new_stateenv,reward[0],done[0],action
ep=0  
game=0
env = environment(TrainSet)
while game!=1:
    done=False
    state=env.reset(ep)
    env_video = []
    ep+=1
    if ep==len(TrainSet):
        ep=0
        game+=1
    while not done:
        action = np.random.randint(2)#env.action_space.sample()
        state,reward,done, _=env.step(action,ep)
    
class ActorNetwork(keras.Model):
    def __init__(self,n_actions, fc1_dims=512,fc2_dims=256,name='actor',chkpt_dir=''):
        super().__init__()
        self.fc1_dims = fc1_dims
        #self.fc2_dims = fc2_dims 
        self.n_actions = n_actions 
        self.model_name = name 
        self.checkpoint_dir = chkpt_dir 
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac20000.h5')
        self.fc1=keras.layers.Dense(self.fc1_dims, activation='tanh')
        #self.fc2=keras.layers.Dense(self.fc2_dims, activation='tanh')
        self.pi = keras.layers.Dense(n_actions, activation='softmax')
        
    def call(self,state):
        value=self.fc1(state)
        #value=self.fc2(value)
        pi = self.pi(value)
        return pi

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=256, name='critic', chkpt_dir=''):
            super().__init__()
            self.fc1_dims =fc1_dims 
            #self.fc2_dims = fc2_dims 
            self.n_actions = n_actions 
            self.model_name = name 
            self.checkpoint_dir = chkpt_dir
            self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac20000.h5')
            self.fc1 = keras.layers.Dense(self.fc1_dims,activation='tanh')
            #self.fc2 = keras.layers.Dense(self.fc2_dims,activation='tanh')
            self.v = keras.layers.Dense(1, activation=None)
    def call(self,state):
        value = self.fc1(state)
        #value = self.fc2(value)
        #loss='sparse_categorical_crossentropy',
        v=self.v(value)
        return v
class Agent:
        def __init__(self, alpha=0.000003, gamma=0.99, n_actions=2):
            self.gamma=gamma
            self.n_actions = n_actions
            self.action = None
            self.action_space=[i for i in range(self.n_actions)]
            self.actor = ActorNetwork(n_actions=n_actions)
            self.critic = CriticNetwork(n_actions=n_actions)
            self.actor.compile(optimizer=keras.optimizers.SGD(learning_rate=alpha))
            self.critic.compile(optimizer=keras.optimizers.SGD(learning_rate=alpha))
        def choose_action(self, observation):
            state = tf.convert_to_tensor([observation])
            probs = self.actor(state)
            #print('This is the action {}'.format(probs))
            action_probabilities=np.argmax(probs)#tfp.distributions.Categorical(probs=probs)
            action = action_probabilities#action_probabilities.sample()
            #print('This is the action {}'.format(action))
#            log_prob=action_probabilities.log_prob(action)
            self.action=action
            return action#action.numpy()[0]
        def save_models(self):
            print('...saving models ...')
            self.actor.save_weights(self.actor.checkpoint_file)
            self.critic.save_weights(self.critic.checkpoint_file)
            
        def load_models(self):
            print('...loading models...')
            self.actor.load_weights(self.actor.checkpoint_file)
            self.critic.load_weights(self.critic.checkpoint_file)
        
        def learn(self, state, reward, state_, done):
            state = tf.convert_to_tensor([state],dtype=tf.float32)
            state_ =tf.convert_to_tensor([state_],dtype=tf.float32)
            reward = tf.convert_to_tensor(reward,dtype=tf.float32)
            with tf.GradientTape(persistent=True) as tape:
                probs = self.actor(state)
                state_value = self.critic(state)
                state_value_ = self.critic(state_)
                state_value = tf.squeeze(state_value)
                state_value_ = tf.squeeze(state_value_)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(self.action)
                delta = reward + self.gamma+state_value_*(1-int(done)) - state_value
                actor_loss =-log_prob*delta
                critic_loss = delta**2
            actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

def plotLearning(scores, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t]=np.mean(scores[max(0,t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig('cumulativeppo.png')
    plt.show()
agent=Agent(alpha=0.01,n_actions=2)
n_games = 0
best_score = 0#env.reward_range[0]
score_history = []
load_checkpoint = False

if load_checkpoint:
    agent.load_models()
ep=0
training_process=[]
observation=env.reset(ep)
while n_games!=5:
    observation=env.reset(ep)
    done=False
    score = 0
    print('this is the game {}'.format(n_games))
    #observation = observation
    stepss=0
    ep+=1
    if ep==len(TrainSet):
        ep=0
        n_games+=1
    while done!=True:
        action = agent.choose_action(observation)
        observation_,reward, done, info = env.step(action,ep)
        stepss+=1
        observation_ = observation_
        score += reward
        if not load_checkpoint:
            agent.learn(observation, reward, observation_,done)
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-10:])
    print('This is the score {} fullscore {} and avg_score {} and episode {}'.format(score,stepss,avg_score,ep))
    training_process.append([score,stepss])
    if avg_score > best_score:
        best_score = avg_score 
        if not load_checkpoint:
            agent.save_models()
        if n_games % 50 >= 0:
            print('episode',ep,'score %.1f' % score, 'avg_score %.1f' % avg_score)
    if n_games % 50 == 0:
        print('episode', ep, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
if not load_checkpoint:
    x=[i+1 for i in range(n_games)]
    plotLearning(score_history, window=100)

agent.load_models()
done=False
Reward=[]
avg_re=0
rew_ep=[]
ep=0
gamess=0
pltr=[]
state=env.reset(ep)
while gamess!=1:
    state=env.reset(ep)
    state = list(np.concatenate(state).flat)
    done=False
    steps=0
    Reward.append([gamess,np.mean(rew_ep)])
    rew_ep=[]
    ep+=1
    if ep==len(TestSet):
        ep=0
        gamess+=1
    while done!=True:
        action=agent.choose_action(state)
        if action==1:
            print('This is the action {}'.format(action))
            break
        state,reward,done,_=env.step(action,ep)
        steps+=1
        rew_ep.append([reward,steps])
        avg_re=np.mean(rew_ep)
        #if done==True:
        #    print('winning')
            #steps=0
        state = list(np.concatenate(state).flat)
    pltr.append(rew_ep)


np.save('A2CtestLabels.npy', pltr)    # .npy extension is added if not given

Reward=np.array(Reward)
plt.figure(figsize=(13, 13))
plt.ylabel('Score')
plt.xlabel('Game')
#plt.plot(Reward[:,0],color='red')
plt.plot(Reward[:,1],color='blue')
plt.savefig('rewardEpActorCritic7.png')
plt.grid(True)
plt.show()


trn_proc=np.array(training_process)
trn_proci=np.array(trn_proc[:,0])
trn_proc_dif=abs(trn_proc[:,0]-trn_proc[:,1])
trn_proci = np.reshape(trn_proci,(3750,1))
trn_proc_dif = np.reshape(trn_proc_dif,(3750,1))
trn_procl=np.hstack((trn_proci,trn_proc_dif))
trn_proclchunk=trn_procl[3000:3750]
# importing pygal
import pygal
import numpy
from pygal.style import Style
  
# change graph color
custom_style = Style(
    colors=('#daa520','#ff00ff'))
  
# creating Bar chart object
#pie_chart = pygal.Bar(style=custom_style) 
  
# creating line chart object
line_chart = pygal.StackedLine(style=custom_style)
  
# naming the title
line_chart.title = 'Stacked Line chart'
  
  
# adding lines
line_chart.add('A', trn_proclchunk[:,0])
line_chart.add('B', trn_proclchunk[:,1])
#line_chart.add('C', numpy.random.rand(5))
#line_chart.add('D', numpy.random.rand(5))
  
line_chart
line_chart.render_to_png('aa.png')

trch=[]
for i in trn_proc:
    print(i)
    prec=i[0]/i[1]
    tot=abs(prec-1)
    trch.append([prec,tot])
trch=np.array(trch)
trpr=trch[3000:3750]
# importing packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# load dataset
y_labels=np.arange(701,750)
df = pd.DataFrame(trpr, index=y_labels)
# plot a Stacked Bar Chart using matplotlib
df.plot(
   # x = 'Name',
    kind = 'barh',
    stacked = True,
    title = 'Correct Predictions',
    mark_right = True,
    grid=True,
    color=['y','c'],
    figsize=(18, 18))
plt.xlabel('Predictions Actor Critic')
plt.ylabel('document')
#plt.yticks(labels=y_labels)
plt.legend(["tp+tn", "fp+fn"]);


import os 
import time 
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from gym import wrappers 
from torch.autograd import Variable 
from collections import deque
#import catch as Catch

save_models=True
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and  not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage=[]
        self.max_size=max_size
        self.ptr = 0
        
    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)]=transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
    def sample(self,batch_size):
        ind=np.random.randint(0, len(self.storage),size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state,next_state,action,reward,done=self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states),np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        #self.layer_1=nn.Linear(state_dim,400)
        #self.layer_2=nn.Linear(400,300)
        #self.layer_3=nn.Linear(300, action_dim)
        #self.Softmax=nn.Softmax(dim=1)
        self.layer_1=nn.Linear(state_dim,128)
        ##self.layer_2=nn.Linear(400,300)
        self.layer_3=nn.Linear(128, action_dim)
        self.Softmax=nn.Softmax(dim=1)

        self.max_action=max_action
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        ##x = F.relu(self.layer_2(x))
        #x = self.max_action * torch.tanh(self.layer_3(x))
        x = torch.tanh(self.layer_3(x))
        x = self.Softmax(x)
        #print(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + 1, 128)
        #self.layer_2 = nn.Linear(400,300)
        self.layer_3 = nn.Linear(128,1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + 1, 128)
        #self.layer_5 = nn.Linear(400,300)
        self.layer_6 = nn.Linear(128,1)
    
    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        #print(xu)
        x1 = F.relu(self.layer_1(xu))
        #x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        
        x2 = F.relu(self.layer_4(xu))
        #x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2
    def Ql(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        #x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

# Selecting the device(CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
    def __init__(self,state_dim,action_dim,max_action):
        self.actor=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters())#params=self.parameters(),lr=7e-3))
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters())#params=self.parameters(),lr=7e-3))
        self.max_action=max_action
    
    def select_action(self,state):
        state=torch.Tensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self,replay_buffer,iterations,batch_size=100,discount=0.99, tau=0.1, policy_noise=0.2,noise_clip=0.5, policy_freq=1):
        
        for b in range(iterations):
            #sample a batch of transition (s,s',a,r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state= torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(np.array(batch_actions)).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done=torch.Tensor(batch_dones).to(device)
            next_action=self.actor_target(next_state)
            noise = torch.Tensor(action).data.normal_(0,policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action=next_action.detach().numpy()#
            noise=noise.detach().numpy()
            noise=np.reshape(noise,(-1,1))
            next_action = next_action + noise#).clamp(-self.max_action, self.max_action)
            next_action = np.argmax(next_action,axis=1)
            next_action=np.reshape(next_action,(-1,1))
            next_action=torch.Tensor(next_action).to(device)
            #print('This is the next action {} this is the next state {}'.format(next_action,next_state))
            #print('This is the next state {} , the next_action {}'.format(len(next_state),len(next_action)))
            target_Q1,target_Q2=self.critic_target(next_state,next_action)
            target_Q=torch.min(target_Q1,target_Q2)
            target_Q=reward + ((1-done) * discount * target_Q).detach()
            #action=np.argmax(action,axis=1)
            action=np.reshape(action,(-1,1))
            #print('This is state {} action {}'.format(state, action))
            #print('This is the state {} , the action {}'.format(len(state),len(action)))
            current_Q1,current_Q2 = self.critic(state,action)
            critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            if b % policy_freq == 0:
                #print('This is the input to actor {}'.format(self.actor(state)))
                selfact=self.actor(state).detach().numpy()#
                selfact = np.argmax(selfact,axis=1)
                selfact=np.reshape(selfact,(-1,1))
                selfact=torch.Tensor(selfact).to(device)
                actor_loss = - self.critic.Ql(state, selfact).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory,filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory,filename))
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

def evaluate_policy(policy,eval_episodes=len(TrainSet)):
    avg_reward=0
    #eval_episodes=500
    #reward_ep=[]
    step=0
    for _ in range(eval_episodes):
        obs=env.reset(_)
        done=False
        while done!= True:
            action = policy.select_action(np.array(list(np.concatenate(obs).flat)))#obs))
            #action=int(np.round(abs(action)))
            #print('This is evaluate action {}'.format(action))
            obs,reward,done, _ =env.step(np.argmax(action),_)
            #print(obs,reward,done)
            done=done
            #print(done)
            step+=1
            avg_reward += reward
    avg_reward /= step
        #reward_ep.append(avg_reward)
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    return avg_reward
#Parameters Initialization
env_name="Risk Management"
seed = 0
start_timesteps=2000
eval_freq=5e3
max_timesteps=1000000
expl_noise=0.1
batch_size=100
discount=0.99
tau=0.15
policy_noise=0.5
noise_clip=0.5
policy_freq=2

file_name="%s_%s_%s" % ('TD3', env_name, str(seed))
print("Settings: %s" % (file_name))



env = environment(TrainSet)
#state=env.reset()
#done=False
#cumulativeReward=0
#while done!=True:
#    action=np.random.randint(0,3,1)
#    new_state,reward,done=env.step(action)
#    print(new_state,action,reward,done)
#    cumulativeReward+=reward[0]
#    done=done[0]
#    print('This is the cumulative Reward {}'.format(cumulativeReward))
#print('The risk management agent achieves {} percentage'.format(cumulativeReward/len(FinancialData)))
s = env.reset(0)

s = list(s)
step_pause = 0.3 # the pause between each plot
#env.render(step_pause) 
# Test
#n_test_steps = 100
continuous_execution = False
print_details = False
#env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
#                max_misses=max_misses, observation_type=observation_type, seed=seed)
#env.seed(seed)
torch.manual_seed(seed)
state_dim =len(s) #env.observation_space.shape[0]
action_dim=2#env.action_space.shape[0]
max_action=float(1)#len(env.action_space.high[0])
policy = TD3(state_dim,action_dim,max_action)
replay_buffer=ReplayBuffer()

evaluations=[evaluate_policy(policy)]

total_timesteps=0
timesteps_since_eval=0
episode_num=0
done=True
t0=time.time()

while total_timesteps < max_timesteps:
    if done:
        if total_timesteps !=0:
            #print("Total Timesteps:{} Episode Num {} Reward {}".format(total_timesteps,episode_num,episode_reward))
            policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
        
        if timesteps_since_eval >=eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy))
            policy.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            
        obs=env.reset(episode_num)
        done=False
        episode_reward=0
        episode_timesteps=0
        episode_num+=1
    if total_timesteps < start_timesteps:
        action = np.random.randint(2)#env.action_space.sample()
    else:
        action = policy.select_action(np.array(obs))
        #action=np.array(int(np.round(abs(action[0]))))
        #print('This is the action: {}'.format(action))
        if expl_noise !=0:
            action = (action + np.random.normal(0, expl_noise, size=1)).clip(0, 1)#env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
            #action=np.array(int(np.round(abs(action[0]))))
    #print(action)
    new_obs,reward,done,_=env.step(np.argmax(action),episode_num)
    print(obs,new_obs,reward,done,episode_num,episode_timesteps)
    done_bool=1 if episode_timesteps + 2 == len(TrainSet[episode_num]) else float(done)#env._max_episode_steps else float(done)
    episode_reward += reward
    replay_buffer.add((list(np.concatenate(obs).flat),list(np.concatenate(new_obs).flat),np.argmax(action), reward, done_bool))
    obs=new_obs
    episode_timesteps +=1
    total_timesteps+=1
    timesteps_since_eval+=1
    if episode_num==len(TrainSet) or len(TrainSet[episode_num])-2==episode_timesteps:
        episode_num+=0
        done=True
evaluations.append(evaluate_policy(policy))
if save_models:policy.save('%s' % (file_name),directory="./pytorch_models")
np.save("./results/%s" % (file_name),evaluations)
#policy = TD3(state_dim,action_dim,max_action)
#policy.load(file_name,'./pytorch_models/')
#np.load("./results/%s" % (file_name),evaluations)
eval_episodes=250
rewardAg=[]
env=environment(TestSet)
for _ in range(eval_episodes):
    avg_reward=0
    obs=env.reset(_)
    done=False
    step=0
    ep_reward=[]
    while done!= True:
        action = policy.select_action(np.array(list(np.concatenate(obs).flat)))
        print(np.argmax(action))
        obs,reward,done, _ =env.step(np.argmax(action),_)
        done=done
        avg_reward += reward
        step+=1
        ep_reward.append(reward)
    #arew=np.array(avg_reward)/20000
    rewardAg.append(sum(ep_reward)/step)
cx=np.arange(0,len(rewardAg))
plt.figure(figsize=(13, 13))
plt.ylabel('Score')
plt.xlabel('Game')
plt.plot(x,rewardAg)
plt.savefig('rewardEpActorCritic7.png')
plt.grid(True)
plt.show()
