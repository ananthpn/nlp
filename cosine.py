'''
Cosine similarity - This module makes it easy to evaluate cosine similarity for a
set of text sentences. This uses NLTK library but makes it simple to find the cosine
similarity by providing a function that accepts text strings as input and the function
performs the vectorization internally.

 -------------------------------- (C) ---------------------------------

                         Author: Anantharaman Palacode Narayana Iyer
                         <narayana.anantharaman@gmail.com>

  Distributed under the BSD license:

    Copyright 2010 (c) Anantharaman Palacode Narayana Iyer, <narayana.anantharaman@gmail.com>

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

        * Redistributions of source code must retain the above
          copyright notice, this list of conditions and the following
          disclaimer.

        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.
'''

from nltk import FreqDist
import nltk
import copy
import math
from nltk.corpus import wordnet
from nltk.corpus import stopwords


class Cosine():
    def __init__(self, stem = True, lemm = True):
        self.raw_inputs = []
        self.inputs = []
        self.vectors = []
        self.words = []
        self.fd = FreqDist()
        self.cos_values = []
        self.stemmer = nltk.porter.PorterStemmer()
        self.lemmatizer = nltk.wordnet.WordNetLemmatizer()        
        self.lemm = lemm
        self.stem = stem        
        return

    def set_input(self, txt):
        self.raw_inputs.append(txt)
        temp = []
        new_text = txt
        if self.stem:
            for word in nltk.word_tokenize(txt):
                if word.lower() in stopwords.words():
                    continue
                temp.append(self.stemmer.stem(word))
            new_text = ' '.join(temp)
        if self.lemm:
            for word in nltk.word_tokenize(new_text):
                if word.lower() in stopwords.words():
                    continue
                temp.append(self.lemmatizer.lemmatize(word))
            new_text = ' '.join(temp)
        self.inputs.append(new_text)
        return

    def setup_tftable(self):
        for txt in self.inputs:
            sents = nltk.sent_tokenize(txt)
            for sent in sents: # for each sentence in the given text
                words = nltk.word_tokenize(sent)
                for word in words:
                    self.fd.inc(word)

        self.tftable = [[k, 0] for k in self.fd.keys()]
        return  self.tftable

    def vectorize(self):
        tft = self.setup_tftable()
        vecs = []
        for txt in self.inputs:
            vecs.append(self.vectorize_one(txt))
        self.vectors = []
        for v in vecs:
            self.vectors.append(tuple(i[1] for i in v))
        return self.vectors

    def vectorize_one(self, txt):
        #we will take bag of words with word count
        myvector = copy.deepcopy(self.tftable)        
        sents = nltk.sent_tokenize(txt)
        for sent in sents: # for each sentence in the given text
            words = nltk.word_tokenize(sent)
            for word in words:
                for item in myvector:
                    if item[0] == word:
                        item[1] += 1
        return myvector

    #initialize a matrix that would contain cosine similarity value for each vector in the LVS against every other vector
    def init_cos_matrix(self, dim):
        values = []
        for m in range(dim):
            row = []
            for n in range(dim):
               row.append(None)
            values.append(row)
        return values
        
    def cosine(self, vecs = None): #returns the cosine similarity of the input vectors taken from self.vectors
        self.cos_values = self.init_cos_matrix(len(self.vectors))
        if vecs == None:
            vecs = self.vectors
        for u in range(len(vecs)):
            #self.cos_values.append([])
            for v in range(u, len(vecs)):
                angle = nltk.cluster.cosine_distance(vecs[u], vecs[v])
                value = math.cos(angle)
                self.cos_values[v][u] = self.cos_values[u][v] = (angle, value, )
        return self.cos_values

    def compute_similarity(self, messages, stem = True, lemm = True, threshold = 0.75):
        #given a list of messages computes the similarity and returns the matrix
        #messages is of the form: [message1, message2, ..., messagen]
        self.stem = stem
        self.lemm = lemm
        for m in messages:
            self.set_input(m)
        self.vectorize()
        values = self.cosine()
        return values

if __name__ == '__main__':
    cosine = Cosine()
    messages = []
    
    while(1):
        t = raw_input("Enter a text or q to quit: ")
        if (t == 'q') or (t == 'quit') or (t == 'qui'):
            break
        messages.append(t)
    values = cosine.compute_similarity(messages)
    for val in values:
        for v in val:
            print '%.3f\t' % v[1],
        print '\n'
