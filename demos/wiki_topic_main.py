'''
Created on 23-Jul-2016

@author: Anantharaman
'''
from feature_function_wiki_topics import WikiFeatureFunctions
from classifiers.maxent_base import LogLinear
import os

ds_path = os.path.join("..", "data", "datasets", "wiki_pages") 

def get_wiki_file_names():
    fnames_1 = os.listdir(ds_path)
    fnames = [os.path.join(ds_path, f) for f in fnames_1]
    return fnames

def prepare_dataset(supported_labels):
    dataset = []
    fnames_1 = os.listdir(ds_path)
    fnames = [os.path.join(ds_path, f) for f in fnames_1]
    for fn, fn1 in zip(fnames, fnames_1):
        label = fn1.split("_")[1] # our convention is that the filename will be name_label
        if label in supported_labels:
            txt = open(fn).read()
            dataset.append([txt, label])
    return dataset

if __name__ == '__main__':
    ff = WikiFeatureFunctions()
    supported_labels = ff.get_supported_labels()
    print "Supported Labels: ", supported_labels
    dataset = prepare_dataset(supported_labels)
    clf = LogLinear(ff)
    clf.train(dataset, max_iter=50)
    while True:
        txt = raw_input("Enter a text for classification: ")
        if txt == "__Q__":
            break
        result = clf.classify(txt)
        print result