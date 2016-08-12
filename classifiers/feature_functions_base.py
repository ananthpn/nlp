'''
Created on 23-Jul-2016
@author: Anantharaman
'''

class FeatureFunctionsBase(object):
    '''
    Base class for any feature functions that can be used with LogLinear
    '''
    def __init__(self):
        self.fdict = {}
        for k, v in FeatureFunctionsBase.__dict__.items():
            if hasattr(v, "__call__"):
                if k[0] == 'f':
                    tag = k[1:].split("_")[0]
                    val = self.fdict.get(tag, [])
                    val.append(v)
                    self.fdict[tag] = val
        self.supported_tags = self.fdict.keys()        
        return
    
    def get_supported_labels(self):
        return self.supported_tags 
    
    def evaluate(self, x, y):
        feats = []
        for t, f in self.fdict.items():
            if t == y:
                for f1 in f:
                    feats.append(int(f1(self, x, y)))
            else:
                for f1 in f:
                    feats.append(0)
        return feats

if __name__ == "__main__":
    ff = FeatureFunctionsBase()
    print ff.supported_tags
