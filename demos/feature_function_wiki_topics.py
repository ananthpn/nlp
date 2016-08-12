'''
Created on 23-Jul-2016
@author: Anantharaman Narayana Iyer

Conventions for writing feature functions:
1. first letter of function name should be f, followed by the label, underscore, function number
e.g. fsports_1(x, y)

'''
from nltk import sent_tokenize, word_tokenize
from classifiers.feature_functions_base import FeatureFunctionsBase

#----------------- Our Mini Gazeteer look up -----------------------------
countries_list = [
            "india", "australia", "germany", "pakistan", "england", "britain",
            "america", "usa", "bangladesh", "lanka", "srilanka", "south africa",
            "zimbabwe", "west indies"
    ]

nationality_list = [
            "indian", "australian", "german", "pakistani", "english", "british",
            "american",  "bangladeshi", "lankan", "srilankan", "south african",
             "west indian"
    ]

sports_roles = [
            "batsman", "bowler", "fielder", "wicket keeper", "captain",
            "vice captain", "coach", "physio",
    ]

sports_terms = [
            "bowling", "batting", "keeping", "fielding", "runs", "played",
            "tournament", "match", "game", "team", "sport", "sports", "winning",
            "losing", "play", "playing", "run", "ball", "bat", "length", "line",
            "wide", "lbw", "running", "test", "international", "t20", "innings",
    ]

sports_types = ["cricket", "tennis", "soccer", "football"]

langs_list = [
            "hindi", "english", "tamil", "kannada", "malayalam", "telugu", 
            "marathi", "punjabi", "bengali"
    ]

film_roles = [
        "hero", "heroine", "villain", "comedian", "director", "producer",
        "singer", "writer", "actor", "actress", 
    ]

film_terms = [
        "play", "screen", "acting", "role", "film", "films", "film", "movies", "cinema",
        "theater", "song", "songs", "music", "direction", "hollywood", "bollywood",
        "koliwood", "sandalwood", "fan", "club"
    ]

film_types = [
        "thriller", "action", "suspense", "comedy", "romantic", "sci-fi", "scifi",
    ]
# -----------------------------------------------------------------------------

class WikiFeatureFunctions(FeatureFunctionsBase):
    def __init__(self):
        super(WikiFeatureFunctions, self).__init__()
        self.fdict = {}
        for k, v in WikiFeatureFunctions.__dict__.items():
            if hasattr(v, "__call__"):
                if k[0] == 'f':
                    tag = k[1:].split("_")[0]
                    val = self.fdict.get(tag, [])
                    val.append(v)
                    self.fdict[tag] = val
        self.supported_tags = self.fdict.keys()        
        return
    
    def check_membership(self, ref_set, my_set):
        """Check if there is any non null intersection between 2 sets"""
        rset = set(ref_set)
        mset = set(my_set)
        if len(rset.intersection(mset)) > 0:
            return 1
        else:
            return 0
    
    # you may write as many functions as you require
    # you should return 0 or 1 for each feature function
    # words is a word tokenized text document and y is a label
    
    # -------------------------- SPORTS --------------------------------------
    def fsports_1(self, words, y):
        """Check for nationality"""
        if (self.check_membership(nationality_list, words)) and (y == "sports"):
            return 1
        return 0

    def fsports_2(self, words, y):
        if (self.check_membership(countries_list, words)) and (y == "sports"):
            return 1
        return 0

    def fsports_3(self, words, y):
        if (self.check_membership(sports_roles, words)) and (y == "sports"):
            return 1
        return 0

    def fsports_4(self, words, y):
        if (self.check_membership(sports_terms, words)) and (y == "sports"):
            return 1
        return 0

    def fsports_5(self, words, y):
        if (self.check_membership(sports_types, words)) and (y == "sports"):
            return 1
        return 0

    # -------------------------- filmS --------------------------------------    
    def ffilm_1(self, words, y):
        if (self.check_membership(langs_list, words)) and (y == "film"):
            return 1
        return 0

    def ffilm_2(self, words, y):
        if (self.check_membership(film_roles, words)) and (y == "film"):
            return 1
        return 0

    def ffilm_3(self, words, y):
        if (self.check_membership(film_terms, words)) and (y == "film"):
            return 1
        return 0

    def ffilm_4(self, words, y):
        if (self.check_membership(film_types, words)) and (y == "film"):
            return 1
        return 0
    
#     def fpolitics_1(self, words, y):
#         return
# 
#     def fpolitics_2(self, words, y):
#         return
    
    def evaluate(self, x, y):
        words = []
        stoks = sent_tokenize(x.lower())
        # create a linear array of word tokens
        for tok in stoks:
            words.extend(word_tokenize(tok))
        return FeatureFunctionsBase.evaluate(self, words, y)



if __name__ == "__main__":
    ff = WikiFeatureFunctions()
    print ff.get_supported_labels()
