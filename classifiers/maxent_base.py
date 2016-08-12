'''
maxent_loglinear.py
MaxEnt Classifier
Author: Anantharaman Narayana Iyer
Date: 23 July 2016

Methods:

p_y_given_x(x, y) => returns prob of y for the given input x
train(X, Y, regularizer=0.01)  => creates a model by learning the params
classify(x) => for a given input x gets the prob distribution across labels and returns argmax tag

NOTE: We use an internal method create_dataset that invokes the feature functions and 
caches the feature vectors and expected labels in 2 data structures:
self.all_data = {}
self.dataset = [] # has the feature vector for expected label - we need this to compute the numerator of hyp function

How to use this classifier?

Prepare your dataset as X, Y where X is a matrix of n trg examples, Y is a matrix of n expected results
Write the feature function layer as a class
Instantiate LogLinear and give it your feature function class
    The feature function class should implement 2 methods: evaluate() and get_supported_labels()
    Evaluate should return the feature vector given x and y
    get_supported_labels should return the list of labels supported
Train the model by calling train()
    You can optionally set regularizer coefficient, default = 0.01
Use the model to classify: classify(x)

'''
import numpy
import math
from scipy.optimize import minimize as mymin 
#from scipy.optimize import fmin_l_bfgs_b as mymin 
import datetime
# ----------------------------------------------------------------------------------------
# maxent implementation
# ----------------------------------------------------------------------------------------
class LogLinear(object):
    def __init__(self, function_obj): 
        self.tag_set = None
        self.model = None # this may be set by load_classifier or by train methods
        self.func = function_obj
        self.iteration = 0
        self.cb_count = 0
        self.cost_value = None        
        return
    
    def create_dataset(self):
        self.dataset = []
        self.all_data = {}
        for h in self.h_tuples: # h represents each example x that we will convert to f(x, y)
            for tag in self.tag_set:
                feats = self.all_data.get(tag, [])
                val = self.get_feats(h[0], tag)
                feats.append(val)
                self.all_data[tag] = feats
                if (h[1] == tag):
                    self.dataset.append(val)
        for k, v in self.all_data.items():
            self.all_data[k] = numpy.array(v)
        self.dataset = numpy.array(self.dataset)
        return
    
    def get_feats(self, xi, tag): # xi is the history tuple and tag is y belonging to Y (the set of all labels
        # xi is of the form: history where history is a 4 tuple by itself
        # self.func is the function object
        return self.func.evaluate(xi, tag)
    
    def cb(self, params):
        print "cb count = ", self.cb_count
        self.cb_count += 1
        return
    
    #------------ TRAIN using NLL cost function - see theory -----------------
    def train(self, history_tuples, reg_lambda = 0.01, max_iter=5):
        # history_tuples, function_obj, reg_lambda = 0.01,
        self.iteration = 0
        self.h_tuples = history_tuples
        self.reg = reg_lambda
        self.dataset = None # this will be set by create_dataset
        self.tag_set = self.func.get_supported_labels() #supported_tags - this is the set of all tags
        self.create_dataset()
        self.dim = self.dataset.shape[1] #len(self.dataset[0])
        self.num_examples = self.dataset.shape[0]
        if (self.model == None) or (self.model.shape[0] != self.dim):
            self.model = numpy.array([0 for _ in range(self.dim)]) # initialize the model to all 0
        dt1 = datetime.datetime.now()
        print 'before training: ', dt1
        try:
            params = mymin(self.cost, self.model, method = 'L-BFGS-B', callback = self.cb, options = {'maxiter':max_iter}) #, jac = self.gradient) # , options = {'maxiter':100}
        except:
            print "Importing alternate minimizer fmin_l_bfgs_b"
            from scipy.optimize import fmin_l_bfgs_b as mymin 
            params = mymin(self.cost, self.model, fprime = self.gradient) # , options = {'maxiter':100}
            print "Min Point is: ", params[1] 
        #self.model = params.x
        dt2 = datetime.datetime.now()
        print 'after training: ', dt2, '  total time = ', (dt2 - dt1).total_seconds()
        return self.cost_value
    
    #-------------------------- HYPOTHESIS FUNCTION P(Y|X) ---------------------------
    def p_y_given_x(self, xi, tag): 
        """Given xi determine the probability of y - note: we have all the f(x, y) values for all y in the dataset"""
        normalizer = 0.0
        feat = self.get_feats(xi, tag) # execute the feature function layer and get the feature vector
        dot_vector = numpy.dot(numpy.array(feat), self.model) # this is the operation: v.f(x, y) in the theory
        
        # the loop below computes the denominator in our maxent hypothesis
        for t in self.tag_set:
            feat = self.get_feats(xi, t)
            dp = numpy.dot(numpy.array(feat), self.model)
            if dp == 0:
                normalizer += 1.0 # this is to avoid computing exp if dot product is 0
            else:
                normalizer += math.exp(dp)
        if dot_vector == 0:
            val = 1.0
        else:
            val = math.exp(dot_vector) #
        result = float(val) / normalizer # this is our maxent hypothesis function, see theory
        return result
    # ---------------------------------------------------------------------------------------
    
    def classify(self, xi):
        maxval = 0.0
        probs = {}
        result = None
        for t in self.tag_set:
            val = self.p_y_given_x(xi, t)
            probs[t] = val
            if val >= maxval:
                maxval = val
                result = t
        return result, probs
    
    #-------------- COST FUNCTION required for the train procedure ---------------
    def cost(self, params):
        self.model = params
        sum_sqr_params = sum([p * p for p in params]) # for regularization
        reg_term = 0.5 * self.reg * sum_sqr_params                
        dot_vector = numpy.dot(self.dataset, self.model) # this is: v.f(x, y) in our theory across all n samples in the dataset
        empirical = numpy.sum(dot_vector) # this is the emperical counts (first term in L(v))
        
        # let us get the normalizer part of the L(v) as below
        # this should be: dot prod computation for all values of y_prime with v and take exp of each of the dot prod
        # then sum this and take log of this
        # the log should be summed over all the training samples
         
        expected = 0.0        
        for j in range((self.num_examples)): # this is sigma over j = 1 to n (0 to n-1)
            mysum = 0.0
            for tag in self.tag_set: # get the jth example feature vector for each tag
                fx_yprime = self.all_data[tag][j] # this is f(xi, y_prime_i) 
                dot_prod = numpy.dot(fx_yprime, self.model)
                if dot_prod == 0:
                    mysum += 1.0
                else:
                    try:
                        mysum += math.exp(dot_prod)
                    except:
                        print "dot_prod = ", dot_prod, " tag = ", tag, " f = ", fx_yprime, " m = ", self.model 
            expected += math.log(mysum)
            
        if (self.iteration % 20) == 0:
            print "Iteration = ", self.iteration, "Cost = ", (expected - empirical + reg_term)
        self.iteration += 1
        self.cost_value = (expected - empirical + reg_term)
        return (expected - empirical + reg_term)

    #-------------- GRADIENT FUNCTION of cost  required for the train procedure ---------------
    def gradient(self, params):
        self.model = params        
        gradient = []
        for k in range(self.dim): # vk is a m dimensional vector
            reg_term = self.reg * params[k]
            empirical = 0.0
            expected = 0.0
            for dx in self.dataset:
                empirical += dx[k]
            for i in range(self.num_examples):
                mysum = 0.0 # exp value per example
                for t in self.tag_set: # for each tag compute the exp value
                    fx_yprime = self.all_data[t][i] #self.get_feats(self.h_tuples[i][0], t)

                    # --------------------------------------------------------
                    # computation of p_y_given_x
                    normalizer = 0.0
                    dot_vector = numpy.dot(numpy.array(fx_yprime), self.model)
                    for t1 in self.tag_set:
                        feat = self.all_data[t1][i]
                        dp = numpy.dot(numpy.array(feat), self.model)
                        if dp == 0:
                            normalizer += 1.0
                        else:
                            normalizer += math.exp(dp)
                    if dot_vector == 0:
                        val = 1.0
                    else:
                        val = math.exp(dot_vector) # 
                    prob = float(val) / normalizer
                    # --------------------------------------------------------
                    mysum += prob * float(fx_yprime[k])                    
                expected += mysum
            gradient.append(expected - empirical + reg_term)
        return numpy.array(gradient)

if __name__ == "__main__":
    pass
