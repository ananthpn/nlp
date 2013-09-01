Cosine - The purpose of this module is to make it easy to evaluate cosine similarity for a
set of text sentences. This is a helper that invokes NLTK library for cosine similarity but makes it simple by:

1. Accepting a list of text messages as a Python list and performing vectorization internally
2. Allowing some preprocessing such as stemming and lemmatization

To compute cosine similarity for a set of text messages, instantiate the class Cosine and invoke
the compute_similarity method with the list of text messages as input.

API:

values compute_similarity(messages, stem = True, lemm = True)

Inputs:

messages - this is a Python list where each element is a text message
stem - this indicates whether stemming needs to be done on the input and this defaults to True
lemm - this indicates whether lemmatization needs to be done on the input and this defaults to True

Outputs:

values - this is a n x n matrix of similarity measures, that is implemented as a list of Python list.
	 The element of each inner list is a 2 element Python tuple of the form (angle, cosine value)

Example of values for input containing 4 messages:

[
   [(-2.2204460492503131e-16, 1.0), (1.0, 0.5403023058681398), (0.71132486540518713, 0.7574976186657894), (1.0, 0.5403023058681398)],
   [(1.0, 0.5403023058681398), (0.0, 1.0), (0.29289321881345254, 0.9574125437190454), (0.0, 1.0)],
   [(0.71132486540518713, 0.7574976186657894), (0.29289321881345254, 0.9574125437190454), (2.2204460492503131e-16, 1.0), (0.29289321881345254, 0.9574125437190454)],
   [(1.0, 0.5403023058681398), (0.0, 1.0), (0.29289321881345254, 0.9574125437190454), (0.0, 1.0)]
]

Usage Example:


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
