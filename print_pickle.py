import pandas as pd
import sys

infile=sys.argv[1]


import pickle

pickle_off = open(infile,"rb")
data2 = pickle.load(pickle_off)
print(data2)
