import pandas as pd
import sys
import argparse

import os

parser = argparse.ArgumentParser(description="Convert csv dataframe to pickle.")
parser.add_argument('f', help='input file name', metavar='infile')
parser.add_argument('-o', help='output file name', required=False, metavar='outfile')
args = parser.parse_args()
print(args)


infile=args.f
df = pd.read_csv(infile, header=None)

# assumes the target (label) is in column 0
# assumes the data (features) are in column 1:

data = {'target':df.iloc[:,0].values, 'data':df.iloc[:,1:].values }

print('shape target: ', data['target'].shape)
print('shape data: ', data['data'].shape)

import pickle
if(args.o):
    outfile = args.o
else:
    outfile = os.path.splitext(infile)[0] + '.pkl'

print('infile:', infile, ' outfile:',  outfile)
pickling_on = open(outfile, "wb")
pickle.dump(data, pickling_on)
pickling_on.close()


pickle_off = open(outfile,"rb")
data2 = pickle.load(pickle_off)
print(data2)
