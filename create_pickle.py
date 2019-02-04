import pandas as pd
import sys

infile=sys.argv[1]
outfile=sys.argv[2]

df = pd.read_csv(infile, header=None)

# assumes the labels are in column 1 and the data are in columns 2-
data = {'target':df.iloc[:,0].values, 'data':df.iloc[:,1:].values }

print('shape target: ', data['target'].shape)
print('shape data: ', data['data'].shape)

import pickle
pickling_on = open(outfile, "wb")
pickle.dump(data, pickling_on)
pickling_on.close()


# pickle_off = open(outfile,"rb")
# data2 = pickle.load(pickle_off)
# print(data2)
