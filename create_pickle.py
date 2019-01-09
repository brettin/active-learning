import pandas as pd
import sys

filename=sys.argv[1]
df = pd.read_csv(filename, header=None)
data = {'target':df.iloc[:,0].values, 'data':df.iloc[:,1:].values }

print('shape target: ', data['target'].shape)
print('shape data: ', data['data'].shape)

import pickle
pickling_on = open("nt_train2.pkl", "wb")
pickle.dump(data, pickling_on)
pickling_on.close()


pickle_off = open("nt_train2.pkl","rb")
data2 = pickle.load(pickle_off)
print(data2)
