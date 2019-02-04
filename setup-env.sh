# set up a new python envirnment
conda create --name py27_tf_19_active-learning python=2.7
conda activate py27_active-learning
conda install -y numpy
conda install -y scipy
conda install -y pandas
conda install -y scikit-learn
conda install -y matplotlib
conda install -y tensorflow-gpu=1.9
conda install -y keras
conda install -y google-apputils

# install the active-learning repository
git clone https://github.com/google/active-learning

# download the iris data
cd active-learning/utils
python  create_data.py --datasets iris

# run an experiment
cd ..
python run_experiment.py --dataset iris --sampling_method uniform
 
# look at the results
more /tmp/toy_experiments/iris_uniform/log-2018-12-06-23-30-16.txt


# Now lets set up to do active learning with the tc3 benchmark code
  188  wget http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/type-class/GDC.FPKM-UQ.csv
  189  wget http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/type-class/GDC_metadata.txt
  190  wget http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/type-class/type_18_300_test.csv
  191  wget http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/type-class/type_18_300_train.csv

# After we get the data, we have to pickle it because that's what the
# active learning run experiment code is looking to read.

python ./create_pickle.py  type_18_300_test.csv
python ./create_pickle.py  type_18_300_train.csv


