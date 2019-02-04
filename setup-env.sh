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


