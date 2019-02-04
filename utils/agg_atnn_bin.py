import itertools
import pandas as pd
import numpy as np
import os
import sys
import gzip
import argparse
import sklearn
import copy

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf

import keras as ke
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


'''
New models must follow the scikit learn api and implement the following methods:

fit(X, y[, sample_weight]): fit the model to the input features and target.
predict(X): predict the value of the input features.
score(X, y): returns target metric given test features and test targets.
decision_function(X) (optional): return class probabilities, distance to decision boundaries, or other metric that can be used by margin sampler as a measure of uncertainty.

'''

class AggAtnnBin(object):

  def __init__(self,
               random_state=1,
               epochs=100,
               batch_size=32,
               solver='rmsprop',
               learning_rate=0.001,
               lr_decay=0.
               ):

    # params
    self.solver = solver
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.lr_decay = lr_decay
    # data
    self.encode_map = None
    self.decode_map = None
    self.model = None
    self.random_state = random_state


  def build_model(self, PS=6212, DR=0.2):
    inputs = Input(shape=(PS,))
    x = Dense(1000, activation='relu')(inputs)
    a = Dense(1000, activation='relu')(x)
    b = Dense(1000, activation='softmax')(x)
    x = ke.layers.multiply([a,b])
    x = Dense(500, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(250, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(125, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(DR)(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(DR)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                 optimizer=SGD(lr=0.00001, momentum=0.9),
                 metrics=['acc', self.r2])

    # Save initial weights so that model can be retrained with same initialization
    self.initial_weights = copy.deepcopy(model.get_weights())
    self.model = model

  # According to the active-learning documentation:
  # fit(X, y[, sample_weight]): fit the model to the input features and target.
  def fit(self, X_train, y_train, sample_weight=None, class_weight=None, validation_data=None):
    if self.model is None:
      self.build_model()


    # We don't want incremental fit so reset learning rate and weights
    K.set_value(self.model.optimizer.lr, self.learning_rate)
    self.model.set_weights(self.initial_weights)

    # We want to compensate for unbalanced class labels
    if class_weight is None:
      print("shape of y_train is: ", y_train.shape)
      print("y_train: ", y_train)
      class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
      print ("class weights: ", class_weights)
      d_class_weights = dict(enumerate(class_weights))
      print ("class weights dict: ", d_class_weights)
      class_weight = d_class_weights

    # create y matrix
    y_mat = np.reshape(y_train, (len(y_train), 1))
    y_mat = ke.utils.to_categorical(y_mat, 2)

    _X_train, _X_test, _y_train, _y_test = sklearn.model_selection.train_test_split(
                                                         X_train, y_mat, test_size=.2)


    checkpointer = ModelCheckpoint(filepath='Agg_attn_bin.autosave.model.h5',
                           verbose=1,
                           save_weights_only=False,
                           save_best_only=True)

    csv_logger = CSVLogger('Agg_attn_bin.training.log')

    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                           factor=0.20,
                           patience=40,
                           verbose=1,
                           mode='auto',
                           min_delta=0.0001,
                           cooldown=3,
                           min_lr=0.000000001)

    early_stop = EarlyStopping(monitor='val_loss',
                             patience=200,
                             verbose=1,
                             mode='auto')

    print("calling fit on X_train and y_mat with shapes: ")
    print(X_train.shape, " ", X_train)
    print(y_mat.shape, " ", y_mat)

    self.model.fit(
      _X_train,
      _y_train,
      batch_size=self.batch_size,
      epochs=self.epochs,
      shuffle=True,
      class_weight=class_weight,
      verbose=1,
      validation_data=(_X_test, _y_test),
      # validation_split=0.2,
      callbacks = [checkpointer, csv_logger, reduce_lr, early_stop])

    # Put any final metrics you want to evaluate on the model here before
    # returning to the framework.
    _y_predict = self.model.predict(_X_test)

    threshold = 0.5
    _y_pred_int  = (_y_predict[:,0] < threshold).astype(np.int)
    _y_test_int = (_y_test[:,0] < threshold).astype(np.int)

    print(sklearn.metrics.roc_auc_score(_y_test_int, _y_pred_int))
    print(sklearn.metrics.balanced_accuracy_score(_y_test_int, _y_pred_int))
    print(sklearn.metrics.classification_report(_y_test_int, _y_pred_int))
    print(sklearn.metrics.confusion_matrix(_y_test_int, _y_pred_int))


  # According to the active-learning documentation, must implement:
  # predict(X): predict the value of the input features.
  def predict(self, X_val):
    predicted = self.model.predict(X_val)
    return predicted

  def score(self, X_val, val_y):
    # y_mat = self.create_y_mat(val_y)
    # create y matrix
    y_mat = np.reshape(val_y, (len(val_y), 1))
    y_mat = ke.utils.to_categorical(y_mat, 2)

    val_acc = self.model.evaluate(X_val, y_mat)[1]
    return val_acc

  # In Rick's code, this is done at data load time.
  # It's needed for the score function above.
  def create_y_mat(self, y):
    y_encode = self.encode_y(y)
    y_encode = np.reshape(y_encode, (len(y_encode), 1))
    y_mat = keras.utils.to_categorical(y_encode, self.n_classes)
    return y_mat


  @staticmethod
  def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


  def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




def load_data(data_path, PL, nb_classes):
  df = (pd.read_csv(data_path,skiprows=1).values).astype('float32')
  df_y = df[:,0].astype('int')
  df_x = df[:, 1:PL].astype(np.float32)

  # scaler = MaxAbsScaler()
  scaler = StandardScaler()
  df_x = scaler.fit_transform(df_x)
  X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size= 0.20, random_state=42)
  print('x_train shape:', X_train.shape)
  print('x_test shape:', X_test.shape)

  Y_train = np_utils.to_categorical(Y_train, nb_classes)
  Y_test = np_utils.to_categorical(Y_test, nb_classes)

  return X_train, X_test, Y_train, Y_test


def run(args):
    data_path=args['in']
    epochs = args['ep']
    PL     = 6213

    model = AggAtnnBin()
    X_train, X_test, Y_train, Y_test = load_data(data_path, PL, 2)

    history = model.fit(X_train, Y_train, 
                        validation_data=(X_test, Y_test))


    score = model.model.evaluate(model.X_test, model.Y_test, verbose=0)
    print("score: ", score)
    return score



#if __name__ == '__main__':
#
#  psr = argparse.ArgumentParser(description='input agg csv file')
#  psr.add_argument('--in',  default='in_file')
#  psr.add_argument('--ep',  type=int, default=400)
#  args=vars(psr.parse_args())
#  print(args)
#
#  run(args)
#  try:
#    K.clear_session()
#  except AttributeError:      # theano does not have this function
#    pass


def graphs(X_test):

    Y_predict = model.predict(X_test)
    threshold = 0.5
    Y_pred_int  = (Y_predict[:,0] < threshold).astype(np.int)
    Y_test_int = (Y_test[:,0] < threshold).astype(np.int)

    class_names=["Non-Response","Response"]
    
    # Compute confusion matrix
    cnf_matrix = sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig('Agg_attn_bin.confusion_without_norm.pdf', bbox_inches='tight')

    plt.close()

    # Plot normalized confusion matrix
    #plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('Agg_attn_bin.confusion_with_norm.pdf', bbox_inches='tight')

    plt.close()

    print(sklearn.metrics.roc_auc_score(Y_test_int, Y_pred_int))
    print(sklearn.metrics.balanced_accuracy_score(Y_test_int, Y_pred_int))
    print(sklearn.metrics.classification_report(Y_test_int, Y_pred_int))
    print(sklearn.metrics.confusion_matrix(Y_test_int, Y_pred_int))
    print("score")
    print(score)

    #exit()

    # summarize history for accuracy                                                                                                              
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('Agg_attn_bin.accuracy.png', bbox_inches='tight')
    plt.savefig('Agg_attn_bin.accuracy.pdf', bbox_inches='tight')

    plt.close()

    # summarize history for loss                                                                                                                  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('Agg_attn_bin.loss.png', bbox_inches='tight')
    plt.savefig('Agg_attn_bin.loss.pdf', bbox_inches='tight')


    print('Test val_loss:', score[0])
    print('Test accuracy:', score[1])

    # serialize model to JSON                                                                                                                     
    model_json = model.to_json()
    with open("Agg_attn_bin.model.json", "w") as json_file:
            json_file.write(model_json)

    # serialize model to YAML                                                                                                                     
    model_yaml = model.to_yaml()
    with open("Agg_attn_bin.model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)


    # serialize weights to HDF5                                                                                                                   
    model.save_weights("Agg_attn_bin.model.h5")
    print("Saved model to disk")

    # load json and create model                                                                                                                  
    json_file = open('Agg_attn_bin.model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)


    # load yaml and create model                                                                                                                  
    yaml_file = open('Agg_attn_bin.model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model_yaml = model_from_yaml(loaded_model_yaml)


    # load weights into new model                                                                                                                 
    loaded_model_json.load_weights("Agg_attn_bin.model.h5")
    print("Loaded json model from disk")

    # evaluate json loaded model on test data                                                                                                     
    loaded_model_json.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

    print('json Validation loss:', score_json[0])
    print('json Validation accuracy:', score_json[1])

    print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))


    # load weights into new model                                                                                                                 
    loaded_model_yaml.load_weights("Agg_attn_bin.model.h5")
    print("Loaded yaml model from disk")

    # evaluate loaded model on test data
    loaded_model_yaml.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

    print('yaml Validation loss:', score_yaml[0])
    print('yaml Validation accuracy:', score_yaml[1])
    print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1]*100))

    # predict using loaded yaml model on test and training data
    predict_yaml_train = loaded_model_yaml.predict(X_train)
    predict_yaml_test = loaded_model_yaml.predict(X_test)

    print('Yaml_train_shape:', predict_yaml_train.shape)
    print('Yaml_test_shape:', predict_yaml_test.shape)

    predict_yaml_train_classes = np.argmax(predict_yaml_train, axis=1)
    predict_yaml_test_classes = np.argmax(predict_yaml_test, axis=1)

    np.savetxt("Agg_attn_bin_predict_yaml_train.csv", predict_yaml_train, delimiter=",", fmt="%.3f")
    np.savetxt("Agg_attn_bin_predict_yaml_test.csv", predict_yaml_test, delimiter=",", fmt="%.3f")
    np.savetxt("Agg_attn_bin_predict_yaml_train_classes.csv", predict_yaml_train_classes, delimiter=",",fmt="%d")
    np.savetxt("Agg_attn_bin_predict_yaml_test_classes.csv", predict_yaml_test_classes, delimiter=",",fmt="%d")
