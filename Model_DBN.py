import numpy as np
from DBN_Models import SupervisedDBNClassification
from Evaluation import evaluation

def Model_DBN(Images,Target):
    classifier = SupervisedDBNClassification(hidden_layers_structure=[128,128],learning_rate_rbm=0.05,learning_rate=0.01,
                                             n_epochs_rbm=5,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    for i in range(Target.shape[1]):
        classifier.fit(Images, Target[:, i])
    d=1