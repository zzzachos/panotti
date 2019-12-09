#! /usr/bin/env python3
'''
Classify sounds using database
Author: Zzzachos

Adapted from Scott H. Hawley's code to run experiments. In particular, I found that removing most of the regularizing elements
led to better performance. 

This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

NOPE. Need to change. Trained using Fraunhofer IDMT's database of monophonic guitar effects,
   clips were 2 seconds long, sampled at 44100 Hz
'''
from __future__ import print_function
import sys
print(sys.path)
print(sys.version)
import numpy as np
from panotti.models import *
from panotti.datautils import *
#from keras.callbacks import ModelCheckpoint #,EarlyStopping
import os
from os.path import isfile
from timeit import default_timer as timer
from panotti.multi_gpu import MultiGPUModelCheckpoint
from panotti.mixup_generator import MixupGenerator
import math


def train_network(weights_file_in="weights.hdf5", weights_file_out = "weights.hdf5", classpath="Preprocessed/Train/",
    epochs=50, batch_size=20, val_split=0.2, tile=False, max_per_class=0,optimizer="adadelta", learningrate = 1,convdropout=0.5, densdropout=0.6):

    np.random.seed(1)  # fix a number to get reproducibility; comment out for random behavior

    # Get the data
    X_train, Y_train, paths_train, class_names = build_dataset(path=classpath,
        batch_size=batch_size, tile=tile, max_per_class=max_per_class)

    # Instantiate the model
    model, serial_model = setup_model(X_train, class_names, weights_file_in=weights_file_in, optimizer=optimizer, lr = learningrate, convdropout=convdropout, densdropout = densdropout)

    save_best_only = (val_split > 1e-6) #want to get rid of cross-validation

    #split_index = int(X_train.shape[0]*(1-val_split))
    #X_val, Y_val = X_train[split_index:], Y_train[split_index:]
    #X_train, Y_train = X_train[:split_index-1], Y_train[:split_index-1]

    checkpointer = MultiGPUModelCheckpoint(filepath=weights_file_out, verbose=1, save_best_only=save_best_only,
          serial_model=serial_model, period=1, class_names=class_names)

    steps_per_epoch = X_train.shape[0] // batch_size
    if False and ((len(class_names) > 2) or (steps_per_epoch > 1)):
        training_generator = MixupGenerator(X_train, Y_train, batch_size=batch_size, alpha=0.25)()
        model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch,
              epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpointer])#,validation_data=(X_val, Y_val))
    elif save_best_only:
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpointer], validation_split=val_split,validation_data=(X_val, Y_val))
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
              verbose=1, callbacks=[checkpointer])#v5 commented out validation_split=val_split, validation_data=(X_train, Y_train))
    #else:
       # model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
             # verbose=1, callbacks=[checkpointer], #validation_split=val_split)
             #)# validation_data=(X_val, Y_val))

    # overwrite text file class_names.txt  - does not put a newline after last class name
    with open('class_names.txt', 'w') as outfile:
        outfile.write("\n".join(class_names))

    # Score the model against Test dataset
    X_test, Y_test, paths_test, class_names_test  = build_dataset(path=classpath+"../Dev/", tile=tile)
    assert( class_names == class_names_test )
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(optimizer, " gives the following results:")
    print('Dev set loss:', score[0])
    print('Dev set accuracy:', score[1])
    




def runoptimizerexperiment(list, rates, directory):
    for l in list:
        for r in rates:
            path = l + str(r) + "weights.hdf5"
            train_network( weights_file_out = path, optimizer=l, learningrate = r)

def rundropoutexperiment(clist, dlist, directory):
    for c in clist:
        for d in dlist:
            path = "convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            train_network(weights_file_out = path, convdropout=c, densdropout=d)

def runbatchsizeexperiment(list):
    for l in list:
        path = "batchsize" + str(l) + "weights.hdf5"
        train_network(weights_file_out = path, epochs=25, batch_size = l, val_split=0,convdropout=0.3, densdropout=0.2)

def newrundropoutexperiment(version, clist, dlist, epochnum): #this is designed to have no cross-validation
    for c in clist:
        for d in dlist:
            path = str(version)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            train_network(weights_file_out = path, epochs=epochnum, batch_size=64, val_split=0, convdropout=c, densdropout=d)

def continuerundropoutexperiment(version, clist, dlist, epochnum): #this is designed to have no cross-validation
    for c in clist:
        for d in dlist:
            inpath = str(version)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            outpath = str(version)+str(.5)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            path = str(version)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            train_network(weights_file_in = inpath, weights_file_out = outpath, epochs=epochnum, batch_size=128, val_split=0, convdropout=c, densdropout=d)        

def morecontinuerundropoutexperiment(version, clist=[0.5], dlist=[0.7], epochnum=10): #this is designed to have no cross-validation
    for c in clist:
        for d in dlist:
            inpath = str(version)+str(.5)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            outpath = str(version)+str(.8)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            path = str(version)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            train_network(weights_file_in = inpath, weights_file_out = outpath, epochs=epochnum, batch_size=128, val_split=0, convdropout=c, densdropout=d)    

def newnewrundropoutexperiment(version, clist, dlist, epochnum, batchs): #this is designed to have no cross-validation
    for c in clist:
        for d in dlist:
            inpath = str(version)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            outpath = str(version)+"_convdropout"+str(c) + "densedropout"+str(d) +"weights.hdf5"
            train_network(weights_file_out = outpath, epochs=epochnum, batch_size=batchs, val_split=0, convdropout=c, densdropout=d)

def newrunoptimizerexperiment(version, list, rates, directory, epochnum, batchs):
    for l in list:
        for r in rates:
            path = str(version)+l + str(r) + "weights.hdf5"
            train_network( weights_file_out = path, optimizer=l, learningrate = r,epochs=epochnum, batch_size=batchs, val_split=0, convdropout=0.4, densdropout=0.6)

if __name__ == '__main__':
#The following two experiments were run with cross-validation as originally written.
    #runoptimizerexperiment(["Adam"], [0.01, 0.0001],"")#"adadelta",
    #rundropoutexperiment([0.5,0.3,0.1],[0.6,0.4,0.2],"")
    
    #I accidentally set this up to not save best only, just to save most recent. whoops! oh well
#for some reason, it is still giving me the "large dropout rate" warning even though ... I should have accounted for that? confused.)
    #rundropoutexperiment([0.3,0.1],[0.4,0.2],"")
    

#Now we run a shorter experiment without any cross-validation
    #runbatchsizeexperiment([20,40,80])
    
#Now we run dropout rate expreiment without any cross-validation and with best batch size... the last experiment had a real overfitting problem 
    #newrundropoutexperiment(2,[0.3,0.5], [0.2,0.4], 25)
    #newrundropoutexperiment(3,[0.3,0.5], [0.2,0.4], 50)
    #newrundropoutexperiment(4,[0.5,0.3], [0.4,0.2], 40)
    #newrundropoutexperiment(5,[0.5,0.7], [0.6, 0.4,0.8],25)
    #do the previous one for ten more epochs each!
    #continuerundropoutexperiment(5,[0.5,0.7], [0.6, 0.4,0.8],10)#batch size 64
    #newnewrundropoutexperiment(6,[0.5,0.6,0.4], [0.6, 0.7], 30,128)
    #continuerundropoutexperiment(6,[0.5], [0.6,0.7],10)#batch size 128
    #newnewrundropoutexperiment(7,[0.4,0.3],[0.5],30,128)
    #newrunoptimizerexperiment(8, ["Adam"], [0.001, 0.0001],"", 40, 128)
    morecontinuerundropoutexperiment(6)
    


#need to add evaluate network parts
#very frustratingly, adadelta needs to be a string and Adams needs to be a function, with eval. Frustrating.     
    
    
#    import argparse
#     parser = argparse.ArgumentParser(description="trains network using training dataset")
#     parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
#         help='weights file (in .hdf5)', default="weights.hdf5")
#     parser.add_argument('-n', '--newweights', #nargs=1, type=argparse.FileType('r'),
#         help='weights file (in .hdf5)', default="weights.hdf5")
#     parser.add_argument('-c', '--classpath', #type=argparse.string,
#         help='Train dataset directory with list of classes', default="Preprocessed/Train/")
#     parser.add_argument('--epochs', default=20, type=int, help="Number of iterations to train for")
#     parser.add_argument('--batch_size', default=40, type=int, help="Number of clips to send to GPU at once")
#     parser.add_argument('--val', default=0.2, type=float, help="Fraction of train to split off for validation")
#     parser.add_argument("--tile", help="tile mono spectrograms 3 times for use with imagenet models",action="store_true")
#     parser.add_argument('-m', '--maxper', type=int, default=0, help="Max examples per class")
#     args = parser.parse_args()
#     train_network(weights_file_in=args.weights, weights_file_out=args.newweights, classpath=args.classpath, epochs=args.epochs, batch_size=args.batch_size,
#         val_split=args.val, tile=args.tile, max_per_class=args.maxper)
