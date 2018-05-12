import numpy as np
import tensorflow as tf
import argparse
import os
import time
import sys

import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from preact_resnet_model import ResNet_model
import utils

def lr_schedule_100(epoch):
    if(epoch < 50):
        return 1.e-3
    elif(epoch < 75):
        return 1.e-4
    else:
        return 1.e-5
def lr_schedule_200(epoch):
    if(epoch < 100):
        return 1.e-3
    elif(epoch < 150):
        return 1.e-4
    else:
        return 1.e-5

def parse_argument(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--regularization", type=int, default=0.0001)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="output/resnet/")
    parser.add_argument("--from_pretrain", type=str, default=None)
    parser.add_argument("--resnet_depth", type=int, default=25)
    parser.add_argument("--mode", type=str, default='both', choices=['train', 'eval', 'both'])
    parser.add_argument("--gpu_fraction", type=float, default=1.)
    
    return parser.parse_args(argv)

def main(args):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_fraction
    set_session(tf.Session(config=config))

    if(not os.path.exists(args.output_dir)):
        os.makedirs(args.output_dir)
    
    (X_train, y_train), (X_test, y_test) = utils.load_cifar(args.normalize)

    train_mode, eval_mode = utils.parse_mode(args.mode)

    if(args.mode == 'eval' and args.from_pretrain == None):
        print("from_pretrain argument needs to be specified in eval mode")
        return

    model = ResNet_model(res_layer_params=(3, 32, args.resnet_depth), reg=args.regularization)
    model.summary()

    #load model from pretrain
    if(args.from_pretrain != None):
        print("Loading weight from pretrain: {}".format(args.from_pretrain))
        model.load_weights(args.from_pretrain)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #train model
    if(train_mode):
        save_checkpoint = ModelCheckpoint(args.output_dir+"/model.h5", monitor='val_acc', save_best_only=True, mode='max')
        early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        if(args.epochs == 100):
            lr_scheduler = LearningRateScheduler(lr_schedule_100)
        elif(args.epochs == 200):
            lr_scheduler = LearningRateScheduler(lr_schedule_200)

        start_time = time.time()
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=args.batch_size, callbacks=[early_stop, lr_scheduler, save_checkpoint])
        end_time = time.time()
        print("Model took {} seconds to train".format(end_time-start_time))
        utils.plot_model_history(history, args.output_dir)

    #evaluate model
    if(eval_mode):
        if(train_mode):
            model.load_weights(args.output_dir+'/model.h5')
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        scores = model.evaluate(X_test, y_test)
        print("{}: {}, {}: {}".format(model.metrics_names[0], scores[0], model.metrics_names[1], scores[1]))

if __name__ == '__main__':
    args = parse_argument(sys.argv[1:])
    utils.print_arguments(args)
    
    main(args)