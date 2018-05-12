import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os

def load_cifar(normalize=True):
    print("Loading cifar data...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    if(normalize):
        X_train = X_train.astype(np.float32)/255.
        X_test  = X_test.astype(np.float32)/255.

    print("Training data shape: ", X_train.shape)
    print("Training labels shape: ", y_train.shape)
    print("Testing data shape: ", X_test.shape)
    print("Testing lables shape: ", y_test.shape)

    return (X_train, y_train), (X_test, y_test)

def plot_model_history(model_history, output_foler):
    fig, axs = plt.subplots(1, 2, 0, figsize=(15,15))
    #summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    # plt.show()

    if(not os.path.exists(output_foler)):
        os.makedirs(output_foler)
    plt.savefig(output_foler + '/history_summary.png')
    
def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
    print("------------------------------------------------")

def parse_mode(mode):
    if(mode == 'both'):
        train_mode = True
        eval_mode = True
    elif(mode == 'eval'):
        train_mode = False
        eval_mode = True
    else:
        train_mode = True
        eval_mode = False
    return train_mode, eval_mode