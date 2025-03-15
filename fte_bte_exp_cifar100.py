#%%
import random
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

import numpy as np
import pickle
import torchvision, glob
from PIL import Image
from tensorflow import keras 

from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.kernel_approximation import RBFSampler
# %%
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

#%%
def get_combined_X(a, b):
    c = np.empty((a.shape[0] + b.shape[0], a.shape[1], a.shape[2], a.shape[3]), dtype=a.dtype)
    c[0::2,:,:,:] = a
    c[1::2,:,:,:] = b
    return c

def get_combined_y(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a
    c[1::2] = b
    return c

def process_data(X, y, mean, var, color=True, flip=False):
    if color:
        X = np.transpose(X, (0, 3, 1, 2))

    # idx = []
    # for i in range(len(np.unique(y))):
    #     idx.append(np.where(y == i)[0])
    # idx = np.concatenate(idx, axis=0)

    # X = X[idx]
    # y = y[idx]   

    X = X.astype(np.float32)/255
    X = (X - mean)/var
    
    if flip:
        X_flip = X[:, :, :, ::-1]
        X_full = get_combined_X(X, X_flip)
        y_full = get_combined_y(y, y)
        X = X.reshape(X.shape[0], -1)
        X_full = X_full.reshape(X_full.shape[0], -1)
        return X, y, X_full, y_full
    
    X = X.reshape(X.shape[0], -1)
    return X, y

# %%
def cross_val_data(data_x, data_y, num_points_per_task, total_task=10, shift=1):
    x = data_x.copy()
    y = data_y.copy()
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    batch_per_task = 5000 // num_points_per_task
    sample_per_class = num_points_per_task // total_task
    test_data_slot = 100 // batch_per_task

    for task in range(total_task):
        for batch in range(batch_per_task):
            for class_no in range(task * 10, (task + 1) * 10, 1):
                indx = np.roll(idx[class_no], (shift - 1) * 100)

                if batch == 0 and class_no == 0 and task == 0:
                    train_x = x[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class],
                        :,
                    ]
                    train_y = y[
                        indx[batch * sample_per_class : (batch + 1) * sample_per_class]
                    ]
                    test_x = x[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ],
                        :,
                    ]
                    test_y = y[
                        indx[
                            batch * test_data_slot
                            + 500 : (batch + 1) * test_data_slot
                            + 500
                        ]
                    ]
                else:
                    train_x = np.concatenate(
                        (
                            train_x,
                            x[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
                    train_y = np.concatenate(
                        (
                            train_y,
                            y[
                                indx[
                                    batch
                                    * sample_per_class : (batch + 1)
                                    * sample_per_class
                                ]
                            ],
                        ),
                        axis=0,
                    )
                    test_x = np.concatenate(
                        (
                            test_x,
                            x[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ],
                                :,
                            ],
                        ),
                        axis=0,
                    )
                    test_y = np.concatenate(
                        (
                            test_y,
                            y[
                                indx[
                                    batch * test_data_slot
                                    + 500 : (batch + 1) * test_data_slot
                                    + 500
                                ]
                            ],
                        ),
                        axis=0,
                    )

    return train_x, train_y, test_x, test_y

#%%
def experiment(train_x, train_y, test_x, test_y, shift, slot, embed_dim=25000, acorn=0):

    embedder = RBFSampler(gamma='scale', n_components=embed_dim)
    embedder.fit(train_x) # The scikit function ignores data passed to it, using on the input dimensions. We are not fitting anything here with data.
    train_x_ = embedder.transform(train_x)
    train_x = np.zeros((len(train_x_), len(train_x_[0])), dtype=float)

    for ii in range(train_x.shape[0]):
        train_x[ii] = train_x_[ii][0]

    # print(train_x.shape, y_train, 'dfdf')
    test_x_ = embedder.transform(test_x)
    test_x = np.zeros((len(test_x_), len(test_x_[0])), dtype=float)

    for ii in range(test_x.shape[0]):
        test_x[ii] = test_x_[ii][0]

    acc = []
    df = pd.DataFrame()
    df_single_task = pd.DataFrame()
    shifts = []
    tasks = []
    base_tasks = []
    accuracies_across_tasks = []

    for task_ii in range(10):
        print("Starting Task {} For Fold {}".format(task_ii, shift))
        if acorn is not None:
            np.random.seed(acorn)

        X=train_x[
                task_ii * 5000
                + slot * num_points_per_task : task_ii * 5000
                + (slot + 1) * num_points_per_task
            ]
        y=train_y[
                task_ii * 5000
                + slot * num_points_per_task : task_ii * 5000
                + (slot + 1) * num_points_per_task
            ]
        # print(X.shape, 'X', y.shape, np.unique(y))
        oa = OAS(assume_centered=False) # Very sample-efficient shrinkage estimator 
        model = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=oa) # Main difference between original paper code and here. Faster, easier to play but roughly equivalent to the online version: https://github.com/tyler-hayes/Deep_SLDA/blob/master/SLDA_Model.py with better-set shrinkage. Tested against original online code with hparam search for shrinkage, returns similar results (\pm 0.8)
        model.fit(X, y)
        # print(model.classes_)
        # print(test_x[task_ii * 1000 : (task_ii + 1) * 1000].shape, 'test')
        preds = model.predict(test_x[task_ii * 1000 : (task_ii + 1) * 1000])
        acc.append(
            np.mean(preds == test_y[task_ii * 1000 : (task_ii + 1) * 1000])
        )
        print('Task ', task_ii, ' accuuracy', acc[-1])
    for task_ii in range(10):
        for task_jj in range(task_ii + 1):
            shifts.append(shift)
            tasks.append(task_jj + 1)
            base_tasks.append(task_ii + 1)
            accuracies_across_tasks.append(
                acc[task_jj]
            )
    
    df['data_fold'] = shifts
    df['task'] = tasks
    df['base_task'] = base_tasks
    df['accuracy'] = accuracies_across_tasks

    df_single_task = pd.DataFrame()
    df_single_task["task"] = range(1, 11)
    df_single_task["data_fold"] = shift
    df_single_task["accuracy"] = acc

    summary = (df, df_single_task)
    print(summary)

    file_to_save = (
        "result/ranDumb"
        + "-"
        + str(slot+1)
        + "-"
        + str(shift+1)
        + ".pickle"
    )
    with open(file_to_save, "wb") as f:
        pickle.dump(summary, f)
# %%
### MAIN HYPERPARAMS ###
num_points_per_task = 500
cifar_mean = (0.5071, 0.4867, 0.4408)
cifar_var = (0.2675, 0.2565, 0.2761)
embed_dim = 1000
########################

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()
data_x = np.concatenate([X_train, X_test])
data_y = np.concatenate([y_train, y_test])
data_y = data_y[:, 0]
# %%
slots = 10
shifts = 6

for shift in range(shifts):
    train_x, train_y, test_x, test_y = cross_val_data(
        data_x, data_y, num_points_per_task, shift=shift
    )
    mean, var = np.array(cifar_mean)[np.newaxis, :, np.newaxis, np.newaxis], np.array(cifar_var)[np.newaxis, :, np.newaxis, np.newaxis]
    train_x, train_y = process_data(train_x, train_y, mean, var, color=True)
    test_x, test_y = process_data(test_x, test_y, mean, var, color=True)
    # print(np.unique(train_y[:500]))
    for slot in range(slots):
        experiment(train_x, train_y, test_x, test_y, shift, slot, embed_dim=embed_dim)