import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_split(file_dir, task, reg_label_dir='graph_labels/bs', class_label_dir='graph_labels/gender', test=False):
    # the goal of this block of code is to remove any files that we do not want to include in the train-test split
    file_names = []
    all_filenames = os.listdir(file_dir)
    for input_filename in all_filenames:
        initial_matrix = np.load(os.path.join(file_dir, input_filename))
        initial_matrix = np.absolute(initial_matrix)
        # remove graphs that are mostly zero
        if np.where(initial_matrix!=0)[0].size < 2000:
            continue
        label = np.load(os.path.join(reg_label_dir, input_filename))
        # some labels are nan - ignore them
        if np.any(np.isnan(label)):
            continue
        file_names.append(input_filename)
    # grab all of the classification labels based on the files we want to use
    label_dict = {file_id: None for file_id in file_names}
    for file in file_names:
        label_dict[file] = np.load(os.path.join(class_label_dir, file)).item()
    label_series = pd.Series(label_dict)
    # make the number of labels even when doing classification
    if task == 'classification':
        ones_len = label_series[label_series==1].shape[0]
        zeroes_len = label_series[label_series==0].shape[0]
        if ones_len>zeroes_len:
            drop_vals = ones_len - zeroes_len
        elif ones_len>zeroes_len:
            drop_vals = zeroes_len - ones_len
        # keep the leftover indices so we can use them when doing inference
        extra_indices = label_series[label_series==1].sample(drop_vals).index
        leftovers = label_series.loc[extra_indices]
        label_series = label_series.drop(extra_indices)
    # create a stand-in value for x and do a stratified train test split - don't care about stratifying if doing regression
    x_temp = label_series.index.to_numpy()
    if task == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(x_temp, label_series.to_numpy(), 
                                                            stratify=label_series.to_numpy(), random_state=0, test_size=0.15)
        # if testing add the removed labels back for inference
        if test:
            X_test = np.concatenate((X_test, leftovers.index.to_numpy()), axis=0)
    else:
        X_train, X_test, y_train, y_test = train_test_split(x_temp, label_series.to_numpy(), 
                                                            random_state=0, test_size=0.15)

    return X_train.tolist(), X_test.tolist()
    
