#!/usr/bin/python
from __future__ import print_function
import numpy as np
import glob

def main():
    predict_matrix_23_name = glob.glob("small_patch_clean_0*_20_modelsmall_patch_clean_023_matrix.npy")
    predict_matrix_24_name = glob.glob("small_patch_clean_0*_20_modelsmall_patch_clean_024_matrix.npy")

    predict_matrix_23_all_models = []
    predict_matrix_24_all_models = []

    real_label_23 = np.load("small_patch_clean_023_label.npy")
    real_label_24 = np.load("small_patch_clean_024_label.npy")
    
    for matrix_name in predict_matrix_23_name:
        predict_matrix_23 = np.load(matrix_name)
        predict_matrix_23[predict_matrix_23 >= 0.5] = 1
        predict_matrix_23[predict_matrix_23 < 0.5] = 0
        predict_matrix_23_all_models.append(predict_matrix_23)


    for matrix_name in predict_matrix_24_name:
        predict_matrix_24 = np.load(matrix_name)
        predict_matrix_24[predict_matrix_24 >= 0.5] = 1
        predict_matrix_24[predict_matrix_24 < 0.5] = 0
        predict_matrix_24_all_models.append(predict_matrix_24)


    predict_matrix_23 = np.asarray(predict_matrix_23_all_models)
    predict_matrix_24 = np.asarray(predict_matrix_24_all_models)
    
    predict_matrix_23_vote = np.mean(predict_matrix_23, axis=0)
    predict_matrix_24_vote = np.mean(predict_matrix_24, axis=0)


    predict_matrix_23_vote[predict_matrix_23_vote >= 0.5] = 1
    predict_matrix_23_vote[predict_matrix_23_vote < 0.5] = 0

    predict_matrix_24_vote[predict_matrix_24_vote >= 0.5] = 1
    predict_matrix_24_vote[predict_matrix_24_vote < 0.5] = 0
    

    match_23 = real_label_23 == predict_matrix_23_vote
    match_24 = real_label_24 == predict_matrix_24_vote


    correct_23 = np.count_nonzero(match_23)
    correct_24 = np.count_nonzero(match_24)

    case_23, labels_23 = match_23.shape
    case_24, labels_24 = match_24.shape

    total_23 = case_23 * labels_23
    total_24 = case_24 * labels_24

    accuracy = float(correct_23 + correct_24)/float(total_23 + total_24)
    print(accuracy)



if __name__ == "__main__":
    main()

