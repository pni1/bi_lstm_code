#!/usr/bin/python
from __future__ import print_function
import sys
import numpy as np

def main():
    model_index = sys.argv[1]
    
    real_label_23 = np.load("small_patch_clean_023_label.npy")
    real_label_24 = np.load("small_patch_clean_024_label.npy")


    predict_label_23_name = "small_patch_clean_0"+str(model_index)+"_20_modelsmall_patch_clean_023_matrix.npy"
    predict_label_24_name = "small_patch_clean_0"+str(model_index)+"_20_modelsmall_patch_clean_024_matrix.npy"

    predict_label_23 = np.load(predict_label_23_name)
    predict_label_24 = np.load(predict_label_24_name)

    predict_label_23[predict_label_23 >= 0.5] = 1
    predict_label_23[predict_label_23 < 0.5] = 0


    predict_label_24[predict_label_24 >= 0.5] = 1
    predict_label_24[predict_label_24 < 0.5] = 0
    

    match_23 = real_label_23 == predict_label_23
    match_24 = real_label_24 == predict_label_24

    correct_23 = np.count_nonzero(match_23)
    correct_24 = np.count_nonzero(match_24)

    case_23, labels_23 = match_23.shape
    case_24, labels_24 = match_24.shape

    total_23 = case_23 * labels_23
    total_24 = case_24 * labels_24

    accuracy = float(correct_23 + correct_24)/float(total_23 + total_24)

    print(accuracy)

if __name__=="__main__":
    main()
