#!/usr/bin/python
from __future__ import print_function
import glob

def main():
    files = glob.glob("2*.txt")
    count = .0
    total_count = .0
    for f in files:
        with open(f) as fin:
            for line in fin.readlines():
                star, score = line.strip().split("\t")
                star = int(star)
                score = float(score)
                total_count += 1
                if (star > 3 and score >= -0.1) or (star <= 3 and score < -0.1):
                    count += 1

    
    accuracy = count / total_count
    print(accuracy)

if __name__=="__main__":
    main()

