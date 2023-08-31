import os
import json

import numpy as np
import pandas as pd
import random

if __name__ == '__main__':

    with open("dictionary.json", "rb") as json_file:
        json_data = json.load(json_file)

    allFiles = []
    counter = 9999
    for root, dirs, files in os.walk(json_data.get('labeledDataSave')):
        for file in files:
            allFiles.append(os.path.join(root, file))

    random.Random(10).shuffle(allFiles)

    train = allFiles[:int(len(allFiles) * 0.6)]  # [1, 2, 3, 4, 5, 6, 7, 8]
    validate = allFiles[int(len(allFiles) * 0.6):int(len(allFiles) * 0.8)]  # [9]
    test = allFiles[int(len(allFiles) * 0.8):]  # [10]

    print(len(train))
    print(len(validate))
    print(len(test))

    if not os.path.exists("train"):
        os.makedirs("train")

    if not os.path.exists("validate"):
        os.makedirs("validate")

    if not os.path.exists("test"):
        os.makedirs("test")

    for file in train:
        oldFile = open(file)
        print(file.split("/"))
        newFile = open(os.path.join("train", file.split("/")[-1][:-4]) + str(counter) + ".csv", 'w')
        newFile.writelines(oldFile.readlines())
        newFile.close()
        oldFile.close()
        counter += 1

    for file in validate:
        oldFile = open(file)
        newFile = open(os.path.join("validate", file.split("/")[-1][:-4]) + str(counter) + ".csv", 'w')
        newFile.writelines(oldFile.readlines())
        newFile.close()
        oldFile.close()
        counter += 1

    for file in test:
        oldFile = open(file)
        newFile = open(os.path.join("test", file.split("/")[-1][:-4]) + str(counter) + ".csv", 'w')
        newFile.writelines(oldFile.readlines())
        newFile.close()
        oldFile.close()
        counter += 1
