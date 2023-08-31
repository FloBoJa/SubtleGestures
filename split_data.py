from datetime import datetime
import os
import json
import pickle

if __name__ == '__main__':

    with open("dictionary.json") as json_file:
        json_data = json.load(json_file)

    labeledData = json_data.get("labeledDataSave")
    headerLength = json_data.get("labeledDataHeaderLength")

    timeStampData = json_data.get("timestampSaveData")
    timestampName = json_data.get("timeStampDataName")

    with open(os.path.join(timeStampData, timestampName + ".pkl"), 'rb') as input:
        labelList = pickle.load(input)

    files = [os.path.join(labeledData, f) for f in os.listdir(labeledData) if os.path.isfile(os.path.join(labeledData, f))]

    newestFile = max(files, key=lambda x: os.path.getctime(x))

    with open(newestFile, 'r') as loggerFile:
        loggerContent = loggerFile.readlines()

    firstlines = loggerContent[:headerLength]
    loggerContent = loggerContent[headerLength:]

    counter = 1

    for label in labelList:
        relevantLines = []
        for x in loggerContent:
            splitLine = x.split(",")
            timestamp = datetime.strptime(splitLine[2], '%Y.%m.%d %H:%M:%S.%f')
            if label[0] <= timestamp <= label[1]:
                relevantLines += [','.join(splitLine[3:])]

        with open(os.path.join(labeledData, label[2] + str(counter) + ".csv"), 'w') as nextFile:
            # nextFile.writelines(firstlines)
            nextFile.writelines(relevantLines)

        counter += 1
