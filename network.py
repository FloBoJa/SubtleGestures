from __future__ import print_function, division
import os
import time
from datetime import datetime
from datetime import timedelta as dttimedelta
import socket
import torch
import pandas as pd
import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

dictionary = {"TILT_HEAD_LEFT.csv": 0, "TILT_HEAD_RIGHT.csv": 1, "TAP_GLASSES_LEFT.csv": 2,
              "TAP_GLASSES_RIGHT.csv": 3, "SLOW_NOD.csv": 4, "PUSH_GLASSES_UP.csv": 5,
              "READJUST_GLASSES_LEFT.csv": 6, "READJUST_GLASSES_RIGHT.csv": 7,
              "TAP_NOSE_LEFT.csv": 8, "TAP_NOSE_RIGHT.csv": 9, "RUB_NOSE.csv": 10, "PUSH_CHEEK_LEFT.csv": 11,
              "PUSH_CHEEK_RIGHT.csv": 12}


class GestureDataset(Dataset):

    def __init__(self, csv_files):
        self.gestures = []
        self.labels = []

        for file in os.listdir(csv_files):
            self.gestures.append(pd.read_csv(os.path.join(csv_files, file)))
            self.labels.append(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return np.array(self.gestures[item]), ''.join([i for i in self.labels[item] if not i.isdigit()])

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, tagset_size):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.m = nn.LogSoftmax(dim=1)

    def forward(self, data_vector, h, c):
        lstm_out, (h1, c1) = self.lstm(data_vector.float(), (h, c))
        tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_scores = self.m(tag_space)
        return tag_scores, (h1, c1)

def updateGestureData(gestureData, dataSocket, maxGestureLength, csvPath):
    referenceTime = datetime.utcnow()
    # remove old data
    newGestureData = []
    minTimestamp = referenceTime - dttimedelta(seconds=maxGestureLength)
    for i in range(len(gestureData[0])):
        splitLine = gestureData[0][i].split(",")
        if datetime.strptime(splitLine[2], '%Y.%m.%d %H:%M:%S.%f') >= minTimestamp:
            newGestureData = gestureData[0][i:]
            break
    gestureData[0] = newGestureData

    receivedData = ""
    while True:
        receivedData += dataSocket.recv(1024).decode("utf-8") 
        if receivedData.count("\n") >= 3:
            lastCompleteLine = receivedData.split("\n")[-2]
            newestTime = datetime.strptime(lastCompleteLine.split(",")[2], '%Y.%m.%d %H:%M:%S.%f')
            if newestTime > referenceTime:
                break
    gestureData[0] += receivedData.split("\n")[1:-2]
    
    with open(csvPath, 'w') as file:
        for line in gestureData[0]:
            file.write(",".join(line.split(",")[3:]) + "\n")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = False
    live = False
    gestureDataSet = GestureDataset(csv_files='./drive/MyDrive/LabeledData')
    hidden_nodes = 128

    train_loader = DataLoader(gestureDataSet, 1, shuffle=True)

    model = Net(11, hidden_nodes, 13)
    if os.path.exists("./model"):
        print("Loading existing model")
        model.load_state_dict(torch.load("./model"))
    model = model.to(device)

    if train:
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        #for epoch in range(750):
        epoch = -1
        while True:
            epoch += 1
            lossEnd = 0
            h = torch.randn(1, 1, hidden_nodes)
            c = torch.randn(1, 1, hidden_nodes)
            h = h.to(device)
            c = c.to(device)
            for data_vector, tag in train_loader:
                model.zero_grad()

                sentence_in = torch.tensor(data_vector, dtype=torch.long).to(device)
                targets = torch.tensor([dictionary[tag[0]]]).to(device)

                tag_scores, (hn, cn) = model(sentence_in, h, c)

                loss = loss_function(tag_scores, torch.tensor(targets))

                lossEnd = lossEnd + loss

                loss.backward()
                optimizer.step()

            print("epoch:", epoch)
            print(tag_scores)
            print(lossEnd)
            # auto save for the impatient
            if epoch > 0 and epoch % 25 == 0:
                torch.save(model.state_dict(), "./model")

        torch.save(model.state_dict(), "./model")

    elif not live: # offline classification
        correct = 0
        incorrect = 0
        
        h = torch.zeros(1, 1, hidden_nodes).to(device)
        c = torch.zeros(1, 1, hidden_nodes).to(device)
        
        for data_vector, tag in train_loader:
            sentence_in = torch.tensor(data_vector, dtype=torch.long).to(device)
            targets = torch.tensor([dictionary[tag[0]]]).to(device)

            tag_scores, _ = model(sentence_in, h, c)

            maximum = torch.max(tag_scores)

            if (tag_scores[0][targets[0]] == maximum):
                correct += 1
            else:
                incorrect += 1

        print(correct)
        print(incorrect)
    else: # live gesture recognition
        gestureData = [[]]
        # this address must be the same as in the Data Logger
        dataSource = ("192.168.178.1", 60000)
        dataFile = "./liveGestureData.csv"
        maxGestureLength = 10 # seconds
        updateFrequency = 1 # seconds

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(dataSource)

        while True:
            updateGestureData(gestureData, s, maxGestureLength, dataFile)
            
            h = torch.zeros(1, 1, hidden_nodes).to(device)
            c = torch.zeros(1, 1, hidden_nodes).to(device)
            
            sentence_in = torch.tensor([np.array(pd.read_csv(dataFile))], dtype=torch.long).to(device)
            
            tag_scores, _ = model(sentence_in, h, c)
            print(tag_scores)
                
            time.sleep(1/updateFrequency)

        s.close()
