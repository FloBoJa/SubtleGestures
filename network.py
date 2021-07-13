from __future__ import print_function, division

import json
import os
import time
from datetime import datetime
from datetime import timedelta as dttimedelta
import socket

import seaborn as sn
import torch
import pandas as pd
import numpy as np
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dictionary = {"TILT_HEAD_LEFT.csv": 0, "TILT_HEAD_RIGHT.csv": 1, "TAP_GLASSES_LEFT.csv": 2,
              "TAP_GLASSES_RIGHT.csv": 3, "SLOW_NOD.csv": 4, "PUSH_GLASSES_UP.csv": 5,
              "READJUST_GLASSES_LEFT.csv": 6, "READJUST_GLASSES_RIGHT.csv": 7,
              "TAP_NOSE_LEFT.csv": 8, "TAP_NOSE_RIGHT.csv": 9, "RUB_NOSE.csv": 10}

def myCollate(batch):
    lengths = [item[0].shape[0] for item in batch]
    target = [item[1] for item in batch]
    max_length = max(lengths)
    padded_X = torch.zeros((len(batch), max_length, 11))
    padded_Y = torch.zeros((len(batch), 1))

    for i, x_len in enumerate(lengths):
        sequence = torch.tensor(batch[i][0])
        padded_X[i, 0:x_len] = sequence[:x_len]

    for i, y_len in enumerate(target):
        padded_Y[i, 0] = dictionary[batch[i][1]]


    return [padded_X, target, lengths]

def myCollate2(batch):
    tensors = [torch.tensor(item[0], dtype=torch.float) for item in batch]
    target = [item[1] for item in batch]

    return [tensors, target]



class GestureDataset(Dataset):

    def __init__(self, csv_files):
        self.gestures = []
        self.labels = []

        for root, dirs, files in os.walk(csv_files):
            for file in files:
                self.gestures.append(pd.read_csv(os.path.join(root, file)))
                self.labels.append(file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return np.array(self.gestures[item]), ''.join([i for i in self.labels[item] if not i.isdigit()])

def last_timestep(self, unpacked, lengths):
    # Index of the last output for each sequence.
    idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                           unpacked.size(2)).unsqueeze(1)
    return unpacked.gather(1, idx).squeeze()

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, tagset_size):
        super().__init__()

        self.relu = nn.ReLU()
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)

        self.dropout = nn.Dropout(0.5)
        self.lstm2hidden = nn.Linear(hidden_dim, 512)
        self.hidden2hidden = nn.Linear(512,256)
        self.hidden2hidden2 = nn.Linear(256,256)
        self.hidden2tag = nn.Linear(256, tagset_size)

        self.m = nn.LogSoftmax(dim=1)

    def forward(self, data, h2, c2):
        packed_input = pack_sequence(data, enforce_sorted=False).to(device)
        lstm_out, (h1, c1) = self.lstm(packed_input)
        seq_unpacked, lens_unpacked = pad_packed_sequence(lstm_out, batch_first=True)
        x = self.lstm2hidden(h1[-1])
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.hidden2hidden(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.hidden2hidden2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        tag_space = self.hidden2tag(x)
        tags = self.m(tag_space)
        return tags, (h1, c1)

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

def validate():
    model.eval()
    correct = 0
    incorrect = 0

    y_pred = []
    y_true = []

    h = torch.zeros(2, 1, hidden_nodes).to(device)
    c = torch.zeros(2, 1, hidden_nodes).to(device)

    for data_vector, tag in validation_loader:

        tag_scores, _ = model(data_vector, h, c)

        tag_scores = tag_scores.tolist()

        for x in range(len(tag_scores)):
            maximum = max(tag_scores[x])
            if (tag_scores[x][dictionary[tag[x]]] == maximum):
                correct += 1
            else:
                incorrect += 1


    print(correct)
    print(incorrect)
    print(correct / (correct + incorrect))

    correct = 0
    incorrect = 0

    h = torch.zeros(2, 60, hidden_nodes).to(device)
    c = torch.zeros(2, 60, hidden_nodes).to(device)

    for data_vector, tag in train_loader:

        tag_scores, _ = model(data_vector, h, c)

        tag_scores = tag_scores.tolist()

        for x in range(len(tag_scores)):
            maximum = max(tag_scores[x])
            if (tag_scores[x][dictionary[tag[x]]] == maximum):
                correct += 1
            else:
                incorrect += 1

    print(correct)
    print(incorrect)
    print(correct / (correct + incorrect))

    model.train()



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("dictionary.json", "rb") as json_file:
        json_data = json.load(json_file)

    train = True
    live = False
    gestureDataSet = GestureDataset(csv_files=json_data.get('labeledDataSave'))
    validation_split = .2
    shuffle_dataset = True
    random_seed = 10

    # Creating data indices for training and validation splits:
    dataset_size = len(gestureDataSet)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    minLoss = torch.tensor(2000)


    hidden_nodes = 1024

    train_loader = DataLoader(gestureDataSet, batch_size=128, sampler=train_sampler, collate_fn=myCollate2, drop_last=True)

    validation_loader = DataLoader(gestureDataSet, batch_size=1, sampler=valid_sampler, collate_fn=myCollate2)


    model = Net(11, hidden_nodes, 11)

    if os.path.exists("./model"):
        print("Loading existing model")
        model.load_state_dict(torch.load("./model"))
    model = model.to(device)

    if train:
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        epoch = -1
        desiredEpochs = 750  # change this to desired epoch
        while epoch < desiredEpochs:
            epoch += 1
            lossEnd = 0
            h = torch.randn(2, 60, hidden_nodes)
            c = torch.randn(2, 60, hidden_nodes)
            h = h.to(device)
            c = c.to(device)
            for data_vector, tag in train_loader:
                model.zero_grad()

                tag_scores, (hn, cn) = model(data_vector, h, c)
                targets = [dictionary[target] for target in tag]

                loss = loss_function(tag_scores, torch.tensor(targets).to(device))

                lossEnd = lossEnd + loss

                loss.backward()
                optimizer.step()

            print("epoch:", epoch)
            print(lossEnd)
            # auto save for the impatient
            if lossEnd < minLoss:
                minLoss = lossEnd
                torch.save(model.state_dict(), "./model")

            if epoch % 25 == 0:
                validate()

            #scheduler.step(lossEnd)

        #torch.save(model.state_dict(), "./model")

    elif not live: # offline classification
        model.eval()
        correct = 0
        incorrect = 0

        y_pred = []
        y_true = []

        h = torch.zeros(2, 1, hidden_nodes).to(device)
        c = torch.zeros(2, 1, hidden_nodes).to(device)

        for data_vector, tag in validation_loader:

            tag_scores, _ = model(data_vector, h, c)

            tag_scores = tag_scores.tolist()

            for x in range(len(tag_scores)):
                maximum = np.argmax(tag_scores[x])
                if (tag_scores[x][dictionary[tag[x]]] == tag_scores[x][maximum]):
                    correct += 1
                else:
                    incorrect += 1


            y_pred.extend([maximum])
            y_true.extend([dictionary[tag[0]]])

        print("correct: " + str(correct))
        print("incorrect: " + str(incorrect))
        print("F1-score: " + str(correct / (correct + incorrect)))


        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in dictionary.keys()],
                             columns=[i for i in dictionary.keys()])
        plt.figure(figsize=(25, 25))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('output.png')

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

