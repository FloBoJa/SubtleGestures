from __future__ import print_function, division

import json
import os
import time
from datetime import datetime
from datetime import timedelta as dttimedelta
import socket

import seaborn as sn
import sklearn
import torch
import pandas as pd
import numpy as np
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tkinter import *
from threading import Thread
from torch.nn import functional

dictionary = {"TILT_HEAD_LEFT.csv": 0, "TILT_HEAD_RIGHT.csv": 1, "TAP_GLASSES_LEFT.csv": 2,
              "TAP_GLASSES_RIGHT.csv": 3, "SLOW_NOD.csv": 4, "PUSH_GLASSES_UP.csv": 5,
              "READJUST_GLASSES_LEFT.csv": 6, "READJUST_GLASSES_RIGHT.csv": 7}
    
tag_scores = []

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

def createStartPage():
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=True)

    label = Label(frame, text="""Enter the IP address and the port that are configured in the J!NS MEME Data Logger.
    The Data Logger must be in its measurement mode.""")
    label.place(relx=0.5, rely=0.2, anchor=CENTER)
    ipBox = Text(
        frame,
        height=12,
        width=40
    )
    ipBox.insert('end', "192.168.178.10")
    ipBox.place(relx=0.5, rely=0.4, anchor=E)
    portBox = Text(
        frame,
        height=12,
        width=40
    )
    portBox.insert('end', "60000")
    portBox.place(relx=0.5, rely=0.4, anchor=W)
    submitButton = Button(
        frame,
        text = "Connect",
        command = lambda: (submitButton.config(state='disabled'),
            startReceiving((ipBox.get('1.0', 'end').strip(),
                int(portBox.get('1.0', 'end').strip())),
                lambda: (frame.destroy()),
                lambda: (submitButton.config(state='normal')
                )
            )
        )
    )
    submitButton.place(relx=0.5, rely=0.7, anchor=CENTER)
    submitButton.focus_set()

def startReceiving(dataSource, onConnection, onFailure):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(dataSource)
    except Exception:
        onFailure()
        return
    onConnection()

    frame = Frame(root)
    frame.pack(fill=BOTH, expand=True)

    label = Label(frame, text="""Press the button right after you performed a gesture.
    The button can be pressed repeatedly. You can also press space bar.""")
    label.place(relx=0.5, rely=0.2, anchor=CENTER)

    classifyButton = Button(
        frame,
        text = "Classify last gesture",
        command = lambda: (
            label.config(text = getMaxTag())
        )
    )
    classifyButton.place(relx=0.5, rely=0.4, anchor=CENTER)
    classifyButton.focus_set()

    label = Label(frame, text="")
    label.place(relx=0.5, rely=0.5, anchor=CENTER)

    clientThread = Thread(target = receivingLoop, args = (s, ))
    clientThread.daemon = True
    clientThread.start()

def getMaxTag():
    maximum = float('-inf')
    for x in range(len(tag_scores)):
        maximum = max(maximum, max(tag_scores[x]))
    for x in range(len(tag_scores)):
        for key in dictionary:
            if tag_scores[x][dictionary[key]] == maximum:
                return key


def receivingLoop(s):
    global tag_scores
    dataFile = "./liveGestureData.csv"
    maxGestureLength = 3 # seconds
    updateFrequency = 0.2 # seconds

    while True:
        updateGestureData(gestureData, s, maxGestureLength, dataFile)
        
        h = torch.zeros(2, 1, hidden_nodes).to(device)
        c = torch.zeros(2, 1, hidden_nodes).to(device)
        
        sentence_in = [torch.tensor(np.array(pd.read_csv(dataFile)), dtype=torch.float).to(device)]
        
        tag_scores, _ = model(sentence_in, h, c)
        
        time.sleep(1/updateFrequency)

    s.close()

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, tagset_size):
        super().__init__()

        self.relu = nn.ELU()
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.5)

        self.dropout = nn.Dropout(0.25)
        self.lstm2hidden = nn.Linear(hidden_dim, 128)
        self.hidden2hidden = nn.Linear(128,64)
        self.hidden2hidden2 = nn.Linear(64,64)
        self.hidden2tag = nn.Linear(64, tagset_size)
        self.ln_norm = nn.LayerNorm([hidden_dim])
        self.m = nn.LogSoftmax(dim=1)

    def forward(self, data, h2, c2):
        packed_input = pack_sequence(data, enforce_sorted=False).to(device)
        lstm_out, (h1, c1) = self.lstm(packed_input)
        seq_unpacked, lens_unpacked = pad_packed_sequence(lstm_out, batch_first=True)
        x = self.ln_norm(h1[-1])
        x = self.lstm2hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden2hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.hidden2hidden2(x)
        x = self.relu(x)
        x = self.dropout(x)
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

    h = torch.zeros(5, 1, hidden_nodes).to(device)
    c = torch.zeros(5, 1, hidden_nodes).to(device)

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

    print(correct)
    print(incorrect)
    print("Accuracy:" + str(correct / (correct + incorrect)))
    print("F1-score: " + str(sklearn.metrics.f1_score(y_true, y_pred, average=None)))

    correct = 0
    incorrect = 0

    y_pred = []
    y_true = []

    h = torch.zeros(5, 60, hidden_nodes).to(device)
    c = torch.zeros(5, 60, hidden_nodes).to(device)

    for data_vector, tag in train_loader:

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

    print(correct)
    print(incorrect)
    print("Accuracy:" + str(correct / (correct + incorrect)))
    print("F1-score: " + str(sklearn.metrics.f1_score(y_true, y_pred, average=None)))

    model.train()



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("dictionary.json", "rb") as json_file:
        json_data = json.load(json_file)

    train = False
    live = False
    gestureDataSet = GestureDataset(csv_files=json_data.get('labeledDataTrainSave'))
    validationSet = GestureDataset(csv_files=json_data.get('labeledDataValidateSave'))

    testSet = GestureDataset(csv_files=json_data.get('labeledDataTestSave'))

    validation_split = .2
    shuffle_dataset = True
    random_seed = 10

    # Creating data indices for training and validation splits:
    dataset_size = len(gestureDataSet)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    minLoss = torch.tensor(2000)


    hidden_nodes = 512

    train_loader = DataLoader(gestureDataSet, batch_size=64, collate_fn=myCollate2, drop_last=True)
    validation_loader = DataLoader(validationSet, batch_size=1, collate_fn=myCollate2)
    testLoader = DataLoader(testSet, batch_size=1, collate_fn=myCollate2)


    model = Net(11, hidden_nodes, 8)

    if os.path.exists("./model"):
        print("Loading existing model")
        model.load_state_dict(torch.load("./model"))
    model = model.to(device)

    if train:
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
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

        for data_vector, tag in testLoader:

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
        print("Accuracy:" + str(correct / (incorrect + correct)))
        print("F1-score: " + str(sklearn.metrics.f1_score(y_true, y_pred, average=None)))
        print("F1-score: " + str(sklearn.metrics.f1_score(y_true, y_pred, average="micro")))
        print("F1-score: " + str(sklearn.metrics.f1_score(y_true, y_pred, average="macro")))
        print("F1-score: " + str(sklearn.metrics.f1_score(y_true, y_pred, average="weighted")))


        cf_matrix = confusion_matrix(y_true, y_pred)
        cf_matrix_n = cf_matrix.astype("float")
        cf_matrix_n.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(cf_matrix_n / cf_matrix_n.sum(axis=1), index=[i for i in dictionary.keys()],
                             columns=[i for i in dictionary.keys()])
        plt.figure(figsize=(25, 25))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('output.png')

    else: # live gesture recognition
        gestureData = [[]]

        root = Tk()
        root.geometry("1280x720")
        root.title("")
        root.resizable(False, False)

        createStartPage()

        root.mainloop()
