from tkinter import messagebox
from datetime import datetime
from enum import Enum
import os
import json
import pickle
import random
import sys

from tkinter import *
from tkvideo import tkvideo


class Language(Enum):
    ENGLISH = "_EN"
    GERMAN = "_DE"


def createStartPage():
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=True)

    label = Label(frame, text="Choose a language / Wähle eine Sprache")
    label.place(relx=0.5, rely=0.4, anchor=CENTER)

    languageButtonGER = Button(frame, text="German / Deutsch",
                               command=lambda: (frame.destroy(), createDescriptionWindow(Language.GERMAN)))
    languageButtonGER.place(relx=0.5, rely=0.45, anchor=CENTER)
    languageButtonEN = Button(frame, text="English / Englisch",
                              command=lambda: (frame.destroy(), createDescriptionWindow(Language.ENGLISH)))
    languageButtonEN.place(relx=0.5, rely=0.5, anchor=CENTER)


def createDescriptionWindow(language: Language):
    frame = createFrame(language)

    welcomeDescription = Text(frame, wrap='word', font=("Arial", 12))
    welcomeDescription.insert(INSERT, json_data.get("glossary").get("glossary" + language.value).get("welcomeMessage"))
    welcomeDescription.config(state=DISABLED)
    welcomeDescription.place(relx=0.5, rely=0.4, anchor=CENTER)

    languageButtonGER = Button(frame, text="Weiter",
                               command=lambda: (frame.destroy(), createGestureWindowFirst(language)))
    languageButtonGER.place(relx=0.5, rely=0.8, anchor=CENTER)


def createGestureWindowFirst(language: Language):
    gestures = json_data.get("gestures")

    if len(gestures) < 1:
        messagebox.showinfo("Missing gestures", "There aren't any gestures defined!")
        sys.exit(-1)

    createGestureWindowIteration(language, gestures)


def createGestureWindowIteration(language: Language, restOfGestures):
    frame = createFrame(language)

    if len(restOfGestures) == 0:
        frame.destroy()
        createLastWindow(language)
    else:
        random.shuffle(restOfGestures)
        gesture = restOfGestures.pop(0)

        count = gesture.get("sampleCount")

        if count > 1:
            gesture["sampleCount"] = count - 1
            restOfGestures.append(gesture)

        gestureDescription = Text(frame, wrap='word', font=("Arial", 12))
        gestureDescription.insert(INSERT, gesture.get("name" + language.value) + "\n\n")
        gestureDescription.insert(INSERT, gesture.get("description" + language.value) + "\n\n")
        gestureDescription.insert(INSERT, "Repititions / Wiederholungen: {sampleCount}".format(sampleCount=count))
        gestureDescription.config(state=DISABLED)
        gestureDescription.place(relx=0.5, rely=0.85, anchor=CENTER)

        lbl = Label(root)
        lbl.pack(pady=20)
        path = './mp4s/' + gesture.get("file")
        player = tkvideo(path, lbl, loop=1)
        player.play()

        languageButtonGER = Button(frame, text="Start",
                                   command=lambda: createStopButton(language, languageButtonGER, frame, gesture,
                                                                    restOfGestures, gestureDescription, count,
                                                                    getCurrentTime(), lbl))
        languageButtonGER.place(relx=0.8, rely=0.5, anchor=CENTER)




def createStopButton(language: Language, button, frame, gesture, restOfGestures, gestureDescription, count, startTime, lbl):
    button.destroy()
    button = Button(frame, text="stop",
                    command=lambda: (addLabel(startTime, getCurrentTime(), gesture.get("label")), print(labelList),
                                     prepareNextGestureWindowIteration(frame, restOfGestures, lbl)))
    button.place(relx=0.8, rely=0.5, anchor=CENTER)


def prepareNextGestureWindowIteration(frame, restOfGestures, lbl):
    lbl.destroy()
    frame.pack_forget()
    frame.destroy()
    createGestureWindowIteration(language, restOfGestures)


def createLastWindow(language: Language):
    frame = createFrame(language)

    automaticLabelingText = Text(frame, wrap='word', font=("Arial", 12))
    automaticLabelingText.insert(INSERT,
                                 json_data.get("glossary").get("glossary" + language.value).get("goodbyeMessage"))
    automaticLabelingText.config(state=DISABLED)
    automaticLabelingText.place(relx=0.5, rely=0.4, anchor=CENTER)

    languageButton = Button(frame, text="Exit",
                               command=lambda: (saveLabelList(), sys.exit(0)))
    languageButton.place(relx=0.5, rely=0.8, anchor=CENTER)


def saveLabelList():
    labeledData = json_data.get("timestampSaveData")
    timestampsName = json_data.get("timeStampDataName")
    savePath = labeledData + timestampsName + ".pkl"
    with open(savePath, 'wb') as output:
        pickle.dump(labelList, output, pickle.HIGHEST_PROTOCOL)



def createFrame(language: Language):
    frame = Frame(root)
    frame.pack(fill=BOTH, expand=True)

    root.title(json_data.get("glossary").get("glossary" + language.value).get("title"))

    return frame


def chooseLanguage(language: Language, chosenLanguage):
    chosenLanguage = language


def addLabel(startTime, endTime, label):
    labelList.append((startTime, endTime, label))


def getCurrentTime():
    return datetime.utcnow()


if __name__ == '__main__':
    language = Language.ENGLISH

    with open("dictionary.json") as json_file:
        json_data = json.load(json_file)

    startTime = 0
    endTime = 0

    labelList = []

    root = Tk()
    root.geometry("1280x720")
    root.title("")
    root.resizable(False, False)

    createStartPage()

    root.mainloop()
