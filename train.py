import mido
import os
from mido import MidiFile
from sklearn.neural_network import MLPClassifier
import pickle



directory = 'D:/CE/dars/learning machine/project/firstPhase/train/'
allTrack=[]
for filename in os.listdir(directory):
    mid = MidiFile(directory+filename)
    vector = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                vector.append(msg.note)
    allTrack.append(vector)
    #print(filename)


data=[]
label = []
for track in allTrack :
    list=[]
    size = len(track)
    if size>14:
        for i in range(0,size-15):
            list.append(track[i:i+15])
    
    for l in list:
        data.append(l[0:7]+l[8:15])
        label.append(l[7])


clf = MLPClassifier(hidden_layer_sizes=(12,12))

clf.fit(data, label)


filename = 'clf_mlp.sav'
pickle.dump(clf, open(filename, 'wb'))

