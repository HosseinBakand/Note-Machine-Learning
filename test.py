import mido
import pickle
#from mxm.midifile import MidiOutFile
import os
from mido import MidiFile
from sklearn.neural_network import MLPClassifier
import numpy as np



loaded_model = pickle.load(open('clf_mlp.sav', 'rb'))

directory = 'D:/CE/dars/learning machine/project/firstPhase/validation/groundTruth/'
#directory = 'D:/CE/dars/learning machine/project/firstPhase/validation/query/'
allTrack=[]

for filename in os.listdir(directory):
    mid = MidiFile(directory+filename)
    vector = []
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                vector.append(msg.note)
    #print(vector)
    max =0 
    for i in range(0,len(vector)-15):
        #print("========")
        list = vector[i:i+7]+vector[i+8:i+15]
        result =loaded_model.predict(np.array(list).reshape(1, -1))
        #print(result)
        
        #print(vector[i+7])
        if(abs(result-vector[i+7])>max):
            max =abs(result-vector[i+7])
        if(abs(result-vector[i+7])>5):
            vector[i+7] = result
    newV=[]
    t = 0
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if hasattr(msg, 'note'):
                msg.note=vector[t]
                newV.append(msg.note)
                t+=1
    #print(newV)           
                


    mid.save('D:/CE/dars/learning machine/project/myCode/result/' + filename)

    print(max)






