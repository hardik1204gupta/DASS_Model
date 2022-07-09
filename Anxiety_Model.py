# Header Files
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

Data=pd.read_excel('Combined_Data.xlsx')  # Loading Dataset

anxiety_x=Data[["I was aware of dryness of my mouth.",
                      "I experienced breathing difficulty (eg, excessively rapid breathing, breathlessness in absence of physical exertion too).",
                      "I experienced trembling (eg in the hands).",
                      "I was worried about situations in which I might panic and make a fool of myself.",
                      "I felt I was close to panic.",
                      "I was aware of the action of my heart in absence of physical exertion (eg, sense of heart rate increase, heart missing a beat).",
                      "I felt scared without any good reason."]].values # features for anxiety data

anxiety_y=Data['Anxiety_Remark'].values # target for anxiety data

anxiety_x_train,anxiety_x_test,anxiety_y_train,anxiety_y_test=train_test_split(anxiety_x,anxiety_y,test_size=0.2) # Train & Test 

Anxiety_Model=SVC() # instance of SVC

Anxiety_Model.fit(anxiety_x_train,anxiety_y_train) # fitting our data

pickle.dump(Anxiety_Model,open('anxiety.pkl','wb'))
