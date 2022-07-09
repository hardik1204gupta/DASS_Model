# Header Files
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

Data=pd.read_excel('Combined_Data.xlsx')  # Loading Dataset

depression_x=Data[["I couldn't seem to experience any positive feeling at all.",
                  "I found it difficult to work up the initiative to do things.",
                  "I felt that I had nothing to look forward to.",
                  "I felt down-hearted and blue.",
                  "I was unable to become enthusiastic about anything.",
                  "I felt I wasn’t worth much as a person.",
                  "I felt that life was meaningless."]].values # features for depression data

depression_y=Data['Depression_Remark'].values # target for depression data

depression_x_train,depression_x_test,depression_y_train,depression_y_test=train_test_split(depression_x,depression_y,test_size=0.2) # Train & Test 

Depression_Model=SVC() # instance of SVC

Depression_Model.fit(depression_x_train,depression_y_train) # fitting our data


pickle.dump(Depression_Model,open('depression.pkl','wb'))
