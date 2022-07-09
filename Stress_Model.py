# Header Files
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

Data=pd.read_excel('Combined_Data.xlsx')  # Loading Dataset

stress_x=Data[["I found it hard to wind down.",
                     "I tended to over-react to situations.",
                     "I felt that I was using a lot of nervous energy.",
                     "I found myself getting agitated.",
                     "I found it difficult to relax.",
                     "I was intolerant of anything that kept me from getting on with what I was doing.",
                     "I felt that I was rather touchy."]].values # features for stress data

stress_y=Data['Stress_Remark'].values # target for stress data

stress_x_train,stress_x_test,stress_y_train,stress_y_test=train_test_split(stress_x,stress_y,test_size=0.2) # Train & Test 

Stress_Model=SVC() # instance of SVC

Stress_Model.fit(stress_x_train,stress_y_train) # fitting our data

pickle.dump(Stress_Model,open('stress.pkl','wb'))
