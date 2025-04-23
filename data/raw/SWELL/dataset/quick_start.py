import pandas as pd
import numpy as np
import os
import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#NOTE: This code is just a quick and dirty proof of concept. Our implimentation in the paper is completely different

def root_directory():
    current_path = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(current_path, os.pardir))
def data_directory():
    return os.path.join(root_directory(), "data")

def load_train_set():
    #Loading a hdf5 file is much much faster
    in_file = os.path.join(data_directory(), "final",  "train.csv")
    return pd.read_csv(in_file)
def load_test_set():
    #Loading a hdf5 file is much much faster
    in_file = os.path.join(data_directory(), "final",  "test.csv")
    return pd.read_csv(in_file)

def simple_model_evaluation():
    select = SelectKBest(k=20)
    train =load_train_set()
    test = load_test_set()
    target = 'condition'
    hrv_features = list(train)
    hrv_features = [x for x in hrv_features if x not in [target]]
    X_train= train[hrv_features]
    y_train= train[target]
    X_test = test[hrv_features]
    y_test = test[target]
    classifiers = [
                    RandomForestClassifier(n_estimators=100, max_features='log2', n_jobs=-1),
                    SVC(C=20, kernel='rbf'),   
                 ]
    for clf in classifiers:
        name = str(clf).split('(')[0]
        if 'svc' == name.lower():
            # Normalize the attribute values to mean=0 and variance=1
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        clf = RandomForestClassifier()
        steps = [('feature_selection', select),
             ('model', clf)]
        pipeline = sklearn.pipeline.Pipeline(steps)
        pipeline.fit(X_train, y_train)
        y_prediction = pipeline.predict(X_test)
        print("----------------------------{0}---------------------------".format(name))
        print(sklearn.metrics.classification_report(y_test, y_prediction))
        print()
        print()
                
    
    
     
if __name__ == '__main__':
    #NOTE: This code is just a quick and dirty proof of concept. 
    #Our implimentation in the paper is completely different
    simple_model_evaluation()
    
   


   