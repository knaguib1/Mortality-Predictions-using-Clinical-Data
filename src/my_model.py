import utils
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
    
    # no change in training data set...
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    
    # need to perform same etl for test dataset...
    
    #### read in data ####
    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    filepath = '../data/test/'
    events = pd.read_csv(filepath + 'events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
    
    #### calculate index date ####
    # subset data for alive patients
    events_temp = events.copy()
    events_temp['timestamp'] = pd.to_datetime(events_temp.timestamp)
    
    # find max timestamp for each patient_id
    indx_date = events_temp.groupby('patient_id').timestamp.max().reset_index()
    
    indx_date.rename(columns = {'timestamp':'indx_date'}, inplace=True)
    
    #### Filter events ####
    # merge relevant data frames
    filtered_events = pd.merge(events, indx_date,  on='patient_id')
    filtered_events = filtered_events.sort_values(by=['patient_id', 'event_id']).reset_index(drop=True)

    #change timestamp from obj to datetime
    filtered_events['timestamp'] = pd.to_datetime(filtered_events.timestamp)

    # timedelta betwen indx_date and timestamp of events
    filtered_events['window'] = filtered_events.indx_date - pd.to_timedelta(2000, unit='d')

    cond1 = filtered_events.timestamp <= filtered_events.indx_date
    cond2 = filtered_events.timestamp >= filtered_events.window

    # filter for observation window
    filtered_events = filtered_events[cond1 & cond2]
    
    #### Aggregate data ####
    # join feature map to events 
    df_temp = pd.merge(filtered_events, feature_map)

    # drop na values
    df_temp = df_temp.dropna(subset=['value'])

    aggregated_events = pd.DataFrame()

    for feat in ['DIAG', 'DRUG', 'LAB']:
        
        agg_temp = df_temp[df_temp.event_id.str.contains(feat)].groupby(['patient_id', 'idx'])
        
        if feat in ['DIAG', 'DRUG']:
            agg_temp = agg_temp.value.sum().reset_index()
            
        else:
            agg_temp = agg_temp.value.count().reset_index()
            
        norm_factor = agg_temp.groupby('idx').value.max().reset_index()
        norm_factor.rename(columns={'value':'norm_factor'}, inplace=True)
        agg_temp = pd.merge(agg_temp, norm_factor, on='idx', how='left')
        aggregated_events = pd.concat([aggregated_events, agg_temp])

    aggregated_events['value'] /= aggregated_events.norm_factor
    # print(aggregated_events.value.describe())
    aggregated_events.drop(columns='norm_factor', inplace=True)

    # rename columns
    aggregated_events.columns = ['patient_id', 'feature_id', 'feature_value']

    aggregated_events['feature_value'] = np.round(aggregated_events.feature_value, 6)
    
    # create a column of tuples
    aggregated_events['feat'] = tuple(zip(aggregated_events.feature_id, aggregated_events.feature_value))
    aggregated_events = aggregated_events.sort_values(by='patient_id')

    # unique patient id's
    patients = aggregated_events.patient_id.unique()

    patient_features = {}

    for patient in patients:

        feats = aggregated_events[aggregated_events.patient_id == patient].feat.to_list()
        patient_features[patient] = feats
    
    test_file = '../deliverables/test_features.txt'
    
    deliverable1 = open(test_file, 'w')
    
    for k in list(patient_features.keys()):
        
        # patient id
        s = str(int(k)) 
        
        for feature, val in sorted(patient_features[k]):
            
            # ignore any 0 values
            if val != 0:
                s += ' ' + str(int(feature)) + ':' + "{:.6f}".format(float(val)) 
        
        s += ' \n' # add new line to end of each line
        deliverable1.write(s)
    deliverable1.close()
    
    X_test, _ = load_svmlight_file(test_file,n_features=3190)
    
    return X_train, Y_train, X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
    
    # stacking method for models
    estimators = [
    ('logreg_l2', LogisticRegression(C=0.5)),
    ('svc', SVC(C=0.804539775093711, kernel='sigmoid')),
    ('gb', GradientBoostingClassifier()),
    ('mlp', MLPClassifier(alpha=0.001466315789473684,
            early_stopping=True,
            hidden_layer_sizes=(200,)))]
    
    # define classifier
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    
    clf.fit(X_train, Y_train)
    
    return clf.predict(X_test)


def main():
    X_train, Y_train, X_test = my_features()
    Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
    utils.generate_submission("../deliverables/test_features.txt",Y_pred)
    #The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

    