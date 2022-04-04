import utils
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    ### alive patients ###
    # subset data for alive patients
    alive = events[~events.patient_id.isin(mortality.patient_id.values)].copy()
    alive['timestamp'] = pd.to_datetime(alive.timestamp)
    
    # find max timestamp for each patient_id
    alive = alive.groupby('patient_id').timestamp.max().reset_index()

    ### dead patients ###
    dead = mortality[['patient_id', 'timestamp']].copy()
    dead['timestamp'] = pd.to_datetime(dead.timestamp) - pd.to_timedelta(30, unit='d')

    # stack dead/aliave dataframes
    indx_date = pd.concat([alive, dead])
    indx_date.rename(columns = {'timestamp':'indx_date'}, inplace=True)
    
    # write csv file
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    
    del [alive, dead]
    
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

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

    # keep columns of interest
    filtered_events = filtered_events[['patient_id', 'event_id', 'value']]
    
    # write to csv
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    
    return filtered_events


def aggregate_events(filtered_events, mortality,feature_map, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
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
    
    # write csv file
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', index=False)
    
    # delete temp dfs
    del [df_temp]
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    # order features
    aggregated_events = aggregated_events.sort_values(by=['patient_id', 'feature_id'])
    
    # create a column of tuples
    aggregated_events['feat'] = tuple(zip(aggregated_events.feature_id, aggregated_events.feature_value))
    
    # unique patient id's
    patients = aggregated_events.patient_id.unique()

    patient_features = {}

    for patient in patients:

        feats = aggregated_events[aggregated_events.patient_id == patient].feat.to_list()
        patient_features[patient] = feats
    
    mortality_df = pd.merge(aggregated_events['patient_id'].drop_duplicates(), 
                        mortality[['patient_id', 'label']], on='patient_id', how='left').fillna(0)

    mortality = {key:val for key, val in zip(mortality_df.patient_id, mortality_df.label)}

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    for k in list(patient_features.keys()):
        
        # patient ed
        sPatient = str(int(k)) 
        
        # mortality value
        sMortality = str(int(mortality[k]))
        
        s = str(int(mortality[k]))
        
        for feature, val in sorted(patient_features[k]):
            
            # ignore any 0 values
            if val != 0:
                s += ' ' + str(int(feature)) + ':' + "{:.6f}".format(float(val))
        
        s += ' \n' # add new line to end of each line
        deliverable1.write(bytes(s,'UTF-8')) #Use 'UTF-8'
        
        
        s = sPatient + ' ' + s
        deliverable2.write(bytes(s,'UTF-8'))
        
    deliverable1.close()
    deliverable2.close()
    
def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()