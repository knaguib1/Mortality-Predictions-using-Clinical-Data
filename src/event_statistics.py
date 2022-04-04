import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    # dead patients aggregation
    dead = events[events.patient_id.isin(mortality.patient_id.values)]
    dead = dead.groupby('patient_id').event_id.count()
    
    avg_dead_event_count = dead.mean()
    max_dead_event_count = dead.max()
    min_dead_event_count = dead.min()
    
    # alive patients aggregation
    alive = events[~events.patient_id.isin(mortality.patient_id.values)]
    alive = alive.groupby('patient_id').event_id.count()
    
    avg_alive_event_count = alive.mean()
    max_alive_event_count = alive.max()
    min_alive_event_count = alive.min()
    
    del [alive, dead]

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    # keep events containing DIAG, LAB, DRUG
    df_temp = events[events.event_id.str.contains('DIAG|LAB|DRUG')]
    
    # dead patient aggregation
    dead = df_temp[df_temp.patient_id.isin(mortality.patient_id.values)]
    dead = dead.groupby('patient_id').timestamp.nunique()
    
    avg_dead_encounter_count = dead.mean()
    max_dead_encounter_count = dead.max()
    min_dead_encounter_count = dead.min()
    
    # alive patient aggregation
    alive = df_temp[~df_temp.patient_id.isin(mortality.patient_id.values)]
    alive = alive.groupby('patient_id').timestamp.nunique()
    
    avg_alive_encounter_count = alive.mean()
    max_alive_encounter_count = alive.max()
    min_alive_encounter_count = alive.min()
    
    del [alive, dead]

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    # find min/max timestamp by patient id
    df_temp = events.groupby('patient_id').agg({'timestamp':['min','max']}).reset_index()
    
    # rename columns
    df_temp.columns = ['patient_id', 'min_ts', 'max_ts']
    
    # time delta between min/max timestamps in days
    df_temp['record_length'] = (pd.to_datetime(df_temp.max_ts) - pd.to_datetime(df_temp.min_ts)).dt.days
    
    # dead patients
    dead = df_temp[df_temp.patient_id.isin(mortality.patient_id.values)]['record_length']
    
    avg_dead_rec_len = dead.mean()
    max_dead_rec_len = dead.max()
    min_dead_rec_len = dead.min()
    
    # alive patients
    alive = df_temp[~df_temp.patient_id.isin(mortality.patient_id.values)]['record_length']
    
    avg_alive_rec_len = alive.mean()
    max_alive_rec_len = alive.max()
    min_alive_rec_len = alive.min()
    
    del[df_temp, dead, alive]

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
