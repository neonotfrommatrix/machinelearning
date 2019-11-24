#Step 1. transformation.py is processing dataset (preparing for machine learning)
#Library to manipulate dataframe 
import pandas as pd

#Function needed to normalize data
def normalize_column(dataframe, column_name):
    dataframe[column_name] = (dataframe[column_name] - dataframe[column_name].min())/(dataframe[column_name].max()- dataframe[column_name].min())
    return dataframe

#Function needed to standardize data
def std_column(dataframe, column_name):
    dataframe[column_name] = (dataframe[column_name] - dataframe[column_name].mean())/dataframe[column_name].std(ddof=0)
    return dataframe

#Function that creates new columns for the key and sharp
def categorize_key(dataframe):
    key_mapping = {
        'A' : 0,
        'B' : 1,
        'C' : 2,
        'D' : 3,
        'E' : 4,
        'F' : 5,
        'G' : 6
    }
    key_col_vals = [key_mapping[key[0]] for key in dataframe.key]
    sharp_col_vals = [1 if '#' in key  else 0 for key in dataframe.key]
    dataframe = dataframe.drop(columns = ['key'])
    dataframe['key'] = key_col_vals
    dataframe['sharp'] = sharp_col_vals
    return dataframe

#Function that ordinalizes time signature
def categorize_time_sig(dataframe):
    time_sig_mapping = {
        '0/4' : 0,
        '1/4' : 0.25,
        '3/4' : 0.75,
        '4/4' : 1,
        '5/4' : 1.25
    }
    time_sig_col_vals =  [time_sig_mapping[time_sig] for time_sig in dataframe.time_signature]
    dataframe.time_signature = time_sig_col_vals
    return dataframe

#Function that categorizes mode in binary
def categorize_mode(dataframe):
    mode_col_vals = [0 if mode == 'Minor' else 1 for mode in dataframe['mode']]
    dataframe['mode'] = mode_col_vals
    return dataframe

#Read in data
data = pd.read_csv('SpotifyFeatures.csv')[:]

#Drop useless data
dropped_columns = ['genre', 'artist_name', 'track_name', 'track_id']

#Useful data contains everything but dropped columns
useful_data = data.drop(columns=dropped_columns)


normalized_data = normalize_column(useful_data, 'loudness')
print(normalized_data)

std_data = std_column(normalized_data, 'tempo')
std_data = std_column(std_data, 'duration_ms')
print(std_data)

post_key_data = categorize_key(std_data)
print(post_key_data)

post_time_sig_data = categorize_time_sig(post_key_data)
print(post_key_data)

post_mode_data = categorize_mode(post_time_sig_data)
print(post_mode_data)

post_mode_data.to_csv('transformed_data.csv')