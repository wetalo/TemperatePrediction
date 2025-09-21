import pandas as pd
class DataTransformation:
    """
    Initializes DataTransformation Object

    Parameters
    -----------
    df: The dataframe featuring the data to be transformed
    config: A dictionary of modifiable variables to be referenced
    """
    def __init__(self,df, config):
        self.config = config
        self.df = df.copy()

    

    """
    Converts the "date" column in our dataframe from a string to a proper datetime
    """
    def change_date_to_datetime(self):
        self.df["date"] = pd.to_datetime(self.df["date"],format="%Y-%m-%d %H:%M:%S")

    
    """
    Only uses the most important features in our model training
    """
    def take_most_important_features(self):
        self.df = self.df[["date","p","T","Tpot"]]


    """
    Reduces the size of our data by 2/3 by taking a screenshot every 30 minutes instead of every 10
    """
    def reduce_data_by_timestamp(self):
        #Take timestamps only every 30 minutes instead of every ten minutes, reducing database size and calculations to only 1/3
        self.df["time_30min"] = self.df['date'].dt.floor('30min')
        self.df_grouped = self.df.groupby('time_30min').mean().reset_index()
        self.df_grouped_feature = self.df_grouped[["p","T"]].copy()

    
    """
    Establishes a date-range timeseries in order to be properly read by DARTS
    """
    def establish_date_range_timeseries(self):
        date_range = pd.date_range(start = '2020-01-01 00:00:00', end = '2020-12-31 23:00:00', freq = '30min')

        #Dataframe timeseries
        self.df_ts = pd.DataFrame(date_range)
        self.df_ts.columns = ['date']


    """
    Combines the data we've worked with along with the date-range time-series in order to be properly read by DARTS
    """
    def concatenate_dataframes(self):
        self.df_final = pd.concat([self.df_ts,self.df_grouped_feature],axis=1)

    def transform_data_pipeline(self):
        self.change_date_to_datetime()
        self.take_most_important_features()      
        self.reduce_data_by_timestamp()  
        self.establish_date_range_timeseries()
        self.concatenate_dataframes()