from darts.models import ARIMA
from darts import TimeSeries
from darts.utils.model_selection import train_test_split

class Model:
    """
    Initializes Model Object

    Parameters
    -----------
    df: The dataframe featuring the data to be transformed
    config: A dictionary of modifiable variables to be referenced
    """
    def __init__(self,df, config):
        self.df = df.copy()
        self.config = config

        """
        Splits dataframe data into training and testing data
        """
    def train_test_split(self):
        dart_ts = TimeSeries.from_dataframe(df=self.df,time_col="date",value_cols="T")
        self.X_train,self.X_test = train_test_split(data=dart_ts,test_size=48)

        """
        Trains an ARIMA model using the data we provided
        """
    def train_model(self):
        self.model=ARIMA()
        self.model.fit(self.X_train)

    """
    Makes a weather prediction using the model we trained
    """
    def predict_weather(self):
        self.pred = self.model.predict(n=42)