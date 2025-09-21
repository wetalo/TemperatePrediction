from darts.models import ARIMA
from darts import TimeSeries
from darts.utils.model_selection import train_test_split

class Model:

    def __init__(self,df, config):
        self.df = df.copy()
        self.config = config

    def train_test_split(self):
        dart_ts = TimeSeries.from_dataframe(df=self.df,time_col="date",value_cols="T")
        self.X_train,self.X_test = train_test_split(data=dart_ts,test_size=48)

    def train_model(self):
        self.model=ARIMA()
        self.model.fit(self.X_train)

    def predict_weather(self):
        self.pred = self.model.predict(n=42)
        print(self.pred)