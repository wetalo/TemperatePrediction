import mlflow
from darts.metrics import smape,mape,mse

class MLFlowControl:

    def __init__(self,prediction, actual_values, model, config):
        self.config = config
        self.prediction = prediction.copy()
        self.actual_values = actual_values.copy()
        self.model = model

    def evaluate_score(self):

        self.smape_val = smape(self.actual_values,self.prediction)
        self.mape_val = mape(self.actual_values,self.prediction)
        self.mse_val = mse(self.actual_values,self.prediction)

    def log_mlflow(self):
            with mlflow.start_run():
                
                mlflow.set_tag("Coder","DS")
                mlflow.log_metric("smape temperature prediction", self.smape_val)
                mlflow.log_metric("mape temperature prediction", self.mape_val)
                mlflow.log_metric("mse temperature prediction", self.mse_val)

                mlflow.statsmodels.log_model(
                    statsmodels_model=self.model,
                    artifact_path="arima_model",
                    registered_model_name="ARIMA-Time-Series-Model",
                )
                