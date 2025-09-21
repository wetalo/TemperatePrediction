import mlflow
from darts.metrics import smape,mape,mse

class MLFlowControl:

    """
    Initialize the MLFlow object to submit logs to the MLFlow tables
    prediction: The weather predictions made by our model
    actual_values: The testing data we compared the prediction results against
    model: The trained model to make weather predictions
    config: A dictionary of modifiable variables to be referenced
    """
    def __init__(self,prediction, actual_values, model, config):
        self.config = config
        self.prediction = prediction.copy()
        self.actual_values = actual_values.copy()
        self.model = model

        """
        Compare the predicted values from the model to the actual test data values
        Calculates smape, mape and mse
        """
    def evaluate_score(self):

        self.smape_val = smape(self.actual_values,self.prediction)
        self.mape_val = mape(self.actual_values,self.prediction)
        self.mse_val = mse(self.actual_values,self.prediction)

        """
        Log the model and the evaluated score to the MLFlow database
        """
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
                