
from src.utils.input import csvLoader, yamlLoader
from src.transform import DataTransformation
from src.model import Model
from src.MLFlow import MLFlowControl

if __name__ == "__main__":
    config = yamlLoader().load_file("params.yaml")
    
    csv_path = config["data"]["raw_data"]

    csvLoader = csvLoader()
    df = csvLoader.load_file(csv_path)

    dataTransformObject = DataTransformation(df,config)
    dataTransformObject.transform_data_pipeline()

    modelTrainingObject = Model(dataTransformObject.df_final,config)
    modelTrainingObject.train_test_split()
    modelTrainingObject.train_model()
    modelTrainingObject.predict_weather()

    mlflowObject = MLFlowControl(prediction= modelTrainingObject.pred,
                                actual_values= modelTrainingObject.X_test,
                                model= modelTrainingObject.model,
                                 config= config)
    mlflowObject.evaluate_score()
    mlflowObject.log_mlflow()
