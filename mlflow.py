# MLflow Tracking Setup
mlflow.set_tracking_uri("http://localhost:5000")
 
experiment_name = "Stock Prediction Experiment"
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
 
if experiment is None:
    mlflow.create_experiment(experiment_name)
 
mlflow.set_experiment(experiment_name)