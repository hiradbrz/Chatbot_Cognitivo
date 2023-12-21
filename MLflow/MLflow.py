import mlflow
from mlflow.pyfunc import PythonModel

class SimpleModel(PythonModel):
    def predict(self, context, model_input):
        # Example: Doubles the input value
        return model_input * 2

# Assuming you're running this in a Python environment where MLflow is installed
with mlflow.start_run():
    # Create an instance of your model
    model = SimpleModel()
    
    # Log the model to MLflow
    # The "simple_model" is the name of the model in MLflow
    mlflow.pyfunc.log_model("simple_model", python_model=model)


# # Replace with your actual run ID
# run_id = "a78c1ae8003f4291999a7000607e09a0"

# # The path to your model artifact relative to the MLflow run
# model_uri = f"runs:/{run_id}/simple_model"

# # Register the model in the MLflow Model Registry
# mlflow.register_model(model_uri, "SimpleModelRegistry")
