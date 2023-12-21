class DialogModelWrapper(DialogModel, PythonModel):
    def predict(self, context: PythonModelContext, model_input):
        # Assuming model_input is a string representing user input
        return super().generate_response_dia(model_input)

# Log the model in MLflow
with mlflow.start_run():
    dialog_model = DialogModelWrapper()
    mlflow.pyfunc.log_model("dialog_model", python_model=dialog_model)

# Register the model in MLflow Model Registry
# Replace 'your_run_id_here' with the actual run ID from MLflow
run_id = "your_run_id_here"
mlflow.register_model(f"runs:/{run_id}/dialog_model", "DialogModel_Registry")
