from sdv.demo import load_tabular_demo
from sdv.tabular import CTGAN
import mlflow
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from sdv.evaluation import evaluate

client = MlflowClient()

import warnings
import os
import pdb
import pandas as pd
warnings.filterwarnings("ignore")

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

EXPERIMENT_NAME = "ctgan_experiment"
mlflow.set_tracking_uri('http://localhost')
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

MODEL_ARTIFACT_PATH = 'ctgan_model'
REGISTERED_MODEL_NAME = 'ctgan_model'
#mlflow.pytorch.autolog()

SDV_MODEL_NAME = "ctgan_model.pkl"

artifacts = {
    "ctgan_model": SDV_MODEL_NAME
}

class SDVWrapper(mlflow.pyfunc.PythonModel):
    """Defines the custom wrapper class."""
    def load_context(self, context):
        from sdv.tabular import CTGAN
        self.sdv_model = CTGAN.load(context.artifacts["ctgan_model"])

    def predict(self, context, model_input):
        return self.sdv_model.sample(num_rows=model_input)

data = load_tabular_demo('student_placements')

data = pd.read_csv(r'data/preprocessed_final.csv')

local_dir = "data/artifact"
if not os.path.exists(local_dir):
  os.mkdir(local_dir)


with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="ctgan") as dl_model_tracking_run:
    mlflow.log_param("x", 1)
    model = CTGAN()
    model.fit(data)
    model.save(SDV_MODEL_NAME)
    mlflow.pyfunc.log_model(python_model=SDVWrapper(),artifact_path=MODEL_ARTIFACT_PATH, artifacts=artifacts)
    new_data = model.sample(len(data))
    mlflow.log_metric("eval_result", evaluate(new_data, data))


run_id = dl_model_tracking_run.info.run_id
print("run_id: {}; lifecycle_stage: {}".format(run_id,
mlflow.get_run(run_id).info.lifecycle_stage))
logged_model = f'runs:/{run_id}/{MODEL_ARTIFACT_PATH}'

model_registry_version = mlflow.register_model(logged_model, REGISTERED_MODEL_NAME)
print(f'Model Name: {model_registry_version.name}')
print(f'Model Version: {model_registry_version.version}')

# Load model as the pyfunc model
model = mlflow.pyfunc.load_model(logged_model)
# To bypass a lightning-flash's bug, we need to set the stage to test so a loaded model can be used to do prediction
new_data = model.predict(11)

print(new_data)
