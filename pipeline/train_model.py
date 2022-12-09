import logging
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import click
import mlflow
import mlflow.pyfunc
from sdv.tabular import CTGAN
from mlflow.tracking.client import MlflowClient
from sdv.evaluation import evaluate

class SDVWrapper(mlflow.pyfunc.PythonModel):
    """Defines the custom wrapper class."""
    def load_context(self, context):
        from sdv.tabular import CTGAN
        self.sdv_model = CTGAN.load(context.artifacts["ctgan_model"])

    def predict(self, context, model_input):
        return self.sdv_model.sample(num_rows=model_input)


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

MODEL_ARTIFACT_PATH = 'ctgan_model'
REGISTERED_MODEL_NAME = 'ctgan_model'
#mlflow.pytorch.autolog()

CTGAN_MODEL_NAME = "ctgan_model.pkl"

artifacts = {
    "ctgan_model": CTGAN_MODEL_NAME
}

@click.command(help="This program finetunes a deep learning model for sentimental classification.")
@click.option("--data_path", default="./data/preprocessed_final.csv", help="This is the path to data.")
@click.option("--epochs", default=500, help="This is the epochs to train the model")
@click.option("--batch_size", default=100, help="This is the batch size to train the model")
@click.option("--pipeline_run_name", default="ctgan_test", help="This is the mlflow run name.")
def task(data_path, epochs, batch_size, pipeline_run_name):

    with mlflow.start_run(run_name=pipeline_run_name) as ctgan_model_tracking_run:
        #import pdb
        #pdb.set_trace()
        data = pd.read_csv(data_path)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_path", MODEL_ARTIFACT_PATH)
        model = CTGAN(epochs=epochs, batch_size=batch_size)
        #import pdb
        #pdb.set_trace()
        model.fit(data)
        model.save(CTGAN_MODEL_NAME)
        mlflow.pyfunc.log_model(python_model=SDVWrapper(),artifact_path=MODEL_ARTIFACT_PATH, artifacts=artifacts)
        
        new_data = model.sample(len(data))
        print(new_data)
        mlflow.log_metric("eval_result", evaluate(new_data, data))
        # mlflow log additional hyper-parameters used in this training

        run_id = ctgan_model_tracking_run.info.run_id
        logger.info("run_id: {}; lifecycle_stage: {}".format(run_id,
                                                             mlflow.get_run(run_id).info.lifecycle_stage))
        mlflow.log_param("mlflow_run_id", run_id)
        mlflow.set_tag('pipeline_step', __file__)


if __name__ == '__main__':
    task()
