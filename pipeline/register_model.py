import click
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@click.command(help="This program registers a trained model .")
@click.option("--mlflow_run_id", default=None, help="This is the mlflow run id")
@click.option("--registered_model_name", default="ctgan_model", help="This is the registered model name.")
@click.option("--model_path", default="ctgan_model", help="This is the registered model path.")
@click.option("--pipeline_run_name", default="ctgan_test", help="This is the mlflow run name.")
def task(mlflow_run_id, model_path, registered_model_name, pipeline_run_name):
    if mlflow_run_id is None or mlflow_run_id == 'None':
        logger.info('no model to register. exit now.')
        return
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        logged_model = f'runs:/{mlflow_run_id}/{model_path}'
        logger.info("logged model uri is: %s", logged_model)
        mlflow.register_model(logged_model, registered_model_name)
        mlflow.set_tag('pipeline_step', __file__)

        # Load model as the pyfunc model
        model = mlflow.pyfunc.load_model(logged_model)
        # To bypass a lightning-flash's bug, we need to set the stage to test so a loaded model can be used to do prediction
        new_data = model.predict(11)

        print(new_data)


    logger.info("finished registering model to %s", registered_model_name)


if __name__ == '__main__':
    task()
