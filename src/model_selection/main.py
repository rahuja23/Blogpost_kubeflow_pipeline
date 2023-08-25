import mlflow
from dotenv import load_dotenv
import os
import argparse
def register_model(args):
    load_dotenv()
    os.getenv("AWS_ACCESS_KEY_ID")
    os.getenv("AWS_SECRET_ACCESS_KEY")
    os.getenv("MLFLOW_S3_ENDPOINT_URL")
    run_id = args["run_id"]
    print("****RUN ID*****", run_id)
    tracking_uri = args["tracking_uri"]
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_experiment_by_name("Test-Run").experiment_id
    current_run = client.get_run(run_id).data.to_dictionary()
    metrics_current_run = current_run["metrics"]
    print("METRICS_CURRENT_RUN:", metrics_current_run)
    best_metrics = max(metrics_current_run, key=metrics_current_run.get)
    print("BEST_METRICS:", best_metrics)
    last_registered_models = client.get_latest_versions(name="Test-Run")
    if len(last_registered_models)>0:
        latest_registered = last_registered_models[0]
    else:
        download_artifact = best_metrics.split("_")[0] + "_output"
        print("DOWNLOAD_ARTIFACT:", download_artifact)

        model_uri = "s3://mlflow/mlflow/{}/{}/artifacts/{}".format(experiment_id, run_id, download_artifact)
        print("MODEL_URI:", model_uri)
        name = "Test-Run"
        print("***************REGISTERING MODEL****************************************")
        mv = client.create_model_version(name, model_uri, run_id)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:5001")
    parser.add_argument("--run_id", type=str, default="")
    args = parser.parse_args()
    return args
if __name__ =="__main__":
    args = parse_args()
    args = vars(args)
    register_model(args)
