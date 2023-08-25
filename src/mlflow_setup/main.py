import mlflow
from typing import NamedTuple
import argparse
from collections import namedtuple
def main(mlflow_uri: str) -> NamedTuple("Outputs", [("exp_id", str), ("run_id", str)]):
    print ("TRACKING URI", mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    try:
        experiment_id = client.create_experiment("Test-Run")
    except:
        experiment_id = client.get_experiment_by_name("Test-Run").experiment_id
    print("Experiment ID", experiment_id)
    run_id = client.create_run(experiment_id).info.run_id
    print("Run ID", run_id)
    output = namedtuple("Outputs", ["exp_id", "run_id"])
    return output(experiment_id, run_id)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:5000")
    args = parser.parse_args()
    main(args.mlflow_uri)