import kfp
import kfp.dsl as dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    InputPath,
    OutputPath,
)
from kfp import compiler
import subprocess
from kfp.aws import use_aws_secret
from typing import NamedTuple

@component(base_image="racahu23/blog_mlflow:1", packages_to_install=['mlflow'])
def mlflow_setup_experiment(tracking_uri:str)->NamedTuple('Outputs',[("exp_id", str), ("run_id", str)]):
    import main
    op = main.main(tracking_uri)
    return op
@component(base_image="racahu23/preprocess:blog",  packages_to_install=['mlflow', 'boto3'])
def twitter_download_preprocess(information: Output[Artifact],  experiment_id: str, run_id: str, tracking_uri: str):
    from main import twitter_sample_download_and_preprocess
    args={
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "log_folder": information.path
    }
    twitter_sample_download_and_preprocess(args)

@component(base_image="racahu23/numpy:blog_final",  packages_to_install=['mlflow', 'boto3'])
def numpy_process(information: Input[Artifact], information_output: Output[Artifact],  experiment_id: str, run_id: str,
                  tracking_uri: str):
    from main import numpy_process
    args = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "log_folder": information.path,
        "output_dir": information_output.path
           }
    op = numpy_process(args)
    return op
@component(base_image="racahu23/scikit:3", packages_to_install=['boto3'])
def sklearn_logistic(information_input: Input[Artifact], experiment_id: str,
                     tracking_uri: str, run_id: str,  sklearn_output: Output[Artifact]):
    from main import sklearn_logistic
    args = {
        "experiment_id": experiment_id,
        "tracking_uri": tracking_uri,
        "log_folder":  information_input.path,
        "output_dir": sklearn_output.path,
        "run_id": run_id
    }
    op = sklearn_logistic(args)
@component(base_image="racahu23/logistic:2",   packages_to_install=['mlflow', 'boto3'])
def logistic_op(sklearn_input:Input[Artifact], logistic_output: Output[Artifact],  experiment_id: str, run_id: str,
                tracking_uri: str):
    from main import logistic
    args = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "log_folder": sklearn_input.path,
        "output_dir": logistic_output.path
    }
    op = logistic(args)
@component(base_image="racahu23/torch:1",  packages_to_install=['mlflow', 'boto3'])
def torch_op(logistic_input:Input[Artifact], torch_output: Output[Artifact],
             experiment_id: str, run_id: str, tracking_uri: str
             ):
    from main import torch_process_logistic
    args = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "log_folder": logistic_input.path,
        "output_dir": torch_output.path
    }
    op = torch_process_logistic(args)

@component(base_image="racahu23/svm:1",  packages_to_install=['mlflow', 'boto3'])
def svm_op(svm_input: Input[Artifact], svm_output: Output[Artifact],
           experiment_id: str, run_id: str, tracking_uri: str
           ):
    from main import svm_process
    args = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "log_folder": svm_input.path,
        "output_dir": svm_output.path
    }
    op = svm_process(args)

@component(base_image="racahu23/register:4",  packages_to_install=['mlflow', 'python-dotenv'])
def register_op( run_id: str, tracking_uri: str):
    from main import register_model
    args = {
        "run_id": run_id,
        "tracking_uri": tracking_uri,
    }
    op = register_model(args)

if __name__ =="__main__":
    @dsl.pipeline(
        name='Twitter nltk pipeline',
        description='Writing code by the other way.'
    )
    def pipeline(mlflow_uri: str):
        pvc_name = "twitter-5000"
        """
        vop = dsl.VolumeOp(
            name=pvc_name,
            resource_name="twitter-5000",
            size="1Gi",
            modes=dsl.VOLUME_MODE_RWM
        )
        """
        op_mlflow = mlflow_setup_experiment(tracking_uri=mlflow_uri)
        download_task= twitter_download_preprocess(experiment_id=op_mlflow.outputs["exp_id"], run_id=op_mlflow.outputs["run_id"], tracking_uri=mlflow_uri)
        numpy_task = numpy_process(information=download_task.outputs['information'], experiment_id=op_mlflow.outputs["exp_id"], run_id=op_mlflow.outputs["run_id"]
                                   , tracking_uri=mlflow_uri).after(
            download_task)


        sklearn_task= sklearn_logistic(information_input=numpy_task.outputs["information_output"], experiment_id=op_mlflow.outputs["exp_id"], tracking_uri=mlflow_uri, run_id=op_mlflow.outputs["run_id"]).after(numpy_task)


        logistic_task= logistic_op(sklearn_input=numpy_task.outputs["information_output"], experiment_id=op_mlflow.outputs["exp_id"], run_id=op_mlflow.outputs["run_id"]
                                   , tracking_uri=mlflow_uri).after(sklearn_task)


        torch_task = torch_op(logistic_input=numpy_task.outputs["information_output"], experiment_id=op_mlflow.outputs["exp_id"], run_id=op_mlflow.outputs["run_id"]
                                   , tracking_uri=mlflow_uri ).after(logistic_task)

        svm_task = svm_op(svm_input=numpy_task.outputs["information_output"], experiment_id=op_mlflow.outputs["exp_id"], run_id=op_mlflow.outputs["run_id"]
                                   , tracking_uri=mlflow_uri).after(torch_task)
        register_task = register_op(tracking_uri=mlflow_uri, run_id=op_mlflow.outputs["run_id"]).after(svm_task)
    client = kfp.Client(namespace="kubeflow", host="http://localhost:8080")
    client.create_run_from_pipeline_func(pipeline,
                                         arguments={"mlflow_uri": "http://mlflow.use-case.svc.cluster.local:5000"},
                                         mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE,
                                         enable_caching=False)