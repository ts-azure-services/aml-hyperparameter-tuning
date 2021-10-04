import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from authentication import ws
from azureml.core import Workspace, Dataset
from azureml.core.compute import ComputeTarget, ComputeInstance, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.environment import Environment
from azureml.core import Experiment, ScriptRunConfig
from azureml.core import Dataset
from azureml.data import OutputFileDatasetConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.parameter_expressions import choice

def target():
    cpu_cluster = ComputeTarget(workspace=ws, name='cpu-cluster')
    experiment_name = "iris-standard"
    exp = Experiment(workspace=ws, name=experiment_name)
    #env = Environment.get(workspace=ws, name='myenv')
    env = Environment.get(workspace=ws, name='AzureML-sklearn-0.24-ubuntu18.04-py37-cpu')

    src = ScriptRunConfig(source_directory=".",
                      script='train.py',
                      arguments=['--kernel', 'linear', '--penalty', 1.0],
                      compute_target=cpu_cluster,
                      environment=env)
    param_sampling = RandomParameterSampling( {
    "--kernel": choice('linear', 'rbf', 'poly', 'sigmoid'),
    "--penalty": choice(0.5, 1, 1.5, 2)
    })

    hyperdrive_config = HyperDriveConfig(run_config=src,
                                     hyperparameter_sampling=param_sampling,
                                     primary_metric_name='Accuracy',
                                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                     max_total_runs=12,
                                     max_concurrent_runs=4)

    # start the HyperDrive run
    hyperdrive_run = exp.submit(hyperdrive_config)
    hyperdrive_run.wait_for_completion(show_output=True)

    # Get best run
    best_run = hyperdrive_run.get_best_run_by_primary_metric()
    print(best_run.get_details()['runDefinition']['arguments'])
    print(best_run.get_file_names())
    
if __name__ == "__main__":
    target()
