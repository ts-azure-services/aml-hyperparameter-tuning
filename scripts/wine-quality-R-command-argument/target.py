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
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.parameter_expressions import choice

def execute_for_wine(
        exp_name=None,
        env_name=None,
        cluster_name=None, 
        source_directory=None,
        R_script=None,
        alpha=None,
        rlambda=None
        ):
    """Run the experiment from a registered dataset
    to an output that gets stored in blob store
    """
    experiment = Experiment(ws, name=exp_name)
    cluster = ws.compute_targets[cluster_name]
    env = Environment.get(workspace=ws, name=env_name)
    config = ScriptRunConfig(
            source_directory= source_directory,
            command=['Rscript',R_script,'--alpha','$AZUREML_SWEEP_alpha','--rlambda','$AZUREML_SWEEP_rlambda'],
            compute_target=cluster,
            environment=env
            )

    param_sampling = RandomParameterSampling({"alpha": choice(0.5,0.6,0.7,0.8),"rlambda": choice(0.5, 1, 1.5, 2)})

    hyperdrive_config = HyperDriveConfig(run_config=config,
                                     hyperparameter_sampling=param_sampling,
                                     primary_metric_name='RMSE',
                                     primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
                                     max_total_runs=12,
                                     max_concurrent_runs=6)

    # start the HyperDrive run
    hyperdrive_run = experiment.submit(hyperdrive_config)
    hyperdrive_run.wait_for_completion(show_output=True)

    # Get best run
    best_run = hyperdrive_run.get_best_run_by_primary_metric()
    print(best_run.get_details()['runDefinition']['arguments'])
    print(best_run.get_file_names())

def main():
    """Main operational flow"""

    # Get input data files, specify default data store, and upload files
    def_blob_store = ws.get_default_datastore()

    execute_for_wine(
        exp_name='wine-quality-R-command-argument',
        env_name='myenv',
        cluster_name='cpu-cluster', 
        source_directory='.',
        R_script='train.R',
        alpha=0.5,
        rlambda=0.5
        )

if __name__ == "__main__":
    main()
