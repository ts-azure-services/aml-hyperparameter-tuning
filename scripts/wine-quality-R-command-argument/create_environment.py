import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from authentication import ws
from azureml.core import Image

# Build custom environment
myenv = Environment.from_dockerfile(
        name = 'myenv',
        dockerfile ='./Dockerfile',
        conda_specification='./conda-specs.yaml'
        )
myenv.register(workspace=ws)
build = myenv.build(workspace=ws)
build.wait_for_completion(show_output=True)
