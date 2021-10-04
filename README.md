# Intent
- A repo to expose some examples of hyperparameter tuning in Azure Machine Learning. While hyperparameter
  tuning is fairly well-documented, when you start to leverage 'R' or more specifically, the 'command'
  argument in the ScriptRunConfig, there are some additional lines of code to work through to ensure
  parameters cycle through multiple combinations. The 'command' argument is mutually exclusive to the use of
  'script/arguments' argument.
- This repo has three examples of hyperparameter tuning, using Python and 'R':
	- ```iris-standard``` This is the cleanest, best-documented example with the iris dataset, using Python
	- ```iris-command-argument``` This uses the above dataset with Python, but with the ```command```
	  argument vs. the ```script, arguments``` parameters.
	- ```wine-quality-R-command-argument```  This uses the command argument with 'R' as the base for training
- If on a Mac, the following packages are helpful to install in your conda environment:
	- ```python-dotenv```
	- ```azureml-defaults```
	- ```azureml-train```


# File Structure
- LICENSE.TXT
- README.md
- requirements.txt
- scripts
	- "Setup scripts" (not a file structure)
		- create-workspace-sprbac.sh ```shell script embedded in 'workflow.sh' to create workspace/infra```
		- clusters.py ```creates a cluster; part of 'workflow.sh' process```
		- Authentication and environment variables
			- authentication.py ```Used to authenticate the workspace with a service principal```
			- config.json ```gets created during workflow```
			- sub.env ```subscription info: needs to be in place prior to execution```
				- Manually create a file called ```sub.env``` with one line: ```SUB_ID="<enter subscription id>"```
			- variables.env ```gets created during workflow```
	- iris-standard
		- target.py ```entry script to initiate hyperparameter training```
		- train.py ```training script in Python```
	- iris-command-argument
		- target.py ```entry script to initiate hyperparameter training```
		- train.py ```training script in Python```
	- wine-quality-R-command-argument
		- Dockerfile ```Docker file specifications for R environment```
		- conda-specs.yaml ```conda specifications for R environment```
		- create_environment.py ```create custom environment to run 'R' scripts```
			- Run this file before running the ```target.py``` script
		- target.py ```entry script to initiate hyperparameter training```
		- train.R ```training script in R```
		- wine_quality.csv ```input data```
	- name-generator
		- adjectives.txt ```used as input into random_name.py```
		- nouns.txt ```used as input into random_name.py```
		- random_name.py ```uses adjectives.txt. and nouns.txt to create a random name```
