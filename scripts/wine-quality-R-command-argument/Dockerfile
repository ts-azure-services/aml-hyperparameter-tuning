FROM continuumio/miniconda3:4.6.14
RUN apt-get update && yes Y | apt-get install build-essential
RUN conda install -c r r-essentials r-argparse r-carrier r-reticulate
#RUN conda install -c conda-forge r-mlflow
#RUN pip install azureml-mlflow
#RUN pip install mlflow
