library(argparse)
library(glmnet)
library(carrier)
library(reticulate)

reticulate::repl_python()
import numpy as np
import argparse
from azureml.core.run import Run
run=Run.get_context()
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=1.0, help='Kernel type to be used in the algorithm')
parser.add_argument('--rlambda', type=float,default=1.0, help='Penalty parameter of the error term')
args = parser.parse_args()
alpha = args.alpha
rlambda = args.rlambda
#mlflow.start_run()
exit

parser <- ArgumentParser(description='Input, and output filepaths')
#parser$add_argument('--input_file_path', help='The input file path')
parser$add_argument('--alpha', help='First hyperparameter')
parser$add_argument('--rlambda', help='Second hyperparameter')
args <- parser$parse_args()

#data <- read.csv(file.path(args$input_file_path,"wine_quality.csv"))
data <- read.csv("wine_quality.csv")
sdata <- sample(1:nrow(data),0.05 * nrow(data))
print(sdata)

# Split the data into training and test sets. (0.75, 0.25) split.
sampled <- sample(1:nrow(data), 0.75 * nrow(data))
train <- data[sampled, ]
test <- data[-sampled, ]

# The predicted column is "quality" which is a scalar from [3, 9]
train_x <- as.matrix(train[, !(names(train) == "quality")])
test_x <- as.matrix(test[, !(names(train) == "quality")])
train_y <- train[, "quality"]
test_y <- test[, "quality"]

alpha <- args$alpha
rlambda <- args$rlambda

reticulate::repl_python()
run.log('Alpha', r.alpha)
run.log('rLambda', r.rlambda)
print(f'Alpha is: {r.alpha}')
print(f'Lambda is: {r.rlambda}')
exit

model <- glmnet(train_x, train_y, alpha = alpha,lambda = rlambda, family= "gaussian", standardize = FALSE)
predictor <- crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model)
predicted <- predictor(test_x)

rmse <- sqrt(mean((predicted - test_y) ^ 2))
mae <- mean(abs(predicted - test_y))
r2 <- as.numeric(cor(predicted, test_y) ^ 2)

message("Elasticnet model (alpha=", alpha, ", lambda=", rlambda, "):")
message("  RMSE: ", rmse)
message("  MAE: ", mae)
message("  R2: ", r2)

reticulate::repl_python()
print(f'RMSE: {r.rmse}')
run.log('RMSE',r.rmse)
run.flush()
exit
