library(tidyverse)
library(data.table)
library(h2o)
library(MLmetrics)
library(lubridate)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="57G")
train = h2o.importFile(path = 'data/csv_cut/data_train.csv')
valid = h2o.importFile(path = 'data/csv_cut/data_val.csv')
# set X and y 
y <- "TrueAnswer"
X = names(train)[c(3, 10:59, 63)]



auto = h2o.automl(
  training_frame=train, 
  x=X,
  y=y,
  max_runtime_secs = 129600,
  nfolds = 2,
  exclude_algos = c('GLM'),
  stopping_metric = 'MSE', 
  stopping_rounds = 10,
  stopping_tolerance = 0.001,
  seed = 123,
  sort_metric = 'MSE',
  project_name = 'baseline_autoML'
)





