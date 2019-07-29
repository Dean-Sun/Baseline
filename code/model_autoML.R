library(tidyverse)
library(data.table)
library(h2o)
library(MLmetrics)
library(lubridate)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="50G")
train = h2o.importFile(path = 'data/group_d_to_i/train.csv')
valid = h2o.importFile(path = 'data/group_d_to_i/valid.csv')
# set X and y 
y_true <- "TrueAnswer"
X = names(train)[c(3, 10:59, 63)]



auto = h2o.automl(
  training_frame=train, 
  x=X,
  y=y_true,
  max_runtime_secs = 54000,
  max_models = 15,
  nfolds = 2,
  exclude_algos = c('GLM'),
  stopping_metric = 'MAE', 
  stopping_rounds = 5,
  stopping_tolerance = 0.01,
  seed = 123,
  sort_metric = 'MAE',
  project_name = 'baseline_autoML'
)


# performance check 
summary(auto)
valid['y_pred_auto'] = h2o.predict(auto, valid)
metrics(valid['y_pred_auto'], valid[y_true])
# plot
valid_dt$y_pred_auto = valid['y_pred_auto']
valid_dt = as.data.table(valid)

plotPred(valid_dt, group = 'GroupG-46', model = 'auto', activity = FALSE)























