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
  max_runtime_secs = 86400,
  leaderboard_frame = valid,
  max_models = 100,
  nfolds = 3,
  exclude_algos = c('GLM','DRF'),
  stopping_metric = 'MSE', 
  stopping_rounds = 5,
  stopping_tolerance = 0.001,
  seed = 123,
  sort_metric = 'MSE',
  project_name = 'baseline_autoML'
)

auto_1 = auto@leader
auto_2 = auto@leaderboard$model_id[2]
auto_3 = auto@leaderboard$model_id[3]
auto_4 = auto@leaderboard$model_id[4]
auto_5 = auto@leaderboard$model_id[5]

h2o.saveModel(auto_1, path="models_server", force=TRUE)
h2o.saveModel(auto_2, path="models_server", force=TRUE)
h2o.saveModel(auto_3, path="models_server", force=TRUE)
h2o.saveModel(auto_4, path="models_server", force=TRUE)
h2o.saveModel(auto_5, path="models_server", force=TRUE)





# performance check 
summary(auto)
valid['y_pred_auto'] = h2o.predict(auto, valid)
metrics(valid['y_pred_auto'], valid[y_true])
# plot
valid_dt$y_pred_auto = valid['y_pred_auto']
valid_dt = as.data.table(valid)

plotPred(valid_dt, group = 'GroupG-46', model = 'auto', activity = FALSE)























