library(tidyverse)
library(data.table)
library(h2o)
library(MLmetrics)
library(lubridate)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="56G")
train = h2o.importFile(path = 'data/csv_cut/data_train.csv')
valid = h2o.importFile(path = 'data/csv_cut/data_val.csv')

#####################################
### try for delete some outliers ####
#####################################

# set X and y 
y <- "TrueAnswer"
X = names(train)[c(3, 10:59, 63)]



##################################################################
######################## Random Forest ###########################
##################################################################

model_rf <- h2o.randomForest(
  model_id="model_rf", 
  training_frame=train, 
  validation_frame=valid,  
  y=y,
  x=X,
  ntrees = 150,
  max_depth = 25,
  min_rows = 1,
  stopping_rounds = 5,
  stopping_metric = 'MSE',
  stopping_tolerance = 0.01
  
  
)

# performance check 
summary(model_rf)
h2o.varimp(model_rf)
h2o.performance(model_rf, newdata=train)    ## full training data
h2o.performance(model_rf, newdata=valid)    ## full validation data

metrics(h2o.predict(model_rf, train), train[y])
valid['y_pred_rf'] = h2o.predict(model_rf, valid)
metrics(valid['y_pred_rf'], valid[y])

# Plot
#valid_dt$y_pred_rf = as.data.table(valid$y_pred_rf)
valid_dt = as.data.table(valid)
plotPred(valid_dt, group = 'GroupF-196', model = 'rf', activity = TRUE)


# Tuning the parameters
hyper_params = list(
  ntrees = c(50,100,150,200,300),
  max_depth = c(7,10,15,17,20,23,26)
)

grid_rf <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  algorithm = "randomForest",
  ## identifier for the grid, to later retrieve it
  grid_id = "grid_rf",
  ## standard model parameters
  x = X,
  y = y,
  training_frame = train,
  validation_frame = valid,
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 0.01, stopping_metric = "MSE"
)





##################################################################
########################### GBM ##################################
##################################################################

model_gbm <- h2o.gbm(model_id="model_gbm", 
                     training_frame=train, 
                     validation_frame=valid,
                     x = X, 
                     y = y,
                     ntrees = 100,
                     max_depth = 10)
# performance check 
summary(model_gbm)

metrics(h2o.predict(model_gbm, train), train[y])
valid['y_pred_gbm'] = h2o.predict(model_gbm, valid)
metrics(valid['y_pred_gbm'], valid[y])

valid_dt$y_pred_gbm = as.data.table(valid$y_pred_gbm)
plotPred(valid_dt, group = 'Group-199', model = 'gbm', activity = FALSE)


# Tuning the parameters
hyper_params = list(
  max_depth = seq(7,20,1),
  ## search a large space of row sampling rates per tree
  sample_rate = seq(0.2,1,0.01),
  ## search a large space of column sampling rates per split
  col_sample_rate = seq(0.2,1,0.01),
  ## search a large space of column sampling rates per tree
  col_sample_rate_per_tree = seq(0.2,1,0.01),
  ## search a large space of how column sampling per split should change as a function of the depth of the split
  col_sample_rate_change_per_level = seq(0.9,1.1,0.01),
  ## search a large space of the number of min rows in a terminal node
  min_rows = 2^seq(0,log2(nrow(train))-1,1),
  ## search a large space of the number of bins for split-finding for continuous and integer columns
  nbins = 2^seq(4,10,1),
  ## search a large space of the number of bins for split-finding for categorical columns
  nbins_cats = 2^seq(4,12,1),
  ## search a few minimum required relative error improvement thresholds for a split to happen
  min_split_improvement = c(0,1e-8,1e-6,1e-4),
  ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
  histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")
)


search_criteria = list(
  ## Random grid search
  strategy = "RandomDiscrete",
  ## limit the runtime to 60 minutes
  max_runtime_secs = 54000,
  ## build no more than 200 models
  max_models = 200,
  ## random number generator seed to make sampling of parameter combinations reproducible
  seed = 1234,
  ## early stopping once the leaderboard of the top 5 models is converged to 0.1% relative difference
  stopping_rounds = 3,
  stopping_metric = "MSE",
  stopping_tolerance = 0.05
)

grid_gbm <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  ## hyper-parameter search configuration (see above)
  search_criteria = search_criteria,
  ## which algorithm to run
  algorithm = "gbm",
  ## identifier for the grid, to later retrieve it
  grid_id = "grid_gbm",
  ## standard model parameters
  x = X,
  y = y,
  training_frame = train,
  validation_frame = valid,
  ## more trees is better if the learning rate is small enough
  ## use "more than enough" trees - we have early stopping
  ntrees = 10000,
  ## smaller learning rate is better
  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
  learn_rate = 0.1,
  ## learning rate annealing: learning_rate shrinks by 1% after every tree
  ## (use 1.00 to disable, but then lower the learning_rate)
  learn_rate_annealing = 0.99,
  ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
  max_runtime_secs = 3600,
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 3, stopping_tolerance = 0.05, stopping_metric = "MSE",
  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
  score_tree_interval = 10,
  score_validation_samples=10000,
  ## base random number generator seed for each model (automatically gets incremented internally for each model)
  seed = 1234
)


## Sort the grid models by AUC
grid_gbm <- h2o.getGrid("final_grid", sort_by = "auc", decreasing = TRUE)
grid_gbm







##################################################################
####################### XGBoost ##################################
##################################################################

model_xgb <- h2o.xgboost(model_id="model_xgb", 
                    training_frame=train, 
                    validation_frame=valid,
                    x = X, 
                    y = y,
                    ntrees = 50,
                    max_depth = 7,
                    stopping_rounds = 3,
                    stopping_metric = 'MSE',
                    stopping_tolerance = 0.01,
                    verbose = TRUE)
# performance check 
summary(model_xgb)

metrics(h2o.predict(model_xgb, train), train[y])
valid['y_pred_xgb'] = h2o.predict(model_xgb, valid)
metrics(valid['y_pred_xgb'], valid[y])

valid_dt$y_pred_xgb = as.data.table(valid$y_pred_xgb)
plotPred(valid_dt, group = 'Group-199', model = 'xgb', activity = FALSE)




































































# Save the model
path <- h2o.saveModel(model_rf, 
                      path="models", force=TRUE)
model_rf <- h2o.loadModel('models/model_rf')
summary(model)




























































































