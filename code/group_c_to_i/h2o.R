library(tidyverse)
library(data.table)
library(h2o)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="55G")
train = h2o.importFile(path = 'data/group_c_to_i/train.csv')
valid = h2o.importFile(path = 'data/group_c_to_i/valid.csv')

# log the label
train['TrueAnswer_log'] = log(train['TrueAnswer'])
valid['TrueAnswer_log'] = log(valid['TrueAnswer'])


# set X and y 
y_true = 'TrueAnswer'
y_log = 'TrueAnswer_log'
X = names(train)[c(3, 10:59, 63)]

##################################################################
######################## Random Forest ###########################
##################################################################

model_rf <- h2o.randomForest(
  model_id="model_rf", 
  training_frame=train, 
  validation_frame=valid,  
  y=y_true,
  x=X,
  ntrees = 130,
  max_depth = 17,
  min_rows = 1,
  stopping_rounds = 5,
  stopping_metric = 'MSE',
  stopping_tolerance = 0.001
  
  
)

# performance check 
summary(model_rf)
valid['y_pred_rf'] = h2o.predict(model_rf, valid)
metrics(valid['y_pred_rf'], valid[y_true])
# plot
valid_dt$y_pred_rf = valid['y_pred_rf']
valid_dt = as.data.table(valid)

plotPred(valid_dt, group = 'GroupG-46', model = 'rf', activity = FALSE)


# Tuning the parameters
hyper_params = list(
  ntrees = c(50,100,150,200,300),
  max_depth = c(10,15,20,25)
)

grid_rf <- h2o.grid(
  ## hyper parameters
  hyper_params = hyper_params,
  algorithm = "randomForest",
  ## identifier for the grid, to later retrieve it
  grid_id = "grid_rf",
  ## standard model parameters
  x = X,
  y = y_true,
  training_frame = train,
  validation_frame = valid,
  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
  stopping_rounds = 5, stopping_tolerance = 0.001, stopping_metric = "MSE"
)

##################################################################
########################### GBM ##################################
##################################################################

model_gbm <- h2o.gbm(model_id="model_gbm", 
                     training_frame=train, 
                     validation_frame=valid,
                     x = X, 
                     y = y_true,
                     ntrees = 200,
                     max_depth = 10,
                     stopping_rounds = 8,
                     stopping_metric = 'MSE',
                     stopping_tolerance = 0.001)
# performance check 
summary(model_gbm)

valid['y_pred_gbm'] = h2o.predict(model_gbm, valid)
metrics(valid['y_pred_gbm'], valid[y_true])

valid_dt$y_pred_gbm = as.data.table(valid$y_pred_gbm)
plotPred(valid_dt, group = 'GroupI-3000', model = 'gbm', activity = FALSE)

##################################################################
######################## XGBoost #################################
##################################################################

model_xgb <- h2o.xgboost(model_id="model_xgb", 
                         training_frame=train, 
                         validation_frame=valid,
                         x = X, 
                         y = y_log,
                         ntrees = 300,
                         max_depth = 10,
                         stopping_rounds = 5,
                         stopping_metric = 'MSE',
                         stopping_tolerance = 0.001,
                         verbose = TRUE)
# performance check 
summary(model_xgb)

valid['y_pred_xgb'] = exp(h2o.predict(model_xgb, valid))
metrics(valid['y_pred_xgb'], valid[y_true])

valid_dt$y_pred_xgb = as.data.table(valid$y_pred_xgb)
plotPred(valid_dt, group = 'GroupD-47', model = 'xgb', activity = FALSE)

################# Tuning the parameters #########################
hyper_params = list(
  ntrees = c(100,200,300,400,500),
  max_depth = seq(5,15,1),
  learn_rate = seq(0.01, 0.2, 0.01),
  sample_rate = seq(0.2,1,0.01),
  col_sample_rate = seq(0.2,1,0.01),
  col_sample_rate_per_tree = seq(0.2,1,0.01),
  min_rows = 2^seq(0,log2(nrow(train))-1,1),
  reg_lambda = seq(0,1,0.1),
  reg_alpha = seq(0,1,0.1)
)

search_criteria = list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 18000,
  max_models = 50,
  seed = 1234,
  stopping_rounds = 3,
  stopping_metric = "MSE",
  stopping_tolerance = 0.0001
)

grid_xgb <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train,
  validation_frame = valid,
  max_runtime_secs = 3600,
  stopping_rounds = 5, 
  stopping_tolerance = 0.0001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)

model_xgb = h2o.getModel('Grid_XGBoost_RTMP_sid_9c61_46_model_R_1565199110448_4_model_9')
summary(model_xgb)

valid['y_pred_xgb'] = exp(h2o.predict(model_xgb, valid))
metrics(valid['y_pred_xgb'], valid[y_true])

valid_dt$y_pred_xgb = as.data.table(valid$y_pred_xgb)
plotPred(valid_dt, group = 'GroupD-47', model = 'xgb', activity = FALSE)

##################################################################
######################## Deep Learning ###########################
##################################################################

model_deep <- h2o.deeplearning(
  model_id="model_deep", 
  training_frame=train, 
  validation_frame=valid,
  x=X,
  y=y_true,
  hidden=c(16,16,16),
  variable_importances=T,
  epochs=1000000,                      ## hopefully converges earlier...
  score_validation_samples=10000,      ## sample the validation dataset (faster)
  stopping_rounds=15,
  stopping_metric="MSE", ## could be "MSE","logloss","r2"
  stopping_tolerance=0.001,
  verbose = TRUE
)
# performance check 
summary(model_deep)

valid['y_pred_deep'] = h2o.predict(model_deep, valid)
metrics(valid['y_pred_deep'], valid[y_true])

valid_dt$y_pred_deep = as.data.table(valid$y_pred_deep)
plotPred(valid_dt, group = 'GroupI-74', model = 'deep', activity = FALSE)



#########################################################################
###################### Stacked Ensembles ################################
#########################################################################
nfolds = 3

rf = h2o.randomForest(
  training_frame=train, 
  nfolds = nfolds,
  y=y_true,
  x=X,
  ntrees = 150,
  max_depth = 18,
  min_rows = 1,
  stopping_rounds = 5,
  stopping_metric = 'MSE',
  stopping_tolerance = 0.001,
  keep_cross_validation_predictions = TRUE,
  seed = 1
)

valid['y_pred_rf'] = h2o.predict(rf, valid)
metrics(valid['y_pred_rf'], valid[y_true])

gbm <- h2o.gbm(
  training_frame=train, 
  nfolds = nfolds,
  x = X, 
  y = y_true,
  ntrees = 200,
  max_depth = 10,
  learn_rate = 0.1,
  stopping_rounds = 5,
  stopping_metric = 'MSE',
  stopping_tolerance = 0.001,
  keep_cross_validation_predictions = TRUE,
  seed = 1
  
)

valid['y_pred_gbm'] = h2o.predict(gbm, valid)
metrics(valid['y_pred_gbm'], valid[y_true])

xgb <- h2o.xgboost(
  training_frame=train, 
  nfolds = nfolds,
  x = X, 
  y = y_true,
  ntrees = 200,
  max_depth = 12,
  learn_rate = 0.1,
  stopping_rounds = 5,
  stopping_metric = 'MSE',
  stopping_tolerance = 0.001,
  keep_cross_validation_predictions = TRUE,
  seed = 1
  
)

valid['y_pred_xgb'] = h2o.predict(xgb, valid)
metrics(valid['y_pred_xgb'], valid[y_true])

deep <- h2o.deeplearning(
  training_frame=train, 
  nfolds = nfolds,
  x=X,
  y=y_true,
  hidden=c(64,64),
  variable_importances=T,
  epochs=300,                      ## hopefully converges earlier...
  stopping_rounds=10,
  stopping_metric="MSE", ## could be "MSE","logloss","r2"
  stopping_tolerance=0.001,
  keep_cross_validation_predictions = TRUE,
  seed = 1
)

valid['y_pred_deep'] = h2o.predict(deep, valid)
metrics(valid['y_pred_deep'], valid[y_true])




ensemble <- h2o.stackedEnsemble(
  x = X,
  y = y_true,
  training_frame = train,
  base_models = list(rf, gbm, xgb)
)

valid['y_pred_ensemble'] = h2o.predict(ensemble, valid)
metrics(valid['y_pred_ensemble'], valid[y_true])


#########################################################################
###################### Save and Load ####################################
#########################################################################
# Save the model
path <- h2o.saveModel(model_xgb, path="models_server/group_c_to_i", force=TRUE)

model <- h2o.loadModel('models_server/group_d_to_i/model_rf')
summary(model)



valid = h2o.importFile(path = 'data/group_i/valid.csv')

valid['y_pred_rf'] = h2o.predict(model, valid)
#test_h2o['y_pred'] = h2o.predict(model, test_h2o)

metrics(valid['y_pred_rf'], valid['TrueAnswer'])
#metrics(test_h2o['y_pred'], test_h2o['TrueAnswer'])


valid_dt = as.data.table(valid)
valid_dt_i = valid_dt










