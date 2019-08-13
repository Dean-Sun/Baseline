library(tidyverse)
library(data.table)
library(h2o)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="10G")
train = h2o.importFile(path = 'data/group_c/train.csv')
valid = h2o.importFile(path = 'data/group_c/valid.csv')

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

model_rf_log <- h2o.randomForest(
  training_frame=train, 
  validation_frame=valid,  
  y=y_log,
  x=X,
  ntrees = 200,
  max_depth = 20,
  min_rows = 1,
  stopping_rounds = 5,
  stopping_metric = 'MSE',
  stopping_tolerance = 0.0001
  
  
)

# performance check 
summary(model_rf_log)
valid['y_pred_rf'] = exp(h2o.predict(model_rf_log, valid))
metrics(valid['y_pred_rf'], valid[y_true])
# plot
valid_dt = as.data.table(valid)
valid_dt$y_pred_rf = as.data.table(valid$y_pred_rf)
plotPred(valid_dt, group = 'GroupA-818', model = 'xgb', activity = FALSE)



##################################################################
########################### GBM ##################################
##################################################################

model_gbm <- h2o.gbm(training_frame=train, 
                     model_id = 'model_gbm_log',
                     validation_frame=valid,
                     x = X, 
                     y = y_log,
                     ntrees = 50,
                     max_depth = 20,
                     stopping_rounds = 5,
                     stopping_metric = 'MSE',
                     stopping_tolerance = 0.001)
# performance check 
summary(model_gbm)

valid['y_pred_gbm'] = exp(h2o.predict(model_gbm, valid))
metrics(valid['y_pred_gbm'], valid[y_true])

valid_dt$y_pred_gbm = as.data.table(valid$y_pred_gbm)
plotPred(valid_dt, group = 'GroupA-804', model = 'gbm', activity = FALSE)


path <- "/home/dsun/Baseline/models_server/mojo/test"
mojo_destination <- h2o.download_mojo(model = model_gbm, path = path)
imported_model <- h2o.import_mojo('/home/dsun/Baseline/models_server/mojo/test/model_gbm_log.zip')




##################################################################
######################## XGBoost #################################
##################################################################

model_xgb <- h2o.xgboost(training_frame=train, 
                         validation_frame=valid,
                         x = X, 
                         y = y_true,
                         ntrees = 400,
                         max_depth = 11,
                         learn_rate = 0.07,
                         col_sample_rate = 0.71,
                         col_sample_rate_per_tree = 0.74,
                         min_rows = 512,
                         reg_alpha = 0.5,
                         reg_lambda = 0.8,
                         sample_rate = 0.22,
                         stopping_rounds = 5,
                         stopping_metric = 'MSE',
                         stopping_tolerance = 0.0001,
                         verbose = TRUE)
# performance check 
summary(model_xgb)

model = h2o.getModel(grid_xgb@model_ids[[1]])

valid['y_pred_xgb'] = exp(h2o.predict(model, valid))
metrics(valid['y_pred_xgb'], valid[y_true])

valid_dt$y_pred_xgb = as.data.table(valid$y_pred_xgb)
plotPred(valid_dt, group = 'GroupA-813', model = 'xgb', activity = TRUE)


################# Tuning the parameters #########################
hyper_params = list(
  ntrees = c(200,300,400),
  max_depth = seq(8,13,1),
  learn_rate = seq(0.01, 0.2, 0.01),
  sample_rate = seq(0.2,1,0.01),
  col_sample_rate = seq(0.2,1,0.01),
  col_sample_rate_per_tree = seq(0.2,1,0.01),
  min_rows = seq(0,500,50),
  reg_lambda = seq(0,1,0.1),
  reg_alpha = seq(0,1,0.1)
)


search_criteria = list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 3600,
  max_models = 50,
  seed = 1234,
  stopping_rounds = 3,
  stopping_metric = "MSE",
  stopping_tolerance = 0.001
)

grid_xgb <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train,
  validation_frame = valid,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)



##################################################################
######################## Deep Learning ###########################
##################################################################

model_deep <- h2o.deeplearning(
  model_id = 'model_deep',
  training_frame=train, 
  validation_frame = valid,
  x=X,
  y=y_true,
  hidden=c(32,32),
  variable_importances=T,
  epochs=1000000,                      ## hopefully converges earlier...
  score_validation_samples=10000,      ## sample the validation dataset (faster)
  stopping_rounds=15,
  stopping_metric="MSE", ## could be "MSE","logloss","r2"
  stopping_tolerance=0.0001,
  verbose = FALSE
)


# performance check 
summary(model_deep)

valid['y_pred_deep'] = ifelse((h2o.predict(model_deep, valid))>0,
                              (h2o.predict(model_deep, valid)),
                              0)


metrics(valid['y_pred_deep'], valid[y_true])

valid_dt$y_pred_deep = as.data.table(valid$y_pred_deep)
plotPred(valid_dt, group = 'GroupA-817', model = 'deep', activity = FALSE)


################# Tuning the parameters #########################
hyper_params = list(
  activation=c("Rectifier"),
  hidden=list(c(20,20),c(50,50), c(64,64), c(8,8,8), c(16,16,16), c(30,30,30),c(25,25,25,25)),
  l1=seq(0,0.8,0.01),
  l2=seq(0,0.8,0.01)
)

search_criteria = list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 3600,
  max_models = 50,
  seed = 1234,
  stopping_rounds = 10,
  stopping_metric = "MSE",
  stopping_tolerance = 0.0001
)

grid_deep <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "deeplearning",
  x = X,
  y = y_true,
  training_frame = train,
  validation_frame = valid,
  max_runtime_secs = 1800,
  stopping_rounds = 10, 
  stopping_tolerance = 0.0001, 
  stopping_metric = "MSE",
  seed = 1234
)





#####################################################################
########################### Ensemble ################################
#####################################################################

#### Average ####
a  = (valid_dt$y_pred_xgb+valid_dt$y_pred_rf+valid_dt$y_pred_deep+valid_dt$y_pred_gbm)/4
valid_dt$y_pred_avg = a
metrics(valid_dt$y_pred_avg, valid_dt$TrueAnswer)




#### Stack #####
train['y_pred_rf'] = exp(h2o.predict(model_rf_log, train))
train['y_pred_gbm'] = exp(h2o.predict(model_gbm, train))
train['y_pred_xgb'] = (h2o.predict(model_xgb, train))
train['y_pred_deep'] = (h2o.predict(model_deep, train))

# check 
metrics(train['y_pred_rf'], train[y_true])


# model 
X_ensemble = names(train)[67:70]

model_glm_ensemble = h2o.glm(
  model_id="model_glm_ensemble", 
  training_frame=train, 
  validation_frame=valid,  
  y=y_true,
  x=X_ensemble,
  family = 'gaussian'
)

# performance check 
summary(model_glm_ensemble)

valid['y_pred_stack'] = (h2o.predict(model_glm_ensemble, valid))
metrics(valid['y_pred_stack'], valid[y_true])



#########################################################################
###################### Save and Load ####################################
#########################################################################
# Save the model
path <- h2o.saveModel(model, path="models_server/separate_models/group_c", force=TRUE)

model <- h2o.import_mojo('/home/dsun/Baseline/models_server/mojo/DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip')
summary(model)


pred = h2o.mojo_predict_csv(
  input_csv_path = 'data/group_a/valid.csv',
  mojo_zip_path = 'models_server/mojo/DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip',
  verbose = T
)



valid['y_pred'] = as.h2o(pred)
metrics(valid['y_pred'], valid['TrueAnswer'])


valid_dt = as.data.table(valid)
names(valid_dt)[68] = 'y_pred_mike'
plotPred(valid_dt, group = 'GroupA-841', model = 'mike', activity = FALSE)








model <- h2o.loadModel('models_server/group_a/model_xgb')
summary(model)





valid['y_pred_xgb'] = exp(h2o.predict(model, valid))
metrics(valid['y_pred_xgb'], valid['TrueAnswer'])


valid_dt = as.data.table(valid)
plotPred(valid_dt, group = 'GroupA-926', model = 'xgb', activity = FALSE)








valid_dt_a = as.data.table(valid)












