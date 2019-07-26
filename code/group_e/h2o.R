library(tidyverse)
library(data.table)
library(h2o)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="58G")
train = h2o.importFile(path = 'data/group_e/train.csv')
valid = h2o.importFile(path = 'data/group_e/valid.csv')

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
  ntrees = 150,
  max_depth = 20,
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
valid_dt = as.data.table(valid)
plotPred(valid_dt, group = 'GroupE-4446', model = 'rf', activity = FALSE)


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
                     ntrees = 150,
                     max_depth = 13,
                     stopping_rounds = 5,
                     stopping_metric = 'MSE',
                     stopping_tolerance = 0.001)
# performance check 
summary(model_gbm)

valid['y_pred_gbm'] = h2o.predict(model_gbm, valid)
metrics(valid['y_pred_gbm'], valid[y_true])

valid_dt$y_pred_gbm = as.data.table(valid$y_pred_gbm)
plotPred(valid_dt, group = 'GroupI-3000', model = 'gbm', activity = FALSE)


##################################################################
######################## Deep Learning ###########################
##################################################################

model_deep <- h2o.deeplearning(
  model_id="model_deep", 
  training_frame=train, 
  validation_frame=valid,
  x=X,
  y=y_true,
  hidden=c(64,64,64),
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
plotPred(valid_dt, group = 'GroupI-3000', model = 'deep', activity = FALSE)







#########################################################################
###################### Save and Load ####################################
#########################################################################
# Save the model
path <- h2o.saveModel(model_deep, path="models_server/group_i", force=TRUE)

model <- h2o.loadModel('models_server/group_i/model_deep')
summary(model)

valid['y_pred'] = h2o.predict(model, valid)
#test_h2o['y_pred'] = h2o.predict(model, test_h2o)


metrics(valid['y_pred'], valid['TrueAnswer'])
#metrics(test_h2o['y_pred'], test_h2o['TrueAnswer'])














