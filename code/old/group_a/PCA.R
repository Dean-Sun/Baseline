library(tidyverse)
library(data.table)
library(caret)
library(h2o)

train = read_csv('data/group_a/train.csv')
valid = read_csv('data/group_a/valid.csv')

y = 'TrueAnswer'
X = names(train)[c(3, 10:59)]


pca <- preProcess(train[X], method=c("center", "scale", "pca"))

print(pca)

pca_train <- predict(pca, train)
pca_valid = predict(pca, valid)

# summarize the transformed dataset
summary(pca_train)

## save the data

write_csv(pca_train, 'data/group_a/train_pca.csv')
write_csv(pca_valid, 'data/group_a/val_pca.csv')



###############################################################################################
###############################################################################################
###############################################################################################


# start h2o session 
h2o.init(nthreads=-1, max_mem_size="52G")
train = h2o.importFile(path = 'data/group_a/train_pca.csv')
valid = h2o.importFile(path = 'data/group_a/val_pca.csv')

# log the label
train['TrueAnswer_log'] = log(train['TrueAnswer'])
valid['TrueAnswer_log'] = log(valid['TrueAnswer'])

# set X and y 
y_true = 'TrueAnswer'
y_log = 'TrueAnswer_log'
X = names(train)[15:25]


##################################################################
######################## Random Forest ###########################
##################################################################

model_rf_log <- h2o.randomForest(
  model_id = 'model_rf_pca',
  training_frame=train, 
  validation_frame=valid,  
  y=y_log,
  x=X,
  ntrees = 20,
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
                     model_id = 'model_gbm_pca',
                     validation_frame=valid,
                     x = X, 
                     y = y_log,
                     ntrees = 200,
                     max_depth = 15,
                     stopping_rounds = 5,
                     stopping_metric = 'MSE',
                     stopping_tolerance = 0.001)
# performance check 
summary(model_gbm)

valid['y_pred_gbm'] = exp(h2o.predict(model_gbm, valid))
metrics(valid['y_pred_gbm'], valid[y_true])

valid_dt$y_pred_gbm = as.data.table(valid$y_pred_gbm)
plotPred(valid_dt, group = 'GroupA-804', model = 'gbm', activity = FALSE)



##################################################################
######################## XGBoost #################################
##################################################################

model_xgb <- h2o.xgboost(training_frame=train, 
                         validation_frame=valid,
                         x = X, 
                         y = y_log,
                         ntrees = 400,
                         max_depth = 11,
                         learn_rate = 0.07,
                         col_sample_rate = 0.8,
                         col_sample_rate_per_tree = 0.8,
                         min_rows = 512,
                         reg_alpha = 0.5,
                         reg_lambda = 0.8,
                         sample_rate = 0.5,
                         stopping_rounds = 5,
                         stopping_metric = 'MSE',
                         stopping_tolerance = 0.0001,
                         verbose = TRUE)
# performance check 
summary(model_xgb)

valid['y_pred_xgb'] = exp(h2o.predict(model_xgb, valid))
metrics(valid['y_pred_xgb'], valid[y_true])

valid_dt$y_pred_xgb = as.data.table(valid$y_pred_xgb)
plotPred(valid_dt, group = 'GroupA-813', model = 'xgb', activity = TRUE)




##################################################################
######################## Deep Learning ###########################
##################################################################

model_deep <- h2o.deeplearning(
  model_id = 'model_deep',
  training_frame=train, 
  validation_frame=valid,
  x=X,
  y=y_true,
  hidden=c(32,32),
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

valid['y_pred_deep'] = (h2o.predict(model_deep, valid))
metrics(valid['y_pred_deep'], valid[y_true])






















