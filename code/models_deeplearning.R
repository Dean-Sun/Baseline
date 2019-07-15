library(tidyverse)
library(data.table)
library(h2o)
library(MLmetrics)
library(lubridate)
source('code/tools.R')
source('code/plott.R')

data = read_csv('data/csv_cut/data_val.csv')
# start h2o session 
h2o.init(nthreads=-1, max_mem_size="10G")
train = h2o.importFile(path = 'data/csv_cut/data_train.csv')
valid = h2o.importFile(path = 'data/csv_cut/data_val.csv')
# set X and y 
y <- "TrueAnswer"
X = names(train)[c(3, 10:59, 63)]


model_deep_1 <- h2o.deeplearning(
  model_id="model_deep_1", 
  training_frame=train, 
  validation_frame=valid,
  x=X,
  y=y,
  hidden=c(5,3),
  variable_importances=T,
  epochs=1000000,                      ## hopefully converges earlier...
  score_validation_samples=10000,      ## sample the validation dataset (faster)
  stopping_rounds=15,
  stopping_metric="MSE", ## could be "MSE","logloss","r2"
  stopping_tolerance=0.001,
  verbose = TRUE
  )

# performance check 
summary(model_deep_1)
h2o.varimp(model_deep_1)
h2o.performance(model_deep_1, newdata=train)    ## full training data
h2o.performance(model_deep_1, newdata=valid)    ## full validation data

metrics(h2o.predict(model_deep_1, train), train[y])
valid['y_pred_deep'] = h2o.predict(model_deep_1, valid)
metrics(valid['y_pred_deep'], valid[y])

# Plot
valid_dt = as.data.table(valid)
plotPred(valid_dt, group = 'GroupH-194', model = 'deep', activity = FALSE)





path <- h2o.saveModel(model_deep_1, 
                      path="models", force=TRUE)



model_rf <- h2o.loadModel('models/model_rf')




summary(model )









