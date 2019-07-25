library(tidyverse)
library(data.table)
library(h2o)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="58G")
train = h2o.importFile(path = 'data/group_a/train.csv')
valid = h2o.importFile(path = 'data/group_a/valid.csv')

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
  ntrees = 100,
  max_depth = 15,
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
plotPred(valid_dt, group = 'GroupA-800', model = 'rf', activity = FALSE)







#########################################################################
###################### Save and Load ####################################
#########################################################################
# Save the model
path <- h2o.saveModel(model_rf, path="models_server/group_a", force=TRUE)

model <- h2o.loadModel('models_server/group_a/model_rf')
summary(model)

valid['y_pred'] = h2o.predict(model, valid)
test_h2o['y_pred'] = h2o.predict(model, test_h2o)


metrics(valid['y_pred'], valid['TrueAnswer'])
metrics(test_h2o['y_pred'], test_h2o['TrueAnswer'])







































