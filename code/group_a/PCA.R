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




model_gbm <- h2o.gbm(training_frame=train, 
                     model_id = 'model_gbm',
                     validation_frame=valid,
                     x = X, 
                     y = y_log,
                     ntrees = 200,
                     max_depth = 20,
                     stopping_rounds = 5,
                     stopping_metric = 'MSE',
                     stopping_tolerance = 0.00001)

# performance check 
summary(model_gbm)

valid['y_pred_gbm'] = exp(h2o.predict(model_gbm, valid))
metrics(valid['y_pred_gbm'], valid[y_true])


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

valid['y_pred_xgb'] = h2o.predict(model_xgb, valid)
metrics(valid['y_pred_xgb'], valid[y_true])




