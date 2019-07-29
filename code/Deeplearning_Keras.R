library(tidyverse)
library(data.table)
library(keras)
library(caret)
source('code/tools.R')
source('code/plott.R')
library(matrixStats)

train = read_csv('data/csv_cut/data_train.csv')
valid = read_csv('data/csv_cut/data_val.csv')

##################################################
############# Data Preprocess ####################
##################################################

# standerize the and log the label 
train = train%>%
  mutate(TrueAnswer_norm = scale(TrueAnswer),
         TrueAnswer_log = log(TrueAnswer))
valid = valid%>%
  mutate(TrueAnswer_norm = scale(TrueAnswer, 
                                 center = mean(train$TrueAnswer),
                                 scale = sd(train$TrueAnswer)),
         TrueAnswer_log = log(TrueAnswer))
# select features and labels
X_num = names(train)[c(3, 10:59)]
X_cat = names(train)[63]

# seperate feature matrix
train_X_num = data.matrix(train[X_num])
valid_X_num = data.matrix(valid[X_num])

train_X_cat = as.matrix(train[X_cat])
valid_X_cat = as.matrix(valid[X_cat])

# standaerize features
train_X_num <- scale(train_X_num) 
col_means_train <- attr(train_X_num, "scaled:center") 
col_stddevs_train <- attr(train_X_num, "scaled:scale")
valid_X_num <- scale(valid_X_num, center = col_means_train, scale = col_stddevs_train)

# one_hot features
dmy <- dummyVars(" ~ .", data = train_X_cat)
train_X_cat = predict(dmy, train_X_cat)
valid_X_cat = predict(dmy, valid_X_cat)


# combine features together 
train_X = cbind(train_X_num, train_X_cat)
valid_X = cbind(valid_X_num, valid_X_cat)


# clean the workspace to save RAM
rm(train_X_num,valid_X_num,train_X_cat,valid_X_cat)
train = train%>%select(-c(10:59))
valid = valid%>%select(-c(10:59))


####################################################################
###################### modeling ####################################
####################################################################

model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu",
              input_shape = dim(train_X)[2]) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop(),
  metrics = list("mse",'mae')
)

model%>%summary()


model %>% fit(
  train_X,
  train$TrueAnswer,
  batch_size = 64,
  epochs = 12,
  #sample_weight = 1/train$TrueAnswer,
  validation_data = list(valid_X, valid$TrueAnswer),
  verbose = 1
)


model%>%evaluate(valid_X, valid$TrueAnswer)

valid['y_pred_deep'] = predict(model, valid_X)


metrics(valid$y_pred_deep, valid$TrueAnswer)

plotPred(valid, group = 'GroupI-182', model = 'deep', activity = TRUE)










