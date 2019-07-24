library(tidyverse)
library(data.table)
library(xgboost)
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



Dtrain = xgb.DMatrix(data = train_X, label = train$TrueAnswer )


#############################################################
###################### Model ################################
#############################################################

xgb = xgb.train(data = Dtrain,
              max.depth = 10,
              eta = 0.05,
              nthread = -1,
              nrounds = 100,
              verbosity = 2,
              objective = wls
              )



valid['y_pred_xgb'] = (predict(xgb, valid_X))


metrics(valid$y_pred_xgb, valid$TrueAnswer)

plotPred(valid, group = 'GroupC-182', model = 'xgb', activity = FALSE)



#################################################################################
########################## Custom Loss ##########################################
#################################################################################

## Squared Log Error (SLE)
sle = function(pred, dtrain){
  label = getinfo(dtrain, 'label')
  grad = (log(pred+1)-log(label+1))/(pred+1)
  hess = (1-log(pred+1)+log(label+1))/(pred+1)^2
  return(list(grad = grad, hess = hess))
}




## Weighted least square 
wls = function(pred, dtrain){
  label = getinfo(dtrain, 'label')
  grad = (2/label)*(pred-label)
  hess = (2/label)
  return(list(rad = grad, hess = hess))
}





























