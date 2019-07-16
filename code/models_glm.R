library(tidyverse)
library(data.table)
library(h2o)
library(MLmetrics)
library(lubridate)
source('code/tools.R')
source('code/plott.R')

# start h2o session 
h2o.init(nthreads=-1, max_mem_size="16G")
train = h2o.importFile(path = 'data/csv_cut/data_train.csv')
valid = h2o.importFile(path = 'data/csv_cut/data_val.csv')
dim(train)
names(train)
h2o.describe(train)


#train['TrueAnswer_log'] = log(train['TrueAnswer'])
#train['TrueAnswer_inv'] = (train['TrueAnswer'])^-1
#valid['TrueAnswer_log'] = log(valid['TrueAnswer'])
#valid['TrueAnswer_inv'] = (valid['TrueAnswer'])^-1

train['TrueAnswer_norm'] = my_scale(train['TrueAnswer'])
valid['TrueAnswer_norm'] = my_scale(valid['TrueAnswer'], 
                                    mean = mean(train['TrueAnswer']),
                                    sd = sd(train['TrueAnswer']))

# set X and y 
y <- "TrueAnswer_norm"
y_log = "TrueAnswer_log"
y_inv = 'TrueAnswer_inv'
X = names(train)[c(3, 10:59, 63)]

##################################################################
################### Baseline Linear Regression ###################
##################################################################
model_baseline <- h2o.glm(
  model_id="model_baseline", 
  training_frame=train, 
  validation_frame=valid,  
  y=y,
  x=X,
  family = 'gaussian'
  
)

# performance check 
summary(model_baseline)
h2o.varimp(model_baseline)
h2o.performance(model_baseline, newdata=train)    ## full training data
h2o.performance(model_baseline, newdata=valid)    ## full validation data

# normal check
metrics(h2o.predict(model_baseline, train), train[y])
# scale back 
metrics(unscale(h2o.predict(model_baseline, train),
                mean = mean(train['TrueAnswer']),
                sd = sd(train['TrueAnswer'])), train['TrueAnswer'])



valid['y_pred_baseline'] = h2o.predict(model_baseline, valid)
metrics(valid['y_pred_baseline'], valid[y])

# scale back 
valid['y_pred_baseline'] = unscale(h2o.predict(model_baseline, valid),
                                   mean = mean(train['TrueAnswer']),
                                   sd = sd(train['TrueAnswer']))
metrics(valid['y_pred_baseline'], valid['TrueAnswer'])



valid_dt = as.data.table(valid)

# id start from 167
plotPred(valid_dt, group = 'GroupF-193', model = 'baseline', activity = FALSE)



##################################################################
################### Poisson Linear Regression ###################
##################################################################
model_poisson <- h2o.glm(
  model_id="model_poisson", 
  training_frame=train, 
  validation_frame=valid,  
  x=X,
  y=y,
  family = 'poisson'
  
)

# performance check 
summary(model_poisson)

metrics(h2o.predict(model_poisson, train), train[y])
valid['y_pred_poisson'] = h2o.predict(model_poisson, valid)
metrics(valid['y_pred_poisson'], valid[y])

valid_dt$y_pred_poisson = as.data.table(valid$y_pred_poisson)
plotPred(valid_dt, group = 'GroupF-193', model = 'poisson', activity = TRUE)



######## distribution check ################
data %>% ggplot(aes(x= TrueAnswer_inv))+geom_histogram(bins= 2000)
data %>% ggplot(aes(x= TrueAnswer_log))+geom_histogram(bins= 2000)
data %>% ggplot(aes(x= TrueAnswer))+geom_histogram(bins= 2000)
###############################################



##################################################################
################### Log Linear Regression ########################
##################################################################
model_gaussian_log <- h2o.glm(
  model_id="model_gaussian_log", 
  training_frame=train, 
  validation_frame=valid,   
  x=X,
  y=y_log,
  family = 'gaussian'
  
)

metrics(exp(h2o.predict(model_gaussian_log, train)), train[y])
valid['y_pred_gaussian_log'] = exp(h2o.predict(model_gaussian_log, valid))
metrics(valid['y_pred_gaussian_log'], valid[y])

valid_dt$y_pred_gaussian_log = as.data.table(valid$y_pred_gaussian_log)
plotPred(valid_dt, group = 'GroupF-193', model = 'gaussian_log', activity = FALSE)

##################################################################
################### Inv Linear Regression ########################
##################################################################
model_gaussian_Inv <- h2o.glm(
  model_id="model_gaussian_Inv", 
  training_frame=train, 
  validation_frame=valid,   
  x=X,
  y=y_inv,
  family = 'gaussian'
  
)

metrics((h2o.predict(model_gaussian_Inv, train))^-1, train[y])
valid['y_pred_gaussian_Inv'] = (h2o.predict(model_gaussian_Inv, valid))^-1
metrics(valid['y_pred_gaussian_Inv'], valid[y])

valid_dt$y_pred_gaussian_Inv = as.data.table(valid$y_pred_gaussian_Inv)
plotPred(valid_dt, group = 'GroupA', model = 'gaussian_Inv')

##################################################################
###################  Inv Poisson Regression ######################
##################################################################
model_poisson_inv <- h2o.glm(
  model_id="model_poisson_inv", 
  training_frame=train, 
  validation_frame=valid,   
  x=X,
  y=y_inv,
  family = 'poisson'
  
)

metrics((h2o.predict(model_poisson_inv, train))^-1, train[y])
valid['y_pred_poisson_Inv'] = (h2o.predict(model_poisson_inv, valid))^-1
metrics(valid['y_pred_poisson_Inv'], valid[y])

valid_dt$y_pred_poisson_Inv = as.data.table(valid$y_pred_poisson_Inv)
plotPred(valid_dt, group = 'GroupA', model = 'poisson_Inv')



##################################################################
################  Basline Fixed for Negitive  ####################
##################################################################

valid['y_pred_baseline_adjust'] = ifelse(valid['y_pred_baseline']>0, 
                                         valid['y_pred_baseline'], 0)

valid_dt$y_pred_baseline_adjust = as.data.table(valid$y_pred_baseline_adjust)
plotPred(valid_dt, group = 'GroupA-170', model = 'baseline_adjust', activity = FALSE)
plotPred(valid_dt, group = 'GroupA-170', model = 'baseline', activity = FALSE)


metrics(valid['y_pred_baseline_adjust'], valid[y])





plotPred(valid_dt, group = 'GroupG', model = 'baseline_adjust')
plotPred(valid_dt, group = 'GroupG', model = 'gaussian_log')
plotPred(valid_dt, group = 'GroupG', model = 'baseline')




##dygraph(mike_check[FileGrp==100, .(TIMESTAMP, Activity, TrueAnswer)]) %>% dyOptions(useDataTimezone = TRUE)


h2o.shutdown(prompt = TRUE)


























