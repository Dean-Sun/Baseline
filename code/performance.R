

mike_model = function(group = 'group_a', mojo = 'DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip', which = 'valid'){
  group_path = paste0('data/', group, '/', which, '.csv')
  mojo_path = paste0('models_server/mojo/', group, '/', mojo)
  pred = h2o.mojo_predict_csv(
    input_csv_path = group_path,
    mojo_zip_path = mojo_path
  )
return(pred)
}

my_model = function(group = 'group_a', model= 'xgb', mojo = 'DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip', which = 'valid'){
  group_path = paste0('data/', group, '/', which, '.csv')
  mojo_path = paste0('models_server/separate_models/mojo/', group, '/', model, '/', mojo)
  if (grepl('log', model)){
    pred = exp(h2o.mojo_predict_csv(input_csv_path = group_path, mojo_zip_path = mojo_path))
  }else{
    pred = h2o.mojo_predict_csv(input_csv_path = group_path, mojo_zip_path = mojo_path)
  }
  return(pred)
}



valid = h2o.importFile(path = 'data/group_i/valid.csv')

model_xgb = h2o.loadModel('models_server/separate_models/group_i/Grid_XGBoost_RTMP_sid_92bb_64_model_R_1565729418965_17_model_6')
valid['y_pred_xgb'] = (h2o.predict(model_xgb, valid))
model_xgb_log = h2o.loadModel('models_server/separate_models/group_i/Grid_XGBoost_RTMP_sid_92bb_64_model_R_1565729418965_18_model_4')
valid['y_pred_xgb_log'] = exp(h2o.predict(model_xgb_log, valid))
model_deep = h2o.loadModel('models_server/separate_models/group_i/model_deep')
valid['y_pred_deep'] = (h2o.predict(model_deep, valid))

valid_dt_i = as.data.table(valid)


###########################################################
################## A ############################
# from 800
metrics(valid_dt_a$y_pred_xgb, valid_dt_a$TrueAnswer)

valid_dt_a$y_pred_mike = mike_model(group = 'group_a', mojo = 'DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip')
metrics(valid_dt_a$y_pred_mike, valid_dt_a$TrueAnswer)

plotPred(valid_dt_a, group = 'GroupA-934', model = c('xgb_log', 'mike'), activity = FALSE)

################## B ############################
# from 2960
metrics(valid_dt_b$y_pred_xgb, valid_dt_b$TrueAnswer)

valid_dt_b$y_pred_mike = mike_model(group = 'group_b', mojo = 'DeepLearning_1_AutoML_20190528_055538.zip')
metrics(valid_dt_b$y_pred_mike, valid_dt_b$TrueAnswer)

plotPred(valid_dt_b, group = 'GroupB-2970', model = c('xgb_log', 'mike'), activity = FALSE)

################## C ############################
# from 3790
metrics(valid_dt_c$y_pred_xgb, valid_dt_c$TrueAnswer)

valid_dt_c$y_pred_mike = mike_model(group = 'group_c', mojo = 'DeepLearning_grid_1_AutoML_20190528_083849_model_18.zip')
metrics(valid_dt_c$y_pred_mike, valid_dt_c$TrueAnswer)

plotPred(valid_dt_c, group = 'GroupC-3796', model = c('xgb_log', 'mike'), activity = FALSE)

################## D ############################
# d 1647
metrics(valid_dt_d$y_pred_rf, valid_dt_d$TrueAnswer)

valid_dt_d$y_pred_mike = mike_model(group = 'group_d', mojo = 'DeepLearning_grid_1_AutoML_20190528_112144_model_32.zip')
metrics(valid_dt_d$y_pred_mike, valid_dt_d$TrueAnswer)

plotPred(valid_dt_d, group = 'GroupD-1700', model = c('xgb_log', 'mike'), activity = FALSE)

################## E ############################
# 4331

metrics(valid_dt_e$y_pred_rf, valid_dt_e$TrueAnswer)

valid_dt_e$y_pred_mike = mike_model(group = 'group_e', mojo = 'DeepLearning_grid_1_AutoML_20190528_140441_model_9.zip')
metrics(valid_dt_e$y_pred_mike, valid_dt_e$TrueAnswer)

plotPred(valid_dt_e, group = 'GroupE-4331', model = c('deep', 'mike'), activity = FALSE)

################## F ############################
# 6053

metrics(valid_dt_f$y_pred_rf, valid_dt_f$TrueAnswer)

valid_dt_f$y_pred_mike = mike_model(group = 'group_f', mojo = 'DeepLearning_grid_1_AutoML_20190528_164841_model_6.zip')
metrics(valid_dt_f$y_pred_mike, valid_dt_f$TrueAnswer)

plotPred(valid_dt_f, group = 'GroupF-6053', model = c('deep', 'mike', 'xgb'), activity = FALSE)

################## G ############################
# 5647 

metrics(valid_dt_g$y_pred_rf, valid_dt_g$TrueAnswer)

valid_dt_g$y_pred_mike = mike_model(group = 'group_g', mojo = 'DeepLearning_1_AutoML_20190528_193315.zip')
metrics(valid_dt_g$y_pred_mike, valid_dt_g$TrueAnswer)

plotPred(valid_dt_g, group = 'GroupG-5649', model = c('mike', 'xgb'), activity = FALSE)

################## H ############################
# 2740

metrics(valid_dt_h$y_pred_rf, valid_dt_h$TrueAnswer)

valid_dt_h$y_pred_mike = mike_model(group = 'group_h', mojo = 'DeepLearning_1_AutoML_20190527_035957.zip')
metrics(valid_dt_h$y_pred_mike, valid_dt_h$TrueAnswer)

plotPred(valid_dt_h, group = 'GroupH-2742', model = c('mike', 'deep'), activity = FALSE)

################## I ############################
# 2733

metrics(valid_dt_i$y_pred_deep, valid_dt_i$TrueAnswer)

valid_dt_i$y_pred_mike = mike_model(group = 'group_i', mojo = 'DeepLearning_grid_1_AutoML_20190528_234542_model_18.zip')
metrics(valid_dt_i$y_pred_mike, valid_dt_i$TrueAnswer)

plotPred(valid_dt_i, group = 'GroupI-2740', model = c('mike', 'deep'), activity = FALSE)




################# Test ###########################


model = h2o.loadModel('models_server/separate_models/group_f/Grid_XGBoost_RTMP_sid_92bb_58_model_R_1565729418965_11_model_5')


#model = h2o.getModel(grid_xgb_c@model_ids[[1]])
valid_test = h2o.importFile(path = 'data/group_f/valid.csv')
valid_test['y_pred_xgb'] = (h2o.predict(model, valid_test))
metrics(valid_test['y_pred_xgb'], valid_test['TrueAnswer'])


h2o.saveModel(model, path="models_server/separate_models/group_c", force=TRUE)




################# Change to Mojo #####################

model = h2o.loadModel('models_server/separate_models/group_i/Grid_XGBoost_RTMP_sid_92bb_64_model_R_1565729418965_18_model_4')
path = 'models_server/separate_models/mojo/group_i/xgb_log'
model = h2o.download_mojo(model, path, get_genmodel_jar = TRUE)



################## Test Time #########################



#################### Group A ##################
a_xgb_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  a_xgb = my_model(group = 'group_a', model= 'xgb', mojo = 'Grid_XGBoost_RTMP_sid_9db4_17_model_R_1565802347713_1_model_21.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  a_xgb_time[i] = time.taken
}
mean(a_xgb_time)


a_xgb_log_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  a_xgb_log =my_model(group = 'group_a', model= 'xgb_log', mojo = 'Grid_XGBoost_RTMP_sid_9db4_17_model_R_1565802347713_2_model_13.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  a_xgb_log_time[i] = time.taken
}
mean(a_xgb_log_time)


a_deep_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  a_deep =my_model(group = 'group_a', model= 'deep', mojo = 'model_deep.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  a_deep_time[i] = time.taken
}

mean(a_deep_time)


a_mike_pred = mike_model(group = 'group_a', mojo = 'DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip')

test = read_csv('data/group_a/test.csv')
metrics(a_mike_pred$predict, test$TrueAnswer)







#################### Group G ##################
g_xgb_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  g_xgb = my_model(group = 'group_g', model= 'xgb', mojo = 'Grid_XGBoost_RTMP_sid_92bb_60_model_R_1565729418965_13_model_5.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  g_xgb_time[i] = time.taken
}
mean(g_xgb_time)


g_xgb_log_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  g_xgb_log =my_model(group = 'group_g', model= 'xgb_log', mojo = 'Grid_XGBoost_RTMP_sid_92bb_60_model_R_1565729418965_14_model_5.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  g_xgb_log_time[i] = time.taken
}
mean(g_xgb_log_time)


g_deep_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  g_deep =my_model(group = 'group_g', model= 'deep', mojo = 'model_deep.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  g_deep_time[i] = time.taken
}

mean(g_deep_time)


g_mike_pred = mike_model(group = 'group_g', mojo = 'DeepLearning_1_AutoML_20190528_193315.zip')

test = read_csv('data/group_g/test.csv')
metrics(a_xgb_log$predict, test$TrueAnswer)





#################### Group I ##################
i_xgb_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  i_xgb = my_model(group = 'group_i', model= 'xgb', mojo = 'Grid_XGBoost_RTMP_sid_92bb_64_model_R_1565729418965_17_model_6.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  i_xgb_time[i] = time.taken
}
mean(i_xgb_time)




i_deep_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  i_deep =my_model(group = 'group_i', model= 'deep', mojo = 'model_deep.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  i_deep_time[i] = time.taken
}

mean(i_deep_time)


i_mike_pred = mike_model(group = 'group_i', mojo = 'DeepLearning_grid_1_AutoML_20190528_234542_model_18.zip')

test = read_csv('data/group_i/test.csv')
metrics(i_xgb$predict, test$TrueAnswer)





#################### Other ##################
xgb_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  xgb = my_model(group = 'group_b', model= 'xgb', mojo = 'Grid_XGBoost_RTMP_sid_9db4_19_model_R_1565802347713_3_model_6.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  xgb_time[i] = time.taken
}
mean(xgb_time)




deep_time = c()
for (i in 1:100){
  start.time <- Sys.time()
  deep =my_model(group = 'group_b', model= 'deep', mojo = 'model_deep.zip')
  end.time <- Sys.time()
  time.taken <- end.time - start.time  
  deep_time[i] = time.taken
}

mean(deep_time)


mike_pred = mike_model(group = 'group_b', mojo = 'DeepLearning_1_AutoML_20190528_055538.zip')

test = read_csv('data/group_b/test.csv')
metrics(a_deep$predict, test$TrueAnswer)


