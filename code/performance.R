

mike_model = function(group = 'group_a', mojo = 'DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip'){
  group_path = paste0('data/', group, '/valid.csv')
  mojo_path = paste0('models_server/mojo/', group, '/', mojo)
  pred = h2o.mojo_predict_csv(
    input_csv_path = group_path,
    mojo_zip_path = mojo_path
  )
return(pred)
}



################## A ############################
# from 800
metrics(valid_dt_a$y_pred_xgb, valid_dt_a$TrueAnswer)
plotPred(valid_dt_a, group = 'GroupA-926', model = 'xgb', activity = FALSE)

valid_dt_a$y_pred_mike = mike_model(group = 'group_a', mojo = 'DeepLearning_grid_1_AutoML_20190528_031324_model_54.zip')
metrics(valid_dt_a$y_pred_mike, valid_dt_a$TrueAnswer)

################## B ############################
# from 2960
metrics(valid_dt_b$y_pred_xgb, valid_dt_b$TrueAnswer)
plotPred(valid_dt_b, group = 'GroupB-2966', model = 'xgb', activity = TRUE)


valid_dt_b$y_pred_mike = mike_model(group = 'group_b', mojo = 'DeepLearning_1_AutoML_20190528_055538.zip')
metrics(valid_dt_b$y_pred_mike, valid_dt_b$TrueAnswer)

################## C ############################
# from 3790
metrics(valid_dt_c$y_pred_xgb, valid_dt_c$TrueAnswer)
plotPred(valid_dt_c, group = 'GroupC-926', model = 'xgb', activity = FALSE)


valid_dt_c$y_pred_mike = mike_model(group = 'group_c', mojo = 'DeepLearning_grid_1_AutoML_20190528_083849_model_18.zip')
metrics(valid_dt_c$y_pred_mike, valid_dt_c$TrueAnswer)

################## D ############################
# d 1
metrics(valid_dt_d$y_pred_rf, valid_dt_d$TrueAnswer)
plotPred(valid_dt_d, group = 'GroupD-4', model = 'rf', activity = FALSE)

valid_dt_d$y_pred_mike = mike_model(group = 'group_d', mojo = 'DeepLearning_grid_1_AutoML_20190528_112144_model_32.zip')
metrics(valid_dt_d$y_pred_mike, valid_dt_d$TrueAnswer)


################## E ############################

metrics(valid_dt_e$y_pred_rf, valid_dt_e$TrueAnswer)
plotPred(valid_dt_e, group = 'GroupE-4', model = 'rf', activity = FALSE)

valid_dt_e$y_pred_mike = mike_model(group = 'group_e', mojo = 'DeepLearning_grid_1_AutoML_20190528_140441_model_9.zip')
metrics(valid_dt_e$y_pred_mike, valid_dt_e$TrueAnswer)

################## F ############################

metrics(valid_dt_f$y_pred_rf, valid_dt_f$TrueAnswer)
plotPred(valid_dt_f, group = 'GroupF-4', model = 'rf', activity = FALSE)

valid_dt_f$y_pred_mike = mike_model(group = 'group_f', mojo = 'DeepLearning_grid_1_AutoML_20190528_164841_model_6.zip')
metrics(valid_dt_f$y_pred_mike, valid_dt_f$TrueAnswer)


################## G ############################

metrics(valid_dt_g$y_pred_rf, valid_dt_g$TrueAnswer)
plotPred(valid_dt_g, group = 'GroupG-4', model = 'rf', activity = FALSE)

valid_dt_g$y_pred_mike = mike_model(group = 'group_g', mojo = 'DeepLearning_1_AutoML_20190528_193315.zip')
metrics(valid_dt_g$y_pred_mike, valid_dt_g$TrueAnswer)


################## H ############################

metrics(valid_dt_h$y_pred_rf, valid_dt_h$TrueAnswer)
plotPred(valid_dt_h, group = 'GroupH-4', model = 'rf', activity = FALSE)

valid_dt_h$y_pred_mike = mike_model(group = 'group_h', mojo = 'DeepLearning_1_AutoML_20190527_035957.zip')
metrics(valid_dt_h$y_pred_mike, valid_dt_h$TrueAnswer)



################## I ############################

metrics(valid_dt_i$y_pred_rf, valid_dt_i$TrueAnswer)
plotPred(valid_dt_i, group = 'GroupI-4', model = 'rf', activity = FALSE)

valid_dt_i$y_pred_mike = mike_model(group = 'group_i', mojo = 'DeepLearning_grid_1_AutoML_20190528_234542_model_18.zip')
metrics(valid_dt_i$y_pred_mike, valid_dt_i$TrueAnswer)








model = h2o.loadModel('models_server/model_deep_1')

model = auto@leader


valid_test = h2o.importFile(path = 'data/group_e/valid.csv')
valid_test['pred'] = h2o.predict(model, valid_test)
metrics(valid_test['pred'], valid_test['TrueAnswer'])








valid_all = valid
valid['pred'] = (h2o.predict(model, valid))
metrics(valid['pred'], valid['TrueAnswer'])













