
hyper_params = list(
  ntrees = c(200,300,400),
  max_depth = seq(8,15,1),
  learn_rate = seq(0.01, 0.2, 0.01),
  sample_rate = seq(0.2,1,0.01),
  col_sample_rate = seq(0.2,1,0.01),
  col_sample_rate_per_tree = seq(0.2,1,0.01),
  min_rows = seq(0,500,50),
  reg_lambda = seq(0,1,0.1),
  reg_alpha = seq(0,1,0.1)
)

search_criteria = list(
  strategy = "RandomDiscrete",
  max_runtime_secs = 3600,
  max_models = 50,
  seed = 1234,
  stopping_rounds = 3,
  stopping_metric = "MSE",
  stopping_tolerance = 0.001
)

train_a = h2o.importFile(path = 'data/group_a/train.csv')
valid_a = h2o.importFile(path = 'data/group_a/valid.csv')

train_b = h2o.importFile(path = 'data/group_b/train.csv')
valid_b = h2o.importFile(path = 'data/group_b/valid.csv')

train_c = h2o.importFile(path = 'data/group_c/train.csv')
valid_c = h2o.importFile(path = 'data/group_c/valid.csv')


train_d = h2o.importFile(path = 'data/group_d/train.csv')
valid_d = h2o.importFile(path = 'data/group_d/valid.csv')

train_e = h2o.importFile(path = 'data/group_e/train.csv')
valid_e = h2o.importFile(path = 'data/group_e/valid.csv')

train_f = h2o.importFile(path = 'data/group_f/train.csv')
valid_f = h2o.importFile(path = 'data/group_f/valid.csv')

train_g = h2o.importFile(path = 'data/group_g/train.csv')
valid_g = h2o.importFile(path = 'data/group_g/valid.csv')

train_h = h2o.importFile(path = 'data/group_h/train.csv')
valid_h = h2o.importFile(path = 'data/group_h/valid.csv')

train_i = h2o.importFile(path = 'data/group_i/train.csv')
valid_i = h2o.importFile(path = 'data/group_i/valid.csv')

# log the label
train_c['TrueAnswer_log'] = log(train_c['TrueAnswer'])
valid_c['TrueAnswer_log'] = log(valid_c['TrueAnswer'])

# set X and y 
y_true = 'TrueAnswer'
y_log = 'TrueAnswer_log'
X = names(train_a)[c(3, 10:59, 63)]

################# A #########################
grid_xgb_a <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_a,
  validation_frame = valid_a,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_a <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_a,
  validation_frame = valid_a,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)

################# B #########################
grid_xgb_b <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_b,
  validation_frame = valid_b,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_b <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_b,
  validation_frame = valid_b,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)

################# C #########################
grid_xgb_c <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_c,
  validation_frame = valid_c,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_c <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_c,
  validation_frame = valid_c,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)

################# D #########################
grid_xgb_d <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_d,
  validation_frame = valid_d,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_d <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_d,
  validation_frame = valid_d,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)

################# E #########################
grid_xgb_e <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_e,
  validation_frame = valid_e,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_e <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_e,
  validation_frame = valid_e,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)

################# F #########################
grid_xgb_f <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_f,
  validation_frame = valid_f,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_f <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_f,
  validation_frame = valid_f,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)


################# G #########################
grid_xgb_g <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_g,
  validation_frame = valid_g,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_g <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_g,
  validation_frame = valid_g,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)

################# H #########################
grid_xgb_h <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_h,
  validation_frame = valid_h,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_h <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_h,
  validation_frame = valid_h,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)


################# I #########################
grid_xgb_i <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_true,
  training_frame = train_i,
  validation_frame = valid_i,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)
grid_xgb_log_i <- h2o.grid(
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  algorithm = "xgboost",
  x = X,
  y = y_log,
  training_frame = train_i,
  validation_frame = valid_i,
  max_runtime_secs = 1800,
  stopping_rounds = 5, 
  stopping_tolerance = 0.001, 
  stopping_metric = "MSE",
  score_tree_interval = 10,
  seed = 1234
)