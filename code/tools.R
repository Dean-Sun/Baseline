library(MLmetrics)
library(data.table)

truncation = function (data, records = 300000, split_rate = NULL, both = FALSE){
  if (is.null(split_rate) == FALSE){
    records = nrow(data)*split_rate
  }
  upper_limit = as.integer(records/1440)
  data_train = data%>%filter(FileGrp <= upper_limit)
  data_val = data%>%filter(FileGrp > upper_limit)
  if (both==TRUE){
    return(list(data_train, data_val))
  }else{
    return(data_train)
  }
}





metrics = function (y_pred, y_true){
  table = data.table(MSE = MSE(y_pred, y_true),
                     RMSE = RMSE(y_pred, y_true),
                     MAE = MAE(y_pred, y_true),
                     MAPE = MAPE(y_pred, y_true),
                     WLS = sum((1/y_true)*(y_true-y_pred)^2))
  return(table)
}


unscale = function(x, mean, sd){
  return(x*sd+mean)
}

my_scale = function(x, mean = NULL, sd = NULL){
  if (is.null(mean) && is.null(sd)){
    return((x-mean(x))/sd(x))
  }else{
    return((x-mean)/sd)
  }
  
}


performance = function(y_pred, y, type=NULL, mean = NULL, sd = NULL){
  if(is.null(type)){
    return(metrics(y_pred, y))
  }else if(type == 'norm'){
    return(metrics(unscale(y_pred, mean, sd), y))
  }else if(type == 'log'){
    return(metrics(exp(y_pred), y))
  }
}




h2o.mojo_predict_csv_mike <- function (input_csv_path, mojo_zip_path, output_csv_path = NULL, 
                                       genmodel_jar_path = NULL, classpath = NULL, java_options = NULL, 
                                       verbose = F) 
{
  default_java_options <- "-Xmx4g -XX:ReservedCodeCacheSize=256m"
  prediction_output_file <- "prediction.csv"
  if (verbose) {
    cat(sprintf("input_csv:\t%s", input_csv_path), "\n")
  }
  if (!file.exists(input_csv_path)) {
    stop(cat(sprintf("Input csv cannot be found at %s", 
                     input_csv_path), "\n"))
  }
  mojo_zip_path <- normalizePath(mojo_zip_path)
  if (verbose) {
    cat(sprintf("mojo_zip:\t%s", mojo_zip_path), "\n")
  }
  if (!file.exists((mojo_zip_path))) {
    stop(cat(sprintf("MOJO zip cannot be found at %s", mojo_zip_path), 
             "\n"))
  }
  parent_dir <- dirname(mojo_zip_path)
  if (is.null(output_csv_path)) {
    output_csv_path <- file.path(parent_dir, prediction_output_file)
  }
  if (is.null(genmodel_jar_path)) {
    genmodel_jar_path <- file.path(parent_dir, "h2o-genmodel.jar")
  }
  if (verbose) {
    cat(sprintf("genmodel_jar:\t%s", genmodel_jar_path), 
        "\n")
  }
  if (!file.exists(genmodel_jar_path)) {
    stop(cat(sprintf("Genmodel jar cannot be found at %s", 
                     genmodel_jar_path), "\n"))
  }
  if (verbose && !is.null(output_csv_path)) {
    cat(sprintf("output_csv:\t%s", output_csv_path), "\n")
  }
  if (is.null(classpath)) {
    classpath <- genmodel_jar_path
  }
  if (verbose) {
    cat(sprintf("classpath:\t%s", classpath), "\n")
  }
  if (is.null(java_options)) {
    java_options <- default_java_options
  }
  if (verbose) {
    cat(sprintf("java_options:\t%s", java_options), "\n")
  }
  cmd <- c("java")
  java_options_list <- strsplit(java_options, " ")
  for (i in 1:length(java_options_list)) {
    cmd <- c(cmd, java_options_list[[i]])
  }
  cmd <- c(cmd, "-cp", classpath, "hex.genmodel.tools.PredictCsv", 
           "--mojo", mojo_zip_path, "--input", input_csv_path, 
           "--output", output_csv_path, "--decimal")
  cmd_str <- paste(cmd, collapse = " ")
  if (verbose) {
    cat(sprintf("java cmd:\t%s", cmd_str), "\n")
  }
  res <- system(cmd_str)
  if (res != 0) {
    msg <- sprintf("SYSTEM COMMAND FAILED (exit status %d)", 
                   res)
    stop(msg)
  }
  result <- fread(output_csv_path)
  return(result)
}



























