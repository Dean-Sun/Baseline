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
                     MAPE = MAPE(y_pred, y_true))
  return(table)
}















































