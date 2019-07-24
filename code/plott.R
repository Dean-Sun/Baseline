library(corrplot)
library(dygraphs)
library(gridExtra)
library(lubridate)

plotAct  = function(data, day){
  if (day %in% data$FileGrp){
    group_name = data$CME_Group[1]
    data %>%
      filter(FileGrp == day)%>%
      select(TIMESTAMP, Activity, TrueAnswer, Baseline)%>%
      as.data.table()%>%
      dygraph(main = group_name, xlab = paste0('FileGrp: ', day))%>%
      dyRangeSelector()%>%
      dyOptions(useDataTimezone = TRUE)
  }else{
    print('invalid day')
  }
  
}



plotCorr = function(data){
  data%>%
    select(c(Activity, TrueAnswer, Baseline, 10:59))%>%
    cor()%>%
    corrplot(method="circle", tl.cex = 0.6)
}


plotDist = function (data){
  limit = quantile(data$Activity, 0.99)
  p1 = data %>% ggplot(aes(x= Activity))+geom_histogram(bins= 500)+xlim(NA, limit)
  p2 = data %>% ggplot(aes(x= TrueAnswer))+geom_histogram(bins= 500)+xlim(NA, limit)
  p3 = data %>% ggplot(aes(x= Baseline))+geom_histogram(bins= 500)+xlim(NA, limit)
  plot(arrangeGrob(p1, p2, p3))
}



plotPred = function(data, group = 'GroupA-1', model = 'baseline', activity = FALSE){
  pred = paste0('y_pred_', model)
  data$TIMESTAMP = ymd_hms(data$TIMESTAMP)
  if (activity == TRUE){
    data %>%
      filter(id == group)%>%
      select(TIMESTAMP, TrueAnswer, Activity, pred)%>%
      as.data.table()%>%
      dygraph(main = paste0('ID: ', group))%>%
      dyRangeSelector()%>%
      dyOptions(useDataTimezone = TRUE)
  }else{
    data %>%
      filter(id == group)%>%
      select(TIMESTAMP, TrueAnswer, pred)%>%
      as.data.table()%>%
      dygraph(main = paste0('ID: ', group))%>%
      dyRangeSelector()%>%
      dyOptions(useDataTimezone = TRUE)
  }
  
}






















































