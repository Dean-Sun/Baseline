plotAct  = function(data, day){
  if (day %in% c(1:length(unique(data$FileGrp)))){
    group_a %>%
      filter(FileGrp == day)%>%
      ggplot()+geom_line(aes(x=TIMESTAMP, y= Activity))+
      geom_line(aes(x=TIMESTAMP, y= TrueAnswer, color = 'TrueAnswer'), size = 1)+
      geom_line(aes(x=TIMESTAMP, y= Baseline, color = 'Baseline'))
  }else{
    print('invalid day')
  }
  
}

###############################################################
########## Test For multiple  values in one date ###############
###############################################################
test = data.table(time = c('2019-01-01','2019-01-02','2019-01-03',
                           '2019-01-01','2019-01-02','2019-01-03'),
                  value = c(1,2,3,11,12,13))
test$time = ymd(test$time)
test = test[1:3]
test %>%
  ggplot()+geom_line(aes(x=time, y= value))


#################################################################
##############################################################
# train, val split
splits <- h2o.splitFrame(data.hex, 0.8, seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%




plotPred = function(data, group = 'GroupA', model = 'baseline', samples = 1000){
  pred = paste0('y_pred_', model)
  data = data%>%filter(CME_Group == group)%>%sample_n(samples)
  data%>%
    mutate(id = 1:nrow(data))%>%
    ggplot()+geom_line(aes(x=id, y= TrueAnswer))+
    geom_line(aes(x=id, y= data[[pred]], color = 'Prediction'))+xlab(paste0(group, '  ', model))
  
}












