library(fst)
library(tidyverse)
library(data.table)
library(corrplot)
library(gridExtra)
library(lubridate)
source('code/plott.R')
source('code/tools.R')

#a = read_fst('data/GroupA_train_dat.fst', as.data.table = TRUE, from = 1, to = 100)

group_a = read_csv('data/csv/A2.csv')
group_b = read_csv('data/csv/B2.csv')
group_c = read_csv('data/csv/C2.csv')
group_d = read_csv('data/csv/D2.csv')
group_e = read_csv('data/csv/E2.csv')
group_f = read_csv('data/csv/F2.csv')
group_g= read_csv('data/csv/G2.csv')
group_h = read_csv('data/csv/H2.csv')
group_i = read_csv('data/csv/I2.csv')

###############################################################
#################### EDA  #####################################
###############################################################
plotAct(group_f, day = 16)
#ggsave("ETL/I/Activity.png", Activity, width = 12, height = 6)
plotCorr(group_f)


####################################################################
####################### Cut the data ###############################
####################################################################
a = truncation(group_a)
write_csv(truncation(group_i), 'data/csv_cut/group_i.csv')

group_a = read_csv('data/csv_cut/group_a.csv')
group_b = read_csv('data/csv_cut/group_b.csv')
group_c = read_csv('data/csv_cut/group_c.csv')
group_d = read_csv('data/csv_cut/group_d.csv')
group_e = read_csv('data/csv_cut/group_e.csv')
group_f = read_csv('data/csv_cut/group_f.csv')
group_g = read_csv('data/csv_cut/group_g.csv')
group_h = read_csv('data/csv_cut/group_h.csv')
group_i = read_csv('data/csv_cut/group_i.csv')



data_train = bind_rows(truncation(group_a, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_b, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_c, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_d, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_e, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_f, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_g, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_h, split_rate = 0.8, both = TRUE)[[1]],
                       truncation(group_i, split_rate = 0.8, both = TRUE)[[1]])
                       
data_val = bind_rows(truncation(group_a, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_b, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_c, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_d, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_e, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_f, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_g, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_h, split_rate = 0.8, both = TRUE)[[2]],
                     truncation(group_i, split_rate = 0.8, both = TRUE)[[2]])

data_train['id'] = paste0(data_train$CME_Group, '-', 
                          as.character(data_train$FileGrp))
data_val['id'] = paste0(data_val$CME_Group, '-', 
                          as.character(data_val$FileGrp))


write_csv(data_train, 'data/csv_cut/data_train.csv')
write_csv(data_val, 'data/csv_cut/data_val.csv')


########################################################

data = read_csv('data/csv_cut/data.csv')

plotCorr(train)

plotAct(data%>%filter(CME_Group == 'GroupA'), day = 265)

plotDist(data%>%filter(CME_Group == 'GroupA'))

temp = data%>%
  filter(CME_Group == 'GroupF', TrueAnswer>100)%>%
  select(TrueAnswer, Activity, Baseline, 10:65)


data%>%filter(CME_Group == 'GroupA')%>%select(FileGrp)



#################### get test data ###################
library(data.table)
library(fst)


test_files <- list.files("../Baseline_Data", pattern="test", recursive = TRUE, full.names = TRUE)
test <- rbindlist(lapply(test_files, function(x) read_fst(x,as.data.table = TRUE, from=1, to=100000)), fill=TRUE)
test


########## groups 1440000, 720000

data = read_fst('../Baseline_Data/GroupA_train_dat.fst', as.data.table = TRUE)
train = truncation(data, split_rate = 0.7, both = TRUE)[[1]]
valid = truncation(data, split_rate = 0.7, both = TRUE)[[2]]

rm(data)
train = train[1:1440000,]
valid = valid[1:720000,]

train['id'] = paste0(train$CME_Group, '-', as.character(train$FileGrp))
valid['id'] = paste0(valid$CME_Group, '-', as.character(valid$FileGrp))

write_csv(train, 'data/group_a/train.csv')
write_csv(valid, 'data/group_a/valid.csv')


test = read_fst('../Baseline_Data/GroupA_test_dat.fst', as.data.table = TRUE)
test$id = paste0(test$CME_Group, '-', as.character(test$FileGrp))
write_csv(test, 'data/group_a/test.csv')



####### Combine group C-I 

train_files = list.files("../Baseline_Data", pattern="train", recursive = TRUE, full.names = TRUE)[3:9]
valid_files <- list.files("../Baseline_Data", pattern="test", recursive = TRUE, full.names = TRUE)[3:9]


train = rbindlist(lapply(train_files, function(x) read_fst(x,as.data.table = TRUE, from=1, to=288000)), fill=TRUE)
valid = rbindlist(lapply(valid_files, function(x) read_fst(x,as.data.table = TRUE, from=1, to=144000)), fill=TRUE)

train$id = paste0(train$CME_Group, '-', as.character(train$FileGrp))
valid$id = paste0(valid$CME_Group, '-', as.character(valid$FileGrp))

write_csv(train, 'data/group_c_to_i/train.csv')
write_csv(valid, 'data/group_c_to_i/valid.csv')





#### plot different group



data = read_csv('data/group_a/train.csv')
data%>%
  filter(FileGrp==5)%>%
  select(TIMESTAMP, TrueAnswer, BL_25_TOP_25_2, BL_50_TOP_25_2, BL_125_TOP_5_2)%>%
  as.data.table()%>%
  dygraph()%>%
  dyRangeSelector()%>%
  dyOptions(useDataTimezone = TRUE)

data%>%
  filter(FileGrp==5)%>%
  select(TIMESTAMP, TrueAnswer, ZeroRoll10, ZeroRoll100, ZeroRoll25)%>%
  as.data.table()%>%
  dygraph()%>%
  dyRangeSelector()%>%
  dyOptions(useDataTimezone = TRUE)

data%>%
  filter(FileGrp==5)%>%
  select(TIMESTAMP, TrueAnswer, RollAvg10, RollAvg50, RollAvg100, RollAvg250)%>%
  as.data.table()%>%
  dygraph()%>%
  dyRangeSelector()%>%
  dyOptions(useDataTimezone = TRUE)



data2 = read_csv('data/group_i/train.csv')
data2%>%
  filter(FileGrp==1)%>%
  select(TIMESTAMP, TrueAnswer, BL_25_TOP_25_2, BL_50_TOP_25_2, BL_125_TOP_5_2)%>%
  as.data.table()%>%
  dygraph()%>%
  dyRangeSelector()%>%
  dyOptions(useDataTimezone = TRUE)





























































































































