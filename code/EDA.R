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



plotCorr(train)

plotAct(data%>%filter(CME_Group == 'GroupF'), day = 193)

plotDist(data%>%filter(CME_Group == 'GroupA'))

temp = data%>%
  filter(CME_Group == 'GroupF', TrueAnswer>100)%>%
  select(TrueAnswer, Activity, Baseline, 10:65)






#################### get test data ###################
library(data.table)
library(fst)
col_names <- c("TIMESTAMP", "TrueAnswer", "Activity", "Lift", "Imp", "CNET",
               "Date", "Baseline", "Simulation_ID", "BL_25_TOP_10_1", "BL_25_TOP_20_1",
               "BL_25_TOP_25_1", "BL_50_TOP_5_1", "BL_50_TOP_10_1", "BL_50_TOP_20_1",
               "BL_50_TOP_25_1", "BL_50_TOP_35_1", "BL_75_TOP_5_1", "BL_125_TOP_5_1",
               "BL_25_TOP_10_2", "BL_25_TOP_20_2", "BL_25_TOP_25_2", "BL_50_TOP_5_2",
               "BL_50_TOP_10_2", "BL_50_TOP_20_2", "BL_50_TOP_25_2", "BL_50_TOP_35_2",
               "BL_75_TOP_5_2", "BL_125_TOP_5_2", "BL_25_TOP_10_3", "BL_25_TOP_20_3", 
               "BL_25_TOP_25_3", "BL_50_TOP_5_3", "BL_50_TOP_10_3", "BL_50_TOP_20_3",
               "BL_50_TOP_25_3", "BL_50_TOP_35_3", "BL_75_TOP_5_3", "BL_125_TOP_5_3", 
               "PercentZero", "ZeroRoll5", "ZeroRoll10", "ZeroRoll25", "ZeroRoll35", 
               "ZeroRoll50", "ZeroRoll80", "ZeroRoll100", "ZeroRoll200", "ZeroRoll300", 
               "RollAvg5", "RollAvg10", "RollAvg25", "RollAvg50", "RollAvg75",
               "RollAvg100", "RollAvg150", "RollAvg200", "RollAvg250", "RollAvg300",
               "CME1", "CME2", "CME3", "CME_Group", "FileGrp")


test_files <- list.files("../Baseline_Data", pattern="test", recursive = TRUE, full.names = TRUE)
test <- rbindlist(lapply(test_files, function(x) read_fst(x,as.data.table = TRUE, from=1, to=100000)), fill=TRUE)
test




test = bind_rows(read_fst('data/fst_test/GroupA_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupB_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupC_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupD_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupE_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupF_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupG_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupH_test_dat.fst', from = 1, to = 10000),
                 read_fst('data/fst_test/GroupI_test_dat.fst', from = 1, to = 10000))































































































































