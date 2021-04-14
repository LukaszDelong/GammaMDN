
#------------------------------------------------------------------------------------------------------------------

#ACTUARIAL APPLICATION: CLAIM AMOUNTS FROM CASDATASETS

#-----------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE ON https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3705225

#----------------------------------------------------------------------------------------------------------------

library(CASdatasets)

#----------------------------------------------------------------------------------------------------------------

#DATA SET

#----------------------------------------------------------------------------------------------------------------

#Read

data(freMPL1)
cas_data=freMPL1
cas_data=cas_data[cas_data$ClaimAmount>0,]

index_remove=which(colnames(cas_data)%in%c("Exposure","RecordBeg","RecordEnd",
                                           "ClaimInd"))
cas_data=cas_data[,-index_remove]

#Continuous regressors 1

index_cont_regressors_1=which(colnames(cas_data)%in%c("LicAge","DrivAge","BonusMalus"))
continuous_regressors_1=cas_data[,index_cont_regressors_1]

#Continuous regressors 2

index_cont_regressors_2=which(colnames(cas_data)%in%c("LicAge","DrivAge","BonusMalus",
                                                      "VehAge","VehPrice","VehMaxSpeed",
                                                      "RiskVar"))
continuous_regressors_2=cas_data[,index_cont_regressors_2]

levels(continuous_regressors_2$VehAge)<-c(levels(continuous_regressors_2$VehAge),6,7,8)
continuous_regressors_2$VehAge[which(continuous_regressors_2$VehAge==c("6-7"))]<-6
continuous_regressors_2$VehAge[which(continuous_regressors_2$VehAge==c("8-9"))]<-7
continuous_regressors_2$VehAge[which(continuous_regressors_2$VehAge==c("10+"))]<-8
continuous_regressors_2$VehAge<-as.numeric(paste(continuous_regressors_2$VehAge))

continuous_regressors_2$VehPrice<-match(continuous_regressors_2$VehPrice,
                                        sort(unique(continuous_regressors_2$VehPrice)))-1

continuous_regressors_2$VehMaxSpeed<-match(continuous_regressors_2$VehMaxSpeed,
                                           sort(unique(continuous_regressors_2$VehMaxSpeed)))-1

continuous_regressors_2$RiskVar<-match(continuous_regressors_2$RiskVar,
                                       sort(unique(continuous_regressors_2$RiskVar)))-1

#Continuous regressors - log and min_max transformations

continuous_regressors_1$BonusMalus<-log(continuous_regressors_1$BonusMalus)
continuous_regressors_2$BonusMalus<-log(continuous_regressors_2$BonusMalus)

continuous_regressors_1<-min_max_scaler_full(continuous_regressors_1)
continuous_regressors_2<-min_max_scaler_full(continuous_regressors_2)

#Binary regressors

index_binary_regressors=which(colnames(cas_data)%in%c("Gender","MariStat",
                                                      "HasKmLimit","VehEnergy"))

binary_regressors=cas_data[,index_binary_regressors]
binary_regressors$Gender<-ifelse(binary_regressors$Gender=="Female",0,1)
binary_regressors$MariStat<-ifelse(binary_regressors$MariStat=="Alone",0,1)
binary_regressors$VehEnergy<-ifelse(binary_regressors$VehEnergy=="regular",0,1)

#Categorical regressors all

categorical_regressors=cas_data[,-c(index_binary_regressors,index_cont_regressors_1,
                                    which(colnames(cas_data)=="ClaimAmount"))]

for (i in c(1:ncol(categorical_regressors))){
  categorical_regressors[,i]<-match(categorical_regressors[,i],sort(unique(categorical_regressors[,i])))-1
}

#-----------------------------------------------------------------------------------------------------------------

#MERGE SMALL CLASSES

#-----------------------------------------------------------------------------------------------------------------

#VehEngine

categorical_regressors$VehEngine[which(categorical_regressors$VehEngine==2)]<-1
dim_VehEngine<-length(unique(categorical_regressors$VehEngine))
categorical_regressors$VehEngine<-match(categorical_regressors$VehEngine,sort(unique(categorical_regressors$VehEngine)))-1

#SocioCat

categorical_regressors$SocioCateg[which(categorical_regressors$SocioCateg%in%c(1,2,3,4,5,9))]<-1
categorical_regressors$SocioCateg[which(categorical_regressors$SocioCateg==12)]<-13
categorical_regressors$SocioCateg[which(categorical_regressors$SocioCateg==17)]<-7
categorical_regressors$SocioCateg[which(categorical_regressors$SocioCateg==20)]<-19
categorical_regressors$SocioCateg[which(categorical_regressors$SocioCateg%in%c(22,23,24))]<-7
dim_SocioCateg=length(unique(categorical_regressors$SocioCateg))
categorical_regressors$SocioCateg<-match(categorical_regressors$SocioCateg,sort(unique(categorical_regressors$SocioCateg)))-1

#VehPrice

categorical_regressors$VehPrice[which(categorical_regressors$VehPrice==1)]<-2
categorical_regressors$VehPrice[which(categorical_regressors$VehPrice%in%c(22,23,24,25))]<-22
dim_VehPrice=length(unique(categorical_regressors$VehPrice))
categorical_regressors$VehPrice<-match(categorical_regressors$VehPrice,sort(unique(categorical_regressors$VehPrice)))-1

#--------------------------------------------------------------------------------------------------------------------

#FINAL DATA SETS

#1 - the data set with categorical regressors
#2 - the data set with the ordered categorical regressors modelled as continuous variables

cat_regressors_1=categorical_regressors
cat_regressors_2=categorical_regressors[,
                 -which(colnames(categorical_regressors)%in%c("VehAge","VehPrice","VehMaxSpeed","RiskVar"))]

x_data_1=data.matrix(cbind(binary_regressors,
                           continuous_regressors_1,
                           cat_regressors_1))

x_data_2=data.matrix(cbind(binary_regressors,
                           continuous_regressors_2,
                           cat_regressors_2))

x_data=x_data_2
y_data=as.vector(cas_data$ClaimAmount)