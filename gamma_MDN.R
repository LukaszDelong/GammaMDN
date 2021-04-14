
#-------------------------------------------------------------------------------------------------------------------

#GAMMA MIXTURE DENSITY NETWORKS

#-------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE ON https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3705225

#--------------------------------------------------------------------------------------------------------------------

#TESTED ON VERSIONS:

#R_3.5.1
#KERAS_2.4.0
#TENSORFLOW_2.2.0

#--------------------------------------------------------------------------------------------------------------------

#LIBRARIES:

library(keras)
library(tensorflow)
library(mclust)

#--------------------------------------------------------------------------------------------------------------------

#DEFINE THE PATH WHERE ALL SCRIPTS HAVE BEEN DOWNLOADED

my_file_path_directory_codes=c("C:/...")

#DEFINE THE PATH WHERE THE FINAL MODEL WILL BE SAVED

#If the EM algorithm is run, the NN from the last EM iteration (for EM_FN) or from the meta model (for EM_NB) is saved

my_file_path_directory=c("C:/...")

#READ THE DATA SET

#Attach names to the columns of the data matrix with the regressors
#Use ordinal coding for the categorical regressors and start from 0
#Apply desired transformations to the continuous regressors
#Put the categorical regressors all together after the continuous regressors in the data matrix

file_path=paste(my_file_path_directory_codes,"synthetic_data_set.rda",sep="")
data_set=data.frame(readRDS(file=file_path))

y_data=as.vector(data_set$y_gamma)
x_data=data.matrix(data_set[, which(colnames(data_set)%in%c("x_1","x_2","x_3"))])

#READ THE MAIN FUNCTIONS TO TRAIN THE MODEL AND READ THE SAVED MODEL

source(paste(my_file_path_directory_codes,"functions_main.r",sep=""))

#-------------------------------------------------------------------------------------------------------------------

#TRAIN THE MODEL

#The parameters to be set:

#trial - the calibration trial/the number of the model being trained
#no_categorical - the number of the categorical regressors in x_data
#validation_split - the proportion of the data set allocated to the validation set
#validation_seed - the seed for the random split of the data set to validation and training
#no_densities - the number of Gamma density components
#neurons - the vector with the number of neurons in three hidden layers, used if algorithm_method="All"
#neurons_alpha_prob - the vector with the number of neurons in three hidden layers for the probabilities and the shape, used if algorithm_method="Separate"
#neurons_beta - the vector with the number of neurons in three hidden layers for the rate, used if algorithm_method="Separate"
#neurons_categorical - the vector with the dimensions of the entity embeddings used for the categorical regressors, used if no_categorical>0
#the dimensions of the entity embeddings are attached to the last no_categorical regressors in x_data in the order as the categorical regressors appear in x_data
#batchsize - the batch size
#learning_rate - the learning rate
#regularization_rate - the regularization rate
#calibration_method - "Direct" for direct optimization or "EM" for the EM algorithm, used if no_densities>1
#algorithm_method - "All" if all parameters are modelled with one NN or "Separate" if the parameters are modelled with three NN
#iterate_em_method - "NB" for Network Boosting of "FN" for Forward Network, used if calibration_method="EM
#no_iterations - the number of EM iterations, used if calibration_method="EM
#no_epochs - the number of epoch for training the NN in each EM iteration, used if calibration_method="EM
#no_epochs_initial_final - the number of epochs for pre-training the meta model with the MSE objective, used if algorithm_method="NB"
#no_epochs_final - the number of epochs for fine-tuning the meta model with the KL divergence objective, used if algorithm_method="NB"
#or the number of epochs for training the model with direct optimization, used if calibration_method="Direct" or no_densities=1
#validation_meta_model - "Y" if the validation set should be used in training the meta model or "N" otherwise, used if algorithm_method="NB"
#noise_scale - the initializer for weights: any positive value for Initializers 1-2, 0 for Initializer 3, -1 for Initializer 4, used if algorithm_method="NB"

new_model=train_mixture_gamma(trial=1,
                              y_data=y_data,
                              x_data=x_data,
                              no_categorical=0,
                              validation_split=0.2,
                              validation_seed=12345,
                              no_densities=3,
                              neurons=c(30,30,30),
                              neurons_alpha_prob=c(10,10,10),
                              neurons_beta=c(10,10,10),
                              neurons_categorical=0,
                              batchsize=1024,
                              learning_rate=0.002,
                              regularization_rate=0,
                              calibration_method="EM",
                              algorithm_method="All",
                              iterate_em_method="NB",
                              no_iterations=50,
                              no_epochs=25,
                              no_epochs_initial_final=50,
                              no_epochs_final=50,
                              validation_meta_model="Y",
                              noise_scale=-1)

#------------------------------------------------------------------------------------------------------------------

#THE CALIBRATION RESULTS

#The values of the log-likelihood in consecutive EM iterations on the training and the validation set

new_model$loss_train_em
new_model$loss_val_em

#The values of the log-likelihood for the final model on the training and the validation set

new_model$loss_train
new_model$loss_val

#The calculation time of consecutive iterations in seconds

new_model$time

#The estimates of the probabilities, the shape and the rate (in that order) from the last EM iteration and from the final model
#For no_densities=1, EM_FN and direct optimization, the final model agrees with the model from the last EM iteration

new_model$em_estimates
new_model$final_estimates

#The quantile residuals

new_model$residuals

#For predictions see below

#Save results

file_path=paste(my_file_path_directory,"em_estimates_",trial,".rda",sep="")
saveRDS(new_model$em_estimates,file=file_path)

#-------------------------------------------------------------------------------------------------------------------

#READ THE SAVED MODEL

new_model=read_mixture_gamma(trial=1,
                             y_data=y_data,
                             x_data=x_data,
                             no_categorical=0,
                             no_densities=3,
                             neurons=c(30,30,30),
                             neurons_alpha_prob=c(10,10,10),
                             neurons_beta=c(10,10,10),
                             neurons_categorical=0,
                             calibration_method="EM",
                             algorithm_method="All",
                             iterate_em_method="NB")

#The NNs for the parameters

model_probabilities_final=new_model$model_probabilities_final
model_alpha_final=new_model$model_alpha_final
model_beta_final=new_model$model_beta_final

#If EM_NB is run, we also need the NN for the parameters on the canonical scale

model_parameters_final_linear=new_model$model_parameters_final_linear

#-------------------------------------------------------------------------------------------------------------------

#PREDICTIONS FOR EM_FN, NO_DENSITIES=1 AND DIRECT OPTIMIZATION

model_probabilities_final%>%predict(x_data)
model_alpha_final%>%predict(x_data)
model_beta_final%>%predict(x_data)

#PREDICTIONS FOR EM_NB

parameters_on_canonical_scale=model_parameters_final_linear%>%predict(x_data)

model_probabilities_final%>%predict(parameters_on_canonical_scale)
model_alpha_final%>%predict(parameters_on_canonical_scale)
model_beta_final%>%predict(parameters_on_canonical_scale)

#The predictions can be done in the same way for the model just trained without reading the model