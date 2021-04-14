
#--------------------------------------------------------------------------------------------------------------------

#READ MODEL AFTER EM_FN OR DIRECT OPTIMIZATION

#--------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE AT www.lukaszdelong.pl AND https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346

#--------------------------------------------------------------------------------------------------------------------

#Functions

source(paste(my_file_path_directory_codes,"functions.r",sep=""),local=TRUE)
source(paste(my_file_path_directory_codes,"functions_gamma.r",sep=""),local=TRUE)

#Initial estimates of the parameters

log_initial_prob=0
log_initial_alpha=0
log_initial_beta=0

#Input

source(paste(my_file_path_directory_codes,"network_inputs.r",sep=""),local=TRUE)

#Network

source(paste(my_file_path_directory_codes,"network_forward.r",sep=""),local=TRUE)

#Likelihood function

inputs<-layer_input(shape=c(2+no_regressors+no_densities))

y_observations<-layer_lambda(inputs,f=function(x){x[,1,drop=FALSE]})  
log_y_observations<-layer_lambda(inputs,f=function(x){x[,2,drop=FALSE]}) 
x_regressors<-layer_lambda(inputs,f=function(x){x[,3:(2+no_regressors),drop=FALSE]})
current_prob<-layer_lambda(inputs,f=function(x){x[,(3+no_regressors):(2+no_regressors+no_densities),drop=FALSE]})

predict_probabilities<-model_probabilities(x_regressors)
log_probabilities<-log_transform_probabilities(predict_probabilities)

predict_alpha<-model_alpha(x_regressors)
lgamma_alpha<-lgamma_transform_alpha(predict_alpha)

predict_beta<-model_beta(x_regressors)
log_beta<-log_transform_beta(predict_beta)

if (calibration_method==0){
  
  loglik<-layer_lambda(list(current_prob,log_probabilities,predict_alpha,lgamma_alpha,predict_beta,log_beta,
                            y_observations,log_y_observations),
                       function_gamma_true)
}

if (calibration_method==1){
  
  loglik<-layer_lambda(list(current_prob,log_probabilities,predict_alpha,lgamma_alpha,predict_beta,log_beta,
                            y_observations,log_y_observations),
                       function_gamma_em)
}

#Model to optimize

model_optimize<-keras_model(inputs=inputs,outputs=loglik)
model_optimize %>% compile(optimizer=optimizer_nadam(lr=learning_rate),loss=max_loss)


#Read the model

file_path=file.path(paste(my_file_path_directory,"gamma_mixture_model_",trial,".h5",sep=""))
load_model_weights_hdf5(model_optimize,file_path)

#Output

model_probabilities_final=model_probabilities
model_alpha_final=model_alpha
model_beta_final=model_beta
