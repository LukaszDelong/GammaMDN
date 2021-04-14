
#-------------------------------------------------------------------------------------------------------------------

#READ META MODEL AFTER EM_NB

#-------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE ON https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3705225

#-------------------------------------------------------------------------------------------------------------------

#Functions

source(paste(my_file_path_directory_codes,"functions.r",sep=""),local=TRUE)
source(paste(my_file_path_directory_codes,"functions_gamma.r",sep=""),local=TRUE)

#Input

source(paste(my_file_path_directory_codes,"network_inputs.r",sep=""),local=TRUE)

#Network

source(paste(my_file_path_directory_codes,"network_meta.r",sep=""),local=TRUE)

#Transformations of normalized log-parameters to original scale

intercept_nn=rep(0,no_densities)
slope_nn=rep(0,no_densities)

inputs=layer_input(shape=c(1+2*no_densities))

model_probabilities_final=inputs%>%
  layer_dense(units=no_densities,activation='softmax',trainable=FALSE,
              weights=list(array(rbind(diag(slope_nn),matrix(0,nrow=1+no_densities,ncol=no_densities)),
                                 dim=c(1+2*no_densities,no_densities)),
                           array(intercept_nn,dim=c(no_densities))))

model_probabilities_final<-keras_model(inputs=inputs,outputs=model_probabilities_final)

intercept_nn=rep(0,no_densities)
slope_nn=rep(0,no_densities)

model_alpha_final=inputs%>%
  layer_dense(units=no_densities,activation=k_exp,trainable=FALSE,
              weights=list(array(rbind(matrix(0,nrow=no_densities,ncol=no_densities),
                                       diag(slope_nn),rep(0,no_densities)),dim=c(1+2*no_densities,no_densities)),
                           array(intercept_nn,dim=c(no_densities))))

model_alpha_final<-keras_model(inputs=inputs,outputs=model_alpha_final)

intercept_nn=0
slope_nn=0

model_beta_final=inputs%>%
  layer_dense(units=1,activation=k_exp,trainable=FALSE,
              weights=list(array(rbind(matrix(0,nrow=2*no_densities,ncol=1),matrix(slope_nn,nrow=1,ncol=1)),
                                 dim=c(1+2*no_densities,1)),
                           array(intercept_nn,dim=c(1))))

model_beta_final<-keras_model(inputs=inputs,outputs=model_beta_final)

#KL divergence loss function

inputs <- layer_input(shape = c(3*no_densities+no_regressors+1))

x_regressors<-layer_lambda(inputs,f=function(x){x[,1:(no_regressors),drop=FALSE]})
probabilities_fitted<-layer_lambda(inputs,f=function(x){x[,(no_regressors+1):(no_regressors+no_densities),drop=FALSE]})
alpha_fitted<-layer_lambda(inputs,f=function(x){x[,(no_regressors+no_densities+1):(no_regressors+2*no_densities),drop=FALSE]})
beta_fitted<-layer_lambda(inputs,f=function(x){x[,(no_regressors+2*no_densities+1),drop=FALSE]})
expected_fitted<-layer_lambda(inputs,f=function(x){x[,(no_regressors+2*no_densities+2):(no_regressors+3*no_densities+1),drop=FALSE]})

predict_parameters_final<-model_parameters_final_linear(x_regressors)

predict_probabilities<-model_probabilities_final(predict_parameters_final)
log_probabilities<-log_transform_probabilities(predict_probabilities)

log_probabilities_fitted<-log_transform_probabilities(probabilities_fitted)

predict_alpha<-model_alpha_final(predict_parameters_final)
lgamma_alpha<-lgamma_transform_alpha(predict_alpha)

lgamma_alpha_fitted<-lgamma_transform_alpha(alpha_fitted)
digamma_alpha_fitted<-digamma_transform_alpha(alpha_fitted)

predict_beta<-model_beta_final(predict_parameters_final)
log_beta<-log_transform_beta(predict_beta)

log_beta_fitted<-log_transform_beta(beta_fitted)

kl_divergence<-layer_lambda(list(probabilities_fitted,log_probabilities_fitted,
                                 alpha_fitted,lgamma_alpha_fitted,beta_fitted,log_beta_fitted,
                                 digamma_alpha_fitted,expected_fitted,
                                 log_probabilities,predict_alpha,lgamma_alpha,predict_beta,log_beta),
                            function_kl)

#Compile

model_optimize<-keras_model(inputs=inputs,outputs=kl_divergence)  
model_optimize%>%compile(optimizer=optimizer_nadam(lr=learning_rate),loss=min_loss)

#Read the model

file_path=file.path(paste(my_file_path_directory,"gamma_mixture_model_",trial,".h5",sep=""))
load_model_weights_hdf5(model_optimize,file_path)
