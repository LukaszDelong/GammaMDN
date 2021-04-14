
#-------------------------------------------------------------------------------------------------------------------

#CALIBRATION OF CLASSICAL GAMMA DISTRIBUTION (ONE GAMMA COMPONENT)

#-------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE ON https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3705225

#-------------------------------------------------------------------------------------------------------------------

#MODEL TO OPTIMIZE

#Functions
  
source(paste(my_file_path_directory_codes,"functions.r",sep=""),local=TRUE)
source(paste(my_file_path_directory_codes,"functions_gamma.r",sep=""),local=TRUE)
  
#Initial estimates of the parameters
  
source(paste(my_file_path_directory_codes,"initial_estimates_gamma.r",sep=""),local=TRUE)
  
#Input
  
source(paste(my_file_path_directory_codes,"network_inputs.r",sep=""),local=TRUE)
  
#Network
  
source(paste(my_file_path_directory_codes,"network_classical.r",sep=""),local=TRUE)
  
#Likelihood function
  
inputs<-layer_input(shape=c(2+no_regressors))
  
y_observations<-layer_lambda(inputs,f=function(x){x[,1,drop=FALSE]})  
log_y_observations<-layer_lambda(inputs,f=function(x){x[,2,drop=FALSE]}) 
x_regressors<-layer_lambda(inputs,f=function(x){x[,3:(2+no_regressors),drop=FALSE]})
  
predict_alpha<-model_alpha(x_regressors)
lgamma_alpha<-lgamma_transform_alpha(predict_alpha)
  
predict_beta<-model_beta(x_regressors)
log_beta<-log_transform_beta(predict_beta)
  
loglik<-layer_lambda(list(predict_alpha,lgamma_alpha,
                          predict_beta,log_beta,y_observations,log_y_observations),
                    function_gamma_0)
  
#Model to optimize
  
model_optimize<-keras_model(inputs=inputs,outputs=loglik)  
model_optimize%>%compile(optimizer=optimizer_nadam(lr=learning_rate),loss=max_loss)
  
file_path=file.path(paste(my_file_path_directory,"gamma_mixture_model_",trial,".h5",sep=""))

CBs<-list(callback_early_stopping(monitor="val_loss", min_delta=0,
                                  patience=25, verbose=0, mode=c("min"),
                                  restore_best_weights=TRUE),
          callback_model_checkpoint(file_path, monitor="val_loss", verbose=0,  
                                    save_best_only=TRUE, save_weights_only=TRUE, save_freq=NULL))
  
print("Model built")
  
#-------------------------------------------------------------------------------------------------------------------

#OPTIMIZATION

start.time <- Sys.time()
  
#Data input

set.seed(validation_seed)
validation_index=sample(c(1:no_observations),validation_split*no_observations,replace = FALSE)
set.seed(NULL)
  
X_all=cbind(y_data,log(y_data),x_data)  
Y_all=rep(0,no_observations)
  
X_train=X_all[-validation_index,]
Y_train=Y_all[-validation_index]
  
X_val=X_all[validation_index,]
Y_val=Y_all[validation_index]
  
#Optimize
  
model_optimize%>%fit(X_train,Y_train,epochs=no_epochs_final,batch_size=batchsize,
                     verbose=0,
                     validation_data=list(X_val,Y_val),callbacks=CBs)

#Predict
  
alpha_estimates<-model_alpha%>%predict(x_data)
beta_estimates<-model_beta%>%predict(x_data)
probabilities_estimates=rep(1,length(y_data))
  
end.time <- Sys.time()

#Define final models

model_alpha_final=model_alpha
model_beta_final=model_beta

alpha_estimates_final=alpha_estimates
beta_estimates_final=beta_estimates
probabilities_estimates_final=probabilities_estimates

#Final true log-likelihood
  
loss_true=log(dgamma(y_data,shape=alpha_estimates_final,rate=beta_estimates_final))
  
loss_train_true=mean(loss_true[-validation_index])
loss_val_true=mean(loss_true[validation_index])

loss_train_em=loss_train_true
loss_val_em=loss_val_true  

calculation_time=difftime(end.time,start.time,units="secs")
  
print(c(loss_train_true,loss_val_true))
  
#Quantile residuals
  
quantile_residuals=pgamma(y_data,shape=alpha_estimates_final,rate=beta_estimates_final)
quantile_residuals=qnorm(quantile_residuals,mean=0,sd=1)
  
