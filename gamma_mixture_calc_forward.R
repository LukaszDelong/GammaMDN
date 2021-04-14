
#-------------------------------------------------------------------------------------------------------------------

#CALIBRATION OF GAMMA MIXTURE DENSITY NETWORK WITH EM_FN

#-------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE AT www.lukaszdelong.pl AND https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346

#-------------------------------------------------------------------------------------------------------------------

#MODEL TO OPTIMIZE

calculation_time=c()
loss_train_em=c()
loss_val_em=c()
  
max_no_iterations=ifelse(calibration_method==0,1,no_iterations)
no_epochs_calibration_method=ifelse(calibration_method==0,no_epochs_final,no_epochs)
patience_calibration_method=ifelse(calibration_method==0,25,5)

#Functions

source(paste(my_file_path_directory_codes,"functions.r",sep=""),local=TRUE)
source(paste(my_file_path_directory_codes,"functions_gamma.r",sep=""),local=TRUE)

#Initial estimates of the parameters

source(paste(my_file_path_directory_codes,"initial_estimates_gamma.r",sep=""),local=TRUE)

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

CBs<-callback_early_stopping(monitor="val_loss", min_delta=0,
                             patience=patience_calibration_method, verbose=0, mode=c("min"),
                             restore_best_weights=TRUE)

print("Model built")

#------------------------------------------------------------------------------------------------------------------

#EM ALGORITHM

for (k in (1:max_no_iterations)){  
  
  if (k==max_no_iterations){
   
    file_path=file.path(paste(my_file_path_directory,"gamma_mixture_model_",trial,".h5",sep=""))
    
    CBs<-list(callback_early_stopping(monitor="val_loss", min_delta=0,
                                      patience=patience_calibration_method, verbose=0, mode=c("min"),
                                      restore_best_weights=TRUE),
              callback_model_checkpoint(file_path, monitor="val_loss", verbose=0,  
                                        save_best_only=TRUE, save_weights_only=TRUE, save_freq=NULL))   
  }
  
  start.time <- Sys.time()
  
  #Initial estimates of the parameters
  
  if (k==1){
    
    posterior_probability=y_data_clusters$z
    
    }else{
    
    initial_prob=probabilities_estimates
    initial_beta=beta_estimates
    initial_alpha=alpha_estimates
      
    posterior_density=c()
    for (i in (1:no_densities)){
      
      posterior_density=cbind(posterior_density,
                              initial_prob[,i]*dgamma(y_data,shape=initial_alpha[,i],rate=initial_beta))
    }
    posterior_probability=posterior_density/apply(posterior_density,1,sum)
  }  
  
  #Data input
  
  set.seed(validation_seed)
  validation_index=sample(c(1:no_observations),validation_split*no_observations,replace = FALSE)
  set.seed(NULL)

  X_all=cbind(y_data,log(y_data),x_data,posterior_probability)  
  Y_all=rep(0,no_observations)

  X_train=X_all[-validation_index,]
  Y_train=Y_all[-validation_index]
  
  X_val=X_all[validation_index,]
  Y_val=Y_all[validation_index]

  #Optimize

  model_optimize%>%fit(X_train,Y_train,epochs=no_epochs_calibration_method,batch_size=batchsize,
                       verbose=0,
                       validation_data=list(X_val,Y_val),callbacks=CBs)
  
  #Predict

  probabilities_estimates<-model_probabilities%>%predict(x_data)
  alpha_estimates<-model_alpha%>%predict(x_data)
  beta_estimates<-model_beta%>%predict(x_data)
  
  end.time<-Sys.time()
  
  #True log-likelihood in EM step

  loss_em=0
  for (i in (1:no_densities)){
  loss_em=(loss_em+
           probabilities_estimates[,i]*dgamma(y_data,shape=alpha_estimates[,i],rate=beta_estimates))
  }
  loss_em=log(loss_em)

  loss_train_em=c(loss_train_em,mean(loss_em[-validation_index]))
  loss_val_em=c(loss_val_em,mean(loss_em[validation_index]))
  
  calculation_time=c(calculation_time,difftime(end.time,start.time,units="secs"))
  
  print(rbind(loss_train_em,loss_val_em))

}

#------------------------------------------------------------------------------------------------------------------

#FINAL NETWORKS

model_probabilities_final=model_probabilities
model_alpha_final=model_alpha
model_beta_final=model_beta

probabilities_estimates_final=data.matrix(probabilities_estimates)
alpha_estimates_final=data.matrix(alpha_estimates)
beta_estimates_final=as.vector(beta_estimates) 

#Final true log-likelihood

loss_true=0
for (i in (1:no_densities)){
  loss_true=(loss_true+
             probabilities_estimates_final[,i]*dgamma(y_data,
                                                      shape=alpha_estimates_final[,i],
                                                      rate=beta_estimates_final))
}
loss_true=log(loss_true)

loss_train_true=mean(loss_true[-validation_index])
loss_val_true=mean(loss_true[validation_index])

print(c(loss_train_true,loss_val_true))

#Quantile residuals

quantile_residuals=0
for (i in (1:no_densities)){
  quantile_residuals=(quantile_residuals+
                      probabilities_estimates_final[,i]*pgamma(y_data,
                                                               shape=alpha_estimates_final[,i],
                                                               rate=beta_estimates_final))
}
quantile_residuals=qnorm(quantile_residuals,mean=0,sd=1)

