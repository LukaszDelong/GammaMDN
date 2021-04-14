
#-------------------------------------------------------------------------------------------------------------------

#CALIBRATION OF GAMMA MIXTURE DENSITY NETWORK WITH EM_NB

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

source(paste(my_file_path_directory_codes,"network_boosting.r",sep=""),local=TRUE)

#Likelihood function

inputs<-layer_input(shape=c(3+3*no_densities+no_regressors))

y_observations<-layer_lambda(inputs,f=function(x){x[,1,drop=FALSE]})  
log_y_observations<-layer_lambda(inputs,f=function(x){x[,2,drop=FALSE]}) 
x_regressors<-layer_lambda(inputs,f=function(x){x[,3:(2+no_regressors),drop=FALSE]})
current_prob<-layer_lambda(inputs,f=function(x){x[,(3+no_regressors):(2+no_regressors+no_densities),drop=FALSE]})
prob_term<-layer_lambda(inputs,f=function(x){x[,(3+no_regressors+no_densities):(2+no_regressors+2*no_densities),drop=FALSE]})
alpha_term<-layer_lambda(inputs,f=function(x){x[,(3+no_regressors+2*no_densities):(2+no_regressors+3*no_densities),drop=FALSE]})
beta_term<-layer_lambda(inputs,f=function(x){x[,(3+no_regressors+3*no_densities),drop=FALSE]})

predict_probabilities<-model_probabilities(list(x_regressors,prob_term))
log_probabilities<-log_transform_probabilities(predict_probabilities)

predict_alpha<-model_alpha(list(x_regressors,alpha_term))
lgamma_alpha<-lgamma_transform_alpha(predict_alpha)

predict_beta<-model_beta(list(x_regressors,beta_term))
log_beta<-log_transform_beta(predict_beta)

loglik<-layer_lambda(list(current_prob,log_probabilities,predict_alpha,lgamma_alpha,predict_beta,log_beta,
                          y_observations,log_y_observations),
                     function_gamma_em)

#Model to optimize

model_optimize<-keras_model(inputs=inputs,outputs=loglik)
model_optimize%>%compile(optimizer=optimizer_nadam(lr=learning_rate),loss=max_loss)

CBs<-callback_early_stopping(monitor="val_loss", min_delta=0,
                             patience=patience_calibration_method, verbose=0, mode=c("min"),
                             restore_best_weights=TRUE)

print("Model built")

#Weights for initialization

model_optimize_weights_base=get_weights(model_optimize)
model_optimize_weights=model_optimize_weights_base
model_optimize_weights_last=model_optimize_weights_base

index_update=c()
for (i in c(1:length(lengths(model_optimize_weights_base)))){
  if (sum(ifelse(unlist(model_optimize_weights_base[[i]])%in%c(0,1),0,1))>0){
    index_update=c(index_update,i)
  }
}

#--------------------------------------------------------------------------------------------------------------------

#EM ALGORITHM

for (k in (1:max_no_iterations)){  
  
  start.time <- Sys.time()
  
  #Initial estimates of the parameters
  
  if (k==1){
    
    initial_prob=matrix(rep(initial_prob,each=no_observations),nrow=no_observations)
    initial_alpha=matrix(rep(initial_alpha,each=no_observations),nrow=no_observations)
    initial_beta=matrix(rep(initial_beta,each=no_observations),nrow=no_observations)
    
    posterior_probability=y_data_clusters$z
    
    }else{
    
    initial_prob=probabilities_estimates
    initial_alpha=alpha_estimates  
    initial_beta=beta_estimates
    
    posterior_density=c()
    for (i in (1:no_densities)){
      
      posterior_density=cbind(posterior_density,
                              initial_prob[,i]*dgamma(y_data,shape=initial_alpha[,i],rate=initial_beta))
    }
    posterior_probability=posterior_density/apply(posterior_density,1,sum)
  }  
  
  log_initial_prob=log(initial_prob)
  log_initial_alpha=log(initial_alpha)
  log_initial_beta=log(initial_beta)
  
  #Data input
  
  set.seed(validation_seed)
  validation_index=sample(c(1:no_observations),validation_split*no_observations,replace = FALSE)
  set.seed(NULL)

  X_all=cbind(y_data,log(y_data),x_data,posterior_probability,
              log_initial_prob,log_initial_alpha,log_initial_beta)  
  Y_all=rep(0,no_observations)

  X_train=X_all[-validation_index,]
  Y_train=Y_all[-validation_index]
  
  X_val=X_all[validation_index,]
  Y_val=Y_all[validation_index]

  #Weights for initialization
  
  if (algorithm_method==1){
    
    for (i in c(1:(3+no_categorical))){
      
      if (noise_scale==0){
        noise_scale_range=max(abs(model_optimize_weights_base[[index_update[i]]]),abs(model_optimize_weights_last[[index_update[i]]]))
      }
      
      if (noise_scale>0){
        noise_scale_range=noise_scale*max(abs(model_optimize_weights_base[[index_update[i]]]))
      }
      
      if (noise_scale<0){
        
        model_optimize_weights[[index_update[i]]]<-model_optimize_weights_last[[index_update[i]]]
        model_optimize_weights[[index_update[3+no_categorical+i]]]<-model_optimize_weights_last[[index_update[i]]]
        model_optimize_weights[[index_update[6+2*no_categorical+i]]]<-model_optimize_weights_last[[index_update[i]]] 
        
        if (i>no_categorical){
          model_optimize_weights[[index_update[i]+1]]<-model_optimize_weights_last[[index_update[i]+1]]
          model_optimize_weights[[index_update[3+no_categorical+i]+1]]<-model_optimize_weights_last[[index_update[i]+1]]
          model_optimize_weights[[index_update[6+2*no_categorical+i]+1]]<-model_optimize_weights_last[[index_update[i]+1]]
        }
        
      }else{
        
        model_optimize_weights[[index_update[i]]]<-matrix(
          runif(nrow(model_optimize_weights[[index_update[i]]])*ncol(model_optimize_weights[[index_update[i]]]),
                -noise_scale_range,noise_scale_range),
          nrow=nrow(model_optimize_weights[[index_update[i]]]),ncol=ncol(model_optimize_weights[[index_update[i]]]))
        
        model_optimize_weights[[index_update[3+no_categorical+i]]]<-model_optimize_weights[[index_update[i]]]
        model_optimize_weights[[index_update[6+2*no_categorical+i]]]<-model_optimize_weights[[index_update[i]]]
      }
    }
  }
  
  if (algorithm_method==2){
    
    for (i in index_update){
      
      if (noise_scale==0){
        noise_scale_range=max(abs(model_optimize_weights_base[[i]]),abs(model_optimize_weights_last[[i]]))
      }
      
      if (noise_scale>0){
        noise_scale_range=noise_scale*max(abs(model_optimize_weights_base[[i]]))
      }
      
      if (noise_scale<0){
        
        index_after_categorical=c((no_categorical+1):(3+no_categorical),
                                  (4+2*no_categorical):(6+2*no_categorical),
                                  (7+3*no_categorical):(9+3*no_categorical))
        index_after_categorical=index_update[index_after_categorical]
        
        model_optimize_weights[[i]]<-model_optimize_weights_last[[i]]  
        if (i%in%index_after_categorical){
        model_optimize_weights[[i+1]]<-model_optimize_weights_last[[i+1]]
        }
        
      }else{
        
        model_optimize_weights[[i]]<-matrix(
          runif(nrow(model_optimize_weights[[i]])*ncol(model_optimize_weights[[i]]),
                -noise_scale_range,noise_scale_range),
          nrow=nrow(model_optimize_weights[[i]]),ncol=ncol(model_optimize_weights[[i]]))
      }
    }
  }   
  
  set_weights(model_optimize,model_optimize_weights)
  
  #Optimize
  
  model_optimize%>% fit(X_train,Y_train,epochs=no_epochs_calibration_method,batch_size=batchsize,
                             verbose=0,
                             validation_data=list(X_val,Y_val),callbacks=CBs)
  model_optimize_weights_last=get_weights(model_optimize)
  
  #Predict

  probabilities_estimates<-model_probabilities%>%predict(list(x_data,log_initial_prob))
  alpha_estimates<-model_alpha%>%predict(list(x_data,log_initial_alpha))
  beta_estimates<-model_beta%>%predict(list(x_data,log_initial_beta))
  
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

#--------------------------------------------------------------------------------------------------------------------

#META MODEL

#Normalized log-parameters

log_probabilities_estimates=log(probabilities_estimates)
min_log_probabilities_estimates=apply(log_probabilities_estimates,2,min)
max_log_probabilities_estimates=apply(log_probabilities_estimates,2,max)
log_probabilities_estimates_scaled=min_max_scaler(log_probabilities_estimates,min_log_probabilities_estimates,max_log_probabilities_estimates)
  
log_alpha_estimates=log(alpha_estimates)
min_log_alpha_estimates=apply(log_alpha_estimates,2,min)
max_log_alpha_estimates=apply(log_alpha_estimates,2,max)
log_alpha_estimates_scaled=min_max_scaler(log_alpha_estimates,min_log_alpha_estimates,max_log_alpha_estimates)
  
log_beta_estimates=log(beta_estimates)
min_log_beta_estimates=min(log_beta_estimates)
max_log_beta_estimates=max(log_beta_estimates)
log_beta_estimates_scaled=min_max_scaler(log_beta_estimates,min_log_beta_estimates,max_log_beta_estimates)
  
#PRE-TRAINING

#Model to optimize - MSE
  
if (no_epochs_initial_final>0){
  
source(paste(my_file_path_directory_codes,"network_inputs.r",sep=""),local=TRUE)
  
source(paste(my_file_path_directory_codes,"network_meta.r",sep=""),local=TRUE)
  
model_parameters_final_linear%>%compile(optimizer=optimizer_nadam(lr=learning_rate),loss ='mse')
   
CBs<-callback_early_stopping(monitor="val_loss", min_delta=0,
                             patience=25, verbose=0, mode=c("min"),
                             restore_best_weights=TRUE)
  
print("Initial meta model built")
  
#Data input
  
parameters_train=cbind(log_probabilities_estimates_scaled,
                       log_alpha_estimates_scaled,
                       log_beta_estimates_scaled)
  
parameters_val=parameters_train
  
#Optimize
  
model_parameters_final_linear%>%fit(x_data,parameters_train,
                                    epochs=no_epochs_initial_final,batch_size=batchsize,
                                    verbose=0,
                                    validation_data=list(x_data,parameters_val),
                                    callbacks=CBs)
  
initial_weights_for_replication=get_weights(model_parameters_final_linear)
}
  
#FINE-TUNING

#Model to optimize - KL divergence
  
source(paste(my_file_path_directory_codes,"network_inputs.r",sep=""),local=TRUE)
  
source(paste(my_file_path_directory_codes,"network_meta.r",sep=""),local=TRUE)
   
if (no_epochs_initial_final>0){
  set_weights(model_parameters_final_linear,initial_weights_for_replication)
}
  
#Transformations of normalized log-parameters to original scale
  
intercept_nn=min_log_probabilities_estimates+(max_log_probabilities_estimates-min_log_probabilities_estimates)/2
slope_nn=(max_log_probabilities_estimates-min_log_probabilities_estimates)/2
  
inputs=layer_input(shape=c(1+2*no_densities))
  
model_probabilities_final=inputs%>%
    layer_dense(units=no_densities,activation='softmax',trainable=FALSE,
                weights=list(array(rbind(diag(slope_nn),matrix(0,nrow=1+no_densities,ncol=no_densities)),
                                   dim=c(1+2*no_densities,no_densities)),
                             array(intercept_nn,dim=c(no_densities))))
  
model_probabilities_final<-keras_model(inputs=inputs,outputs=model_probabilities_final)
  
intercept_nn=min_log_alpha_estimates+(max_log_alpha_estimates-min_log_alpha_estimates)/2
slope_nn=(max_log_alpha_estimates-min_log_alpha_estimates)/2
  
model_alpha_final=inputs%>%
    layer_dense(units=no_densities,activation=k_exp,trainable=FALSE,
                weights=list(array(rbind(matrix(0,nrow=no_densities,ncol=no_densities),
                                         diag(slope_nn),rep(0,no_densities)),dim=c(1+2*no_densities,no_densities)),
                             array(intercept_nn,dim=c(no_densities))))
  
model_alpha_final<-keras_model(inputs=inputs,outputs=model_alpha_final)
  
intercept_nn=min_log_beta_estimates+(max_log_beta_estimates-min_log_beta_estimates)/2
slope_nn=(max_log_beta_estimates-min_log_beta_estimates)/2
  
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
  
file_path=file.path(paste(my_file_path_directory,"gamma_mixture_model_",trial,".h5",sep=""))
  
CBs<-list(callback_early_stopping(monitor="val_loss", min_delta=0,
                                patience=25, verbose=0, mode=c("min"),
                                restore_best_weights=TRUE),
         callback_model_checkpoint(file_path, monitor="val_loss", verbose=0,  
                                save_best_only=TRUE, save_weights_only=TRUE, save_freq=NULL))
  
print("Meta model built")

#Data input

expected_estimates=data.matrix(alpha_estimates/as.vector(beta_estimates))

X_all=cbind(x_data,
            probabilities_estimates,alpha_estimates,beta_estimates,expected_estimates)
Y_all=rep(0,no_observations)
  
if (validation_meta_model==1){
  
  X_train=X_all[-validation_index,]
  Y_train=Y_all[-validation_index]
  
  X_val=X_all[validation_index,]
  Y_val=Y_all[validation_index]
}

if (validation_meta_model==0){
  
  X_train=X_all
  Y_train=Y_all
  
  X_val=X_all
  Y_val=Y_all
}

#Optimize

model_optimize%>%fit(X_train,Y_train,epochs=no_epochs_final,batch_size=batchsize,
                     verbose=0,
                     validation_data=list(X_val,Y_val),
                     callbacks=CBs)
  
#Predict
  
parameters_estimates_final<-model_parameters_final_linear%>%predict(x_data)
  
probabilities_estimates_final<-parameters_estimates_final[,1:no_densities]
probabilities_estimates_final=min_max_scaler_back(probabilities_estimates_final,
                                                  min_log_probabilities_estimates,max_log_probabilities_estimates)
probabilities_estimates_final<-softmax_transform(probabilities_estimates_final)
  
alpha_estimates_final<-parameters_estimates_final[,(no_densities+1):(2*no_densities)]
alpha_estimates_final=min_max_scaler_back(alpha_estimates_final,
                                          min_log_alpha_estimates,max_log_alpha_estimates)
alpha_estimates_final<-exp(alpha_estimates_final)
  
beta_estimates_final<-parameters_estimates_final[,(1+2*no_densities)]
beta_estimates_final=min_max_scaler_back(beta_estimates_final,
                                         min_log_beta_estimates,max_log_beta_estimates)
beta_estimates_final<-exp(beta_estimates_final)

#Final true log-likelihood

loss_true=0
for (i in (1:no_densities)){
  loss_true=(loss_true+
      probabilities_estimates_final[,i]*dgamma(y_data,shape=alpha_estimates_final[,i],rate=beta_estimates_final))
}
loss_true=log(loss_true)

loss_train_true=mean(loss_true[-validation_index])
loss_val_true=mean(loss_true[validation_index])

print(c(loss_train_true,loss_val_true))

#Quantile residuals

quantile_residuals=0
for (i in (1:no_densities)){
  quantile_residuals=(quantile_residuals+
      probabilities_estimates_final[,i]*pgamma(y_data,shape=alpha_estimates_final[,i],rate=beta_estimates_final))
}
quantile_residuals=qnorm(quantile_residuals,mean=0,sd=1)
