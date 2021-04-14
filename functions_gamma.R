
#-------------------------------------------------------------------------------------------------------------------

#FUNCTIONS RELATED TO CALIBRATION OF GAMMA MDNS

#-------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE AT www.lukaszdelong.pl AND https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346

#-------------------------------------------------------------------------------------------------------------------

#Min and Max loss

max_loss<-custom_metric("max",function(y_true,y_pred){k_mean(y_true-y_pred)})

min_loss<-custom_metric("min",function(y_true,y_pred){k_mean(y_pred-y_true)})

#Gamma log-likelihoods

function_gamma_0<-function(args){
  c(predict_alpha,lgamma_alpha,predict_beta,log_beta,y_observations,log_y_observations) %<-% args
  
  loglik<-(predict_alpha*log_y_observations-log_y_observations+predict_alpha*log_beta
              -predict_beta*y_observations-lgamma_alpha)
} 

function_gamma_em<-function(args){
  c(current_prob,log_probabilities,predict_alpha,lgamma_alpha,predict_beta,log_beta,
    y_observations,log_y_observations) %<-% args
    
  loglik<-(
    k_sum(layer_multiply(list(current_prob,log_probabilities)),axis=2,keepdims=TRUE)  
    +k_sum(layer_multiply(list(current_prob,predict_alpha)),axis=2,keepdims=TRUE)*log_y_observations
    +k_sum(layer_multiply(list(current_prob,predict_alpha)),axis=2,keepdims=TRUE)*log_beta
    -predict_beta*y_observations-log_y_observations
    -k_sum(layer_multiply(list(current_prob,lgamma_alpha)),axis=2,keepdims=TRUE)
    )
}

function_gamma_true<-function(args){
  c(current_prob,log_probabilities,predict_alpha,lgamma_alpha,predict_beta,log_beta,
    y_observations,log_y_observations) %<-% args
    
  loglik<-(
    k_log(
      k_sum(
        k_exp(
          log_probabilities+predict_alpha*log_beta-lgamma_alpha+predict_alpha*log_y_observations
          -log_y_observations-predict_beta*y_observations),
        axis=2,keepdims=TRUE))
    )
}

#Kullback-Leibler divergence

function_kl<-function(args){
  c(probabilities_fitted,log_probabilities_fitted,
    alpha_fitted,lgamma_alpha_fitted,beta_fitted,log_beta_fitted,
    digamma_alpha_fitted,expected_fitted,
    log_probabilities,predict_alpha,lgamma_alpha,predict_beta,log_beta) %<-% args
  
  kl_divergence<-(
    (k_sum(layer_multiply(list(probabilities_fitted,log_probabilities_fitted)),axis=2,keepdims=TRUE)  
     -k_sum(layer_multiply(list(probabilities_fitted,log_probabilities)),axis=2,keepdims=TRUE)) 
    
    +(k_sum(layer_multiply(list(probabilities_fitted,predict_alpha)),axis=2,keepdims=TRUE)*log_beta_fitted
      -k_sum(layer_multiply(list(probabilities_fitted,predict_alpha)),axis=2,keepdims=TRUE)*log_beta)
    
    +(k_sum(layer_multiply(list(probabilities_fitted,lgamma_alpha)),axis=2,keepdims=TRUE)
      -k_sum(layer_multiply(list(probabilities_fitted,lgamma_alpha_fitted)),axis=2,keepdims=TRUE))
    
    +(k_sum(layer_multiply(list(probabilities_fitted,digamma_alpha_fitted,alpha_fitted)),axis=2,keepdims=TRUE)
      -k_sum(layer_multiply(list(probabilities_fitted,digamma_alpha_fitted,predict_alpha)),axis=2,keepdims=TRUE))
    
    +k_sum(layer_multiply(list(probabilities_fitted,expected_fitted)),axis=2,keepdims=TRUE)*(predict_beta-beta_fitted)
    )
}  

#Transformations of parameters for NN

inputs=layer_input(shape=c(no_densities))

log_transform_probabilities=inputs%>%
  layer_dense(units=no_densities,activation=k_log,trainable=FALSE,
              weights=list(array(diag(no_densities),dim=c(no_densities,no_densities)),
                           array(0,dim=c(no_densities))))

log_transform_probabilities<-keras_model(inputs=inputs,outputs=log_transform_probabilities)

inputs=layer_input(shape=c(1))

log_transform_beta=inputs%>%
  layer_dense(units=1,activation=k_log,trainable=FALSE,
              weights=list(array(1,dim=c(1,1)),array(0,dim=c(1))))

log_transform_beta<-keras_model(inputs=inputs,outputs=log_transform_beta)

inputs=layer_input(shape=c(no_densities))

lgamma_transform_alpha=inputs%>%
  layer_dense(units=no_densities,activation=tf$math$lgamma,trainable=FALSE,
              weights=list(array(diag(no_densities),dim=c(no_densities,no_densities)),array(0,dim=c(no_densities))))

lgamma_transform_alpha<-keras_model(inputs=inputs,outputs=lgamma_transform_alpha)

inputs=layer_input(shape=c(no_densities))

digamma_transform_alpha=inputs%>%
  layer_dense(units=no_densities,activation=tf$math$digamma,trainable=FALSE,
              weights=list(array(diag(no_densities),dim=c(no_densities,no_densities)),array(0,dim=c(no_densities))))

digamma_transform_alpha<-keras_model(inputs=inputs,outputs=digamma_transform_alpha)

