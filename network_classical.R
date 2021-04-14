
#---------------------------------------------------------------------------------------------------------------------

#NETWORK FOR CLASSICAL GAMMA DISTRIBUTION
#Construct NN for classical Gamma distribution with shape and rate depending on regressors

#---------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE ON https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3705225

#---------------------------------------------------------------------------------------------------------------------

if (algorithm_method==1){
  
  model_parameters=inputs_nn%>%
    layer_dense(units=neurons_1,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_2,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_3,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=2,activation='linear',trainable = TRUE,
                weights=list(array(0,dim=c(neurons_3,2)),
                             array(c(log_initial_alpha,log_initial_beta),dim=c(2))))
  
  model_alpha=model_parameters%>%
    layer_dense(units=1,activation=k_exp,trainable = FALSE,
                weights=list(array(rbind(c(1),c(0)),dim=c(2,1)),array(0,dim=c(1))))
  
  model_alpha<-keras_model(inputs=inputs,outputs=model_alpha)
  
  model_beta=model_parameters%>%
    layer_dense(units=1,activation=k_exp,trainable = FALSE,
                weights=list(array(rbind(c(0),c(1)),dim=c(2,1)),array(0,dim=c(1))))
  
  model_beta<-keras_model(inputs=inputs,outputs=model_beta)
}

if (algorithm_method==2){
  
  model_alpha=inputs_nn%>%
    layer_dense(units=neurons_1_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_2_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_3_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=1,activation=k_exp,trainable = TRUE,
                weights=list(array(0,dim=c(neurons_3_alpha_prob,1)),
                             array(log_initial_alpha,dim=c(1))))
  
  model_alpha<-keras_model(inputs=inputs,outputs=model_alpha)
  
  model_beta=inputs_nn%>%
    layer_dense(units=neurons_1_beta,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_2_beta,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_3_beta,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=1,activation=k_exp,trainable = TRUE,
                weights=list(array(0,dim=c(neurons_3_beta,1)),
                             array(log_initial_beta,dim=c(1))))
  
  model_beta<-keras_model(inputs=inputs,outputs=model_beta)  
}    

print("Network built")