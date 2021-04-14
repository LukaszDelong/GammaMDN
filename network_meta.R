
#--------------------------------------------------------------------------------------------------------------------

#META MODEL FOR GAMMA MDN
#Construct NN to replicate the model achieved with EM_NB

#--------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE AT www.lukaszdelong.pl AND https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346

#-------------------------------------------------------------------------------------------------------------------

  if (algorithm_method==1){
    
    model_parameters_final_linear=inputs_nn%>%
      layer_dense(units=neurons_1,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_2,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_3,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=1+2*no_densities,activation='linear')
    
    model_parameters_final_linear<-keras_model(inputs=inputs,outputs=model_parameters_final_linear)
  }
  
  if (algorithm_method==2){
    
    model_probabilities_final_linear=inputs_nn%>%
      layer_dense(units=neurons_1_alpha_prob,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_2_alpha_prob,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_3_alpha_prob,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=no_densities,activation='linear')
    
    model_alpha_final_linear=inputs_nn%>%
      layer_dense(units=neurons_1_alpha_prob,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_2_alpha_prob,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_3_alpha_prob,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=no_densities,activation='linear')
    
    model_beta_final_linear=inputs_nn%>%
      layer_dense(units=neurons_1_beta,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_2_beta,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=neurons_3_beta,activation='tanh',
                  kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
      layer_dense(units=1,activation='linear')
    
    model_parameters_final_linear=list(model_probabilities_final_linear,
                                       model_alpha_final_linear,
                                       model_beta_final_linear)%>%layer_concatenate
    
    model_parameters_final_linear<-keras_model(inputs=inputs,outputs=model_parameters_final_linear)
  }    
  
  print("Meta network built")
