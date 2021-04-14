
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

#------------------------------------------------------------------------------------------------------------------

#MAIN FUNCTIONS TO TRAIN THE MODEL AND READ THE SAVED MODEL

#------------------------------------------------------------------------------------------------------------------

#TRAIN THE MODEL

train_mixture_gamma<-function(trial,
                              y_data,
                              x_data,
                              no_categorical,
                              validation_split,
                              validation_seed,
                              no_densities,
                              neurons,
                              neurons_alpha_prob,
                              neurons_beta,
                              neurons_categorical,
                              batchsize,
                              learning_rate,
                              regularization_rate,
                              calibration_method,
                              algorithm_method,
                              iterate_em_method,
                              no_iterations,
                              no_epochs,
                              no_epochs_initial_final,
                              no_epochs_final,
                              validation_meta_model,
                              noise_scale){
  
  #Set the algorithm
  
  if (algorithm_method=="All"){algorithm_method=1}
  if (algorithm_method=="Separate"){algorithm_method=2}
  
  if (calibration_method=="Direct"){calibration_method=0}
  if (calibration_method=="EM"){calibration_method=1}
  
  if (iterate_em_method=="NB"){iterate_em_method=0}
  if (iterate_em_method=="FN"){iterate_em_method=1}
  
  if (validation_meta_model=="Y"){validation_meta_model=1}
  if (validation_meta_model=="N"){validation_meta_model=0}
  
  #Define the neurons
  
  if (algorithm_method==1){
    
    neurons_1=neurons[1]
    neurons_2=neurons[2]
    neurons_3=neurons[3]
  }
  
  if (algorithm_method==2){
    
    neurons_1_alpha_prob=neurons_alpha_prob[1]
    neurons_2_alpha_prob=neurons_alpha_prob[2]
    neurons_3_alpha_prob=neurons_alpha_prob[3]
  
    neurons_1_beta=neurons_beta[1]
    neurons_2_beta=neurons_beta[2]
    neurons_3_beta=neurons_beta[3]
  }
  
  if (no_categorical>0){
    
    for (i in c((ncol(x_data)-no_categorical+1):ncol(x_data))){
      assign(paste("neurons",colnames(x_data)[i],sep="_"),neurons_categorical[i-ncol(x_data)+no_categorical])  
    }
  }
  
  #Run the scripts
  
  if (no_densities==1){
    source(paste(my_file_path_directory_codes,"gamma_mixture_calc_classical.r",sep=""),local=TRUE)}
  
  if (no_densities>1){
    
    if (calibration_method==0){
      source(paste(my_file_path_directory_codes,"gamma_mixture_calc_forward.r",sep=""),local=TRUE)}
      
    if (calibration_method==1){
      
      if (iterate_em_method==0){
        source(paste(my_file_path_directory_codes,"gamma_mixture_calc_boosting.r",sep=""),local=TRUE)}
        
      if (iterate_em_method==1){
        source(paste(my_file_path_directory_codes,"gamma_mixture_calc_forward.r",sep=""),local=TRUE)}
    }
  }
      
  return_list_estimates=list("loss_train_em"=loss_train_em,
                             "loss_val_em"=loss_val_em,
                             "loss_train"=loss_train_true,
                             "loss_val"=loss_val_true,
                             "time"=calculation_time,
                             "em_estimates"=cbind(probabilities_estimates,
                                                  alpha_estimates,
                                                  beta_estimates),
                             "final_estimates"=cbind(probabilities_estimates_final,
                                                     alpha_estimates_final,
                                                     beta_estimates_final),
                             "residuals"=quantile_residuals)
  
  if (no_densities==1){
  
    return_list_models=list("model_alpha_final"=model_alpha_final,
                            "model_beta_final"=model_beta_final)
  }
  
  if (no_densities>1){
    
    if (calibration_method==0){
      
      return_list_models=list("model_probabilities_final"=model_probabilities_final,
                              "model_alpha_final"=model_alpha_final,
                              "model_beta_final"=model_beta_final)
    }
    
    if (calibration_method==1){
      
      if (iterate_em_method==0){
        
        return_list_models=list("model_parameters_final_linear"=model_parameters_final_linear,
                                "model_probabilities_final"=model_probabilities_final,
                                "model_alpha_final"=model_alpha_final,
                                "model_beta_final"=model_beta_final)
      }
      
      if (iterate_em_method==1){
        
        return_list_models=list("model_probabilities_final"=model_probabilities_final,
                                "model_alpha_final"=model_alpha_final,
                                "model_beta_final"=model_beta_final)
      }
    }
  }
  
  return(c(return_list_estimates,return_list_models))   

}
  
#-----------------------------------------------------------------------------------------------------------------

#READ THE MODEL

read_mixture_gamma<-function(trial,
                             y_data,
                             x_data,
                             no_categorical,
                             no_densities,
                             neurons,
                             neurons_alpha_prob,
                             neurons_beta,
                             neurons_categorical,
                             calibration_method,
                             algorithm_method,
                             iterate_em_method){
  
  #Default parameters for the network reconstruction
  
  learning_rate=0.002
  regularization_rate=0
  
  #Set the algorithm
  
  if (algorithm_method=="All"){algorithm_method=1}
  if (algorithm_method=="Separate"){algorithm_method=2}
  
  if (calibration_method=="Direct"){calibration_method=0}
  if (calibration_method=="EM"){calibration_method=1}
  
  if (iterate_em_method=="NB"){iterate_em_method=0}
  if (iterate_em_method=="FN"){iterate_em_method=1}
  
  #Define the neurons
  
  if (algorithm_method==1){
    
    neurons_1=neurons[1]
    neurons_2=neurons[2]
    neurons_3=neurons[3]
  }
  
  if (algorithm_method==2){
    
    neurons_1_alpha_prob=neurons_alpha_prob[1]
    neurons_2_alpha_prob=neurons_alpha_prob[2]
    neurons_3_alpha_prob=neurons_alpha_prob[3]
    
    neurons_1_beta=neurons_beta[1]
    neurons_2_beta=neurons_beta[2]
    neurons_3_beta=neurons_beta[3]
  }
  
  if (no_categorical>0){
    
    for (i in c((ncol(x_data)-no_categorical+1):ncol(x_data))){
      assign(paste("neurons",colnames(x_data)[i],sep="_"),neurons_categorical[i-ncol(x_data)+no_categorical])  
    }
  }
  
  #Run the scripts

  if (no_densities==1){
    
    source(paste(my_file_path_directory_codes,"read_model_classical.r",sep=""),local=TRUE)
    
    return_list=list("model_alpha_final"=model_alpha_final,
                     "model_beta_final"=model_beta_final)
  }
    
  if (no_densities>1){
    
    if (calibration_method==0){
    
      source(paste(my_file_path_directory_codes,"read_model_fn.r",sep=""),local=TRUE)
    
      return_list=list("model_probabilities_final"=model_probabilities_final,
                       "model_alpha_final"=model_alpha_final,
                       "model_beta_final"=model_beta_final)
      }
    
    if (calibration_method==1){
      
        if (iterate_em_method==0){
        
          source(paste(my_file_path_directory_codes,"read_model_nb.r",sep=""),local=TRUE)
        
          return_list=list("model_parameters_final_linear"=model_parameters_final_linear,
                           "model_probabilities_final"=model_probabilities_final,
                           "model_alpha_final"=model_alpha_final,
                           "model_beta_final"=model_beta_final)
          }
      
        if (iterate_em_method==1){
        
          source(paste(my_file_path_directory_codes,"read_model_fn.r",sep=""),local=TRUE)
        
          return_list=list("model_probabilities_final"=model_probabilities_final,
                           "model_alpha_final"=model_alpha_final,
                           "model_beta_final"=model_beta_final)
          }
    }
  }
  
  return(return_list)   
}
