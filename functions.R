
#-------------------------------------------------------------------------------------------------------------------

#FUNCTIONS

#-------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE ON https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3705225

#-------------------------------------------------------------------------------------------------------------------

#min max scalers

min_max_scaler_back<-function(my_matrix,my_min,my_max){
  
  if (is.vector(my_matrix)==TRUE){
    
    z=(my_matrix+1)*(my_max-my_min)/2+my_min
    
  }else{
    
    z=(((my_matrix+1)*
          (matrix(rep(my_max,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE)
           -matrix(rep(my_min,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE)))/2
       +matrix(rep(my_min,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE))
  }
  
  return(z)
} 

min_max_scaler<-function(my_matrix,my_min,my_max){
  
  if (is.vector(my_matrix)==TRUE){
    
    z=2*(my_matrix-my_min)/(my_max-my_min)-1   
    
  }else{
    
    z=(2*((my_matrix-matrix(rep(my_min,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE))/
            (matrix(rep(my_max,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE)
             -matrix(rep(my_min,nrow(my_matrix)),nrow(my_matrix),byrow=TRUE)))-1)
  }
  
  return(z)
} 

min_max_scaler_full<-function(my_matrix){
  
  if (is.vector(my_matrix)==TRUE){
    
    my_min=min(my_matrix)
    my_max=max(my_matrix)
    
  }else{
    
   my_min=apply(my_matrix,2,min)
   my_max=apply(my_matrix,2,max)
  }
  
  z=min_max_scaler(my_matrix,my_min,my_max)
    
  return(z)
} 

#softmax

softmax_transform<-function(my_matrix){
  
  if (is.vector(my_matrix)==TRUE){
    
    z=exp(my_matrix)/sum(exp(my_matrix))
    
  }else{
    
    z=exp(my_matrix)/apply(exp(my_matrix),1,sum)
  }
  
  return(z)
} 

