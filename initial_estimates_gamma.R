
#-----------------------------------------------------------------------------------------------------------------------

#INITIAL ESTIMATES OF THE PARAMETERS FOR MIXTURE OF GAMMA DISTRIBUTIONS
#Estimates without taking into account the regressors

#-----------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE AT www.lukaszdelong.pl AND https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346

#-----------------------------------------------------------------------------------------------------------------------

if (no_densities==1){
  
  initial_beta=mean(y_data)/var(y_data)
  initial_alpha=mean(y_data)*initial_beta
  
  log_initial_alpha=log(initial_alpha)
  log_initial_beta=log(initial_beta)
}

if (no_densities>1){
 
  y_data_clusters<-Mclust((y_data)^(1/3),G=no_densities)

  initial_prob=y_data_clusters$parameters$pro

  initial_mean=c()
  for (i in (1:no_densities)){
  
    initial_mean=c(initial_mean,
                   sum(y_data_clusters$z[,i]*y_data)/sum(y_data_clusters$z[,i]))
  }

  initial_beta=mean(y_data)/(mean(y_data^2)-sum(initial_mean^2*initial_prob))
  initial_alpha=initial_mean*initial_beta

  log_initial_prob=log(initial_prob)
  log_initial_alpha=log(initial_alpha)
  log_initial_beta=log(initial_beta)
}