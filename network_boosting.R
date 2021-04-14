
#-------------------------------------------------------------------------------------------------------------------

#NETWORK BOOSTING FOR GAMMA MDN
#Construct NN for mixture of Gamma distributions with probabilities, shape and rate depending on regressors

#-------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE AT www.lukaszdelong.pl AND https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346

#-------------------------------------------------------------------------------------------------------------------

if (algorithm_method==1){
  
  model_parameters=inputs_nn%>%
    layer_dense(units=neurons_1,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_2,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_3,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=1+2*no_densities,activation='linear',trainable = TRUE,
                weights=list(array(0,dim=c(neurons_3,1+2*no_densities)),
                             array(0,dim=c(1+2*no_densities))))
  
  model_probabilities_0=model_parameters%>%
    layer_dense(units=no_densities,activation='linear',trainable = FALSE,
                weights=list(array(rbind(diag(no_densities),matrix(0,nrow=1+no_densities,ncol=no_densities)),
                                   dim=c(1+2*no_densities,no_densities)),
                             array(0,dim=c(no_densities))))
  
  model_alpha_0=model_parameters%>%
    layer_dense(units=no_densities,activation='linear',trainable = FALSE,
                weights=list(array(rbind(matrix(0,nrow=no_densities,ncol=no_densities),
                                         diag(no_densities),rep(0,no_densities)),dim=c(1+2*no_densities,no_densities)),
                             array(0,dim=c(no_densities))))
  
  model_beta_0=model_parameters%>%
    layer_dense(units=1,activation='linear',trainable = FALSE,
                weights=list(array(rbind(matrix(0,nrow=2*no_densities,ncol=1),matrix(1,nrow=1,ncol=1)),
                                   dim=c(1+2*no_densities,1)),
                             array(0,dim=c(1))))
}

if (algorithm_method==2){
  
  model_probabilities_0=inputs_nn%>%
    layer_dense(units=neurons_1_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_2_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_3_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=no_densities,activation='linear',trainable = TRUE,
                weights=list(array(0,dim=c(neurons_3_alpha_prob,no_densities)),array(0,dim=c(no_densities))))
  
  model_alpha_0=inputs_nn%>%
    layer_dense(units=neurons_1_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_2_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_3_alpha_prob,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=no_densities,activation='linear',trainable = TRUE,
                weights=list(array(0,dim=c(neurons_3_alpha_prob,no_densities)),array(0,dim=c(no_densities))))
  
  model_beta_0=inputs_nn%>%
    layer_dense(units=neurons_1_beta,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_2_beta,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=neurons_3_beta,activation='tanh',
                kernel_regularizer = regularizer_l2(l = regularization_rate))%>%
    layer_dense(units=1,activation='linear',trainable = TRUE,
                weights=list(array(0,dim=c(neurons_3_beta,1)),array(0,dim=c(1))))
}    

#Probabilities

initial_term=layer_input(shape=c(no_densities))

initial_weight=rep(0,no_densities)
initial_weight[1]<-1

model_probabilities_0_next=model_probabilities_0%>%
  layer_dense(units=1,activation='linear',trainable=FALSE,
              weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))

initial_term_next=initial_term%>%
  layer_dense(units=1,activation='linear',trainable=FALSE,
              weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))%>%
  layer_dense(units=1,activation='linear',trainable=TRUE,use_bias=FALSE,
              weights=list(array(c(1),dim=c(1,1))))

model_probabilities_next=list(model_probabilities_0_next,initial_term_next)%>%layer_concatenate%>%
  layer_dense(units=1,activation='linear',trainable=FALSE,
              weights=list(array(c(1,1),dim=c(2,1)),array(0,dim=c(1))))

model_probabilities=model_probabilities_next

for (i in (2:no_densities)){
  
  initial_weight=rep(0,no_densities)
  initial_weight[i]<-1
  
  model_probabilities_0_next=model_probabilities_0%>%
    layer_dense(units=1,activation='linear',trainable=FALSE,
                weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))
  
  initial_term_next=initial_term%>%
    layer_dense(units=1,activation='linear',trainable=FALSE,
                weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))%>%
    layer_dense(units=1,activation='linear',trainable=TRUE,
                use_bias=FALSE,
                weights=list(array(c(1),dim=c(1,1))))
  
  model_probabilities_next=list(model_probabilities_0_next,initial_term_next)%>%layer_concatenate%>%
    layer_dense(units=1,activation='linear',trainable=FALSE,
                weights=list(array(c(1,1),dim=c(2,1)),array(0,dim=c(1))))
  
  model_probabilities=list(model_probabilities,model_probabilities_next)%>%layer_concatenate
}

model_probabilities=model_probabilities%>%
  layer_dense(units=no_densities,activation='softmax',trainable = FALSE,
              weights=list(array(diag(no_densities),dim=c(no_densities,no_densities)),
                           array(0,dim=c(no_densities))))

model_probabilities<-keras_model(inputs=c(inputs,initial_term),outputs=model_probabilities)

#Alpha

initial_term=layer_input(shape=c(no_densities))

initial_weight=rep(0,no_densities)
initial_weight[1]<-1

model_alpha_0_next=model_alpha_0%>%
  layer_dense(units=1,activation='linear',trainable=FALSE,
              weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))

initial_term_next=initial_term%>%
  layer_dense(units=1,activation='linear',trainable=FALSE,
              weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))%>%
  layer_dense(units=1,activation='linear',trainable=TRUE,use_bias=FALSE,
              weights=list(array(c(1),dim=c(1,1))))

model_alpha_next=list(model_alpha_0_next,initial_term_next)%>%layer_concatenate%>%
  layer_dense(units=1,activation='linear',trainable=FALSE,
              weights=list(array(c(1,1),dim=c(2,1)),array(0,dim=c(1))))

model_alpha=model_alpha_next

for (i in (2:no_densities)){
  
  initial_weight=rep(0,no_densities)
  initial_weight[i]<-1
  
  model_alpha_0_next=model_alpha_0%>%
    layer_dense(units=1,activation='linear',trainable=FALSE,
                weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))
  
  initial_term_next=initial_term%>%
    layer_dense(units=1,activation='linear',trainable=FALSE,
                weights=list(array(initial_weight,dim=c(no_densities,1)),array(0,dim=c(1))))%>%
    layer_dense(units=1,activation='linear',trainable=TRUE,use_bias=FALSE,
                weights=list(array(c(1),dim=c(1,1))))
  
  model_alpha_next=list(model_alpha_0_next,initial_term_next)%>%layer_concatenate%>%
    layer_dense(units=1,activation='linear',trainable=FALSE,
                weights=list(array(c(1,1),dim=c(2,1)),array(0,dim=c(1))))
  
  model_alpha=list(model_alpha,model_alpha_next)%>%layer_concatenate
}

model_alpha=model_alpha%>%
  layer_dense(units=no_densities,activation=k_exp,trainable=FALSE,
              weights=list(array(diag(no_densities),dim=c(no_densities,no_densities)),
                           array(0,dim=c(no_densities))))

model_alpha<-keras_model(inputs=c(inputs,initial_term),outputs=model_alpha)

#Beta

initial_term=layer_input(shape=c(1))

initial_term_next=initial_term%>%
  layer_dense(units=1,activation='linear',trainable=TRUE,use_bias=FALSE,
              weights=list(array(1,dim=c(1,1))))

model_beta=list(model_beta_0,initial_term_next)%>%layer_concatenate%>%
  layer_dense(units=1,activation=k_exp,trainable=FALSE,
              weights=list(array(c(1,1),dim=c(2,1)),array(0,dim=c(1))))

model_beta<-keras_model(inputs=c(inputs,initial_term),outputs=model_beta)

print("Network built")
