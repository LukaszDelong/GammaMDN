
#---------------------------------------------------------------------------------------------------------------------

#INPUT FOR NN
#Construct input for NN from the regressors

#---------------------------------------------------------------------------------------------------------------------

#AUTHOR: £UKASZ DELONG (SGH WARSAW SCHOOL OF ECONOMICS)
#e-mail: lukasz.delong@sgh.waw.pl
#DATE: 14TH APRIL 2021 (VERSION 1)

#BASED ON THE PAPER: 
#£. DELONG, M. LINDHOLM, M.W. WUTHRICH, 2021, 
#GAMMA MIXTURE DENSITY NETWORKS AND THEIR APPLICATION TO MODELLING INSURANCE CLAIM AMOUNTS
#AVAILABLE AT www.lukaszdelong.pl AND https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=2255346

#---------------------------------------------------------------------------------------------------------------------

no_observations=length(y_data)
no_regressors=ncol(x_data)

if (no_categorical==0){
  
  inputs=layer_input(shape=c(no_regressors),dtype="float32") 
  inputs_nn=inputs
}

if (no_categorical>0){
  
  cat_regressors=data.matrix(x_data[,c((ncol(x_data)-no_categorical+1):ncol(x_data))])
  
  if (no_categorical==1){
  colnames(cat_regressors)=colnames(x_data)[ncol(x_data)]
  }
  
  for (i in c(1:ncol(cat_regressors))){
    assign(paste("dim",colnames(cat_regressors)[i],sep="_"),length(unique(cat_regressors[,i])))
  }
  
  inputs=layer_input(shape=c(no_regressors),dtype="float32") 

  initial_weight=rbind(diag(1,no_regressors-no_categorical),
                     matrix(0,nrow=no_categorical,ncol=no_regressors-no_categorical))

  inputs_nn=inputs%>%
            layer_dense(units=no_regressors-no_categorical,activation='linear',trainable=FALSE,
                  weights=list(array(initial_weight,dim=c(no_regressors,no_regressors-no_categorical)),
                               array(0,dim=c(no_regressors-no_categorical))))

  for (i in c(1:no_categorical)){
  
    initial_weight=rep(0,no_regressors)
    initial_weight[no_regressors-no_categorical+i]<-1
  
    inputs_cat=inputs%>%
               layer_dense(units=1,activation='linear',trainable=FALSE,
                           weights=list(array(initial_weight,dim=c(no_regressors,1)),
                                        array(0,dim=c(1))))%>%
    layer_embedding(input_dim=get(paste("dim",colnames(cat_regressors)[i],sep="_")),
                    output_dim=get(paste("neurons",colnames(cat_regressors)[i],sep="_")),
                    input_length=1)%>%layer_flatten
  
    inputs_nn=list(inputs_nn,inputs_cat)%>%layer_concatenate
  }
}
