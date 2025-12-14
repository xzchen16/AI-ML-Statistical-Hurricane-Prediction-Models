#### This function has as an input Y (counts of tropical cyclones) and X a matrix with possible
#### covariates (rows are years and columns are possible variables). 
#### The options for the analysis are:
#### 1. Variable selection? If TRUE -> use LASSO, if not, use all the variables

library(lars)
library(abind)
library(reshape)
library(MASS)
library(plyr)
library(glmnet)
library(ggplot2)

set.seed(1)

##################################
###### define functions  #########
##################################

getscores<-function(pred,obs,average,Y){
  score          <-  -dpois(obs,pred,log=TRUE)
  clima1         <-  -dpois(obs,average,log=TRUE)
  nphistfit      <-   ecdf(Y)
  clima2        <-   -log(nphistfit(obs)-nphistfit(obs-1))
  H1            <-   clima1-score
  H2            <-   clima2-score
  return(list(H1=H1,H2=H2))
}

runLASSO<-function(Y,X,a,b,c){
  range          <-  c(a:b)[-c]
  cordata        <-  cor(do.call(cbind,X[range,]))
  hc             <-  hclust(dist(cordata))
  varord         <-  (cutree(hc, k = 10))
  sortedlist     <-  sort(abs(cor(cbind(Y[range],X[range,]))[1,-1]))
  sel            <-  c()
  for(i in 1:10){
    allvar       <-  varord[varord==i]
    selected     <-  sort(sortedlist[names(allvar)], decreasing=T)[1]
    sel          <-  c(sel,names(selected))
  }
  data           <-  X[,sel]
  data1          <-  data.frame(cbind(Y[range],
                                      scale(data[range,])))
  names(data1)   <-  c("resp",names(data))
  
  cvout          <-  cv.glmnet(x=data.matrix(data1[,-1]),y=as.vector(data1[,1]),
                               family = "poisson", alpha=1)
  lassout        <-  glmnet(x=data.matrix(data1[,-1]),y=as.vector(data1[,1]), 
                            family = "poisson", lambda=cvout$lambda.min, alpha=1)
  
  mm             <-  apply(data[range,],2,mean)
  sd             <-  apply(data[range,],2,sd)
  newdata1       <-  data.frame(cbind(Y[c],t(apply(data[c,],1,
                                                   function(x)(x-mm)/sd))))
  names(newdata1)<-  c("resp",names(data))
  pred           <-  predict(cvout, s="lambda.min",alpha=1,
                             newx=data.matrix(newdata1[,-1]),type="response")
  res            <-  data.frame(cbind(pred,Y[c]),mean(Y[range]))
  names(res)     <-  c("pred","real","average")
  scores         <-  getscores(res$pred,res$real,res$average,Y[range])
  return(list(res=res,scores=scores,sel=names(which(lassout$beta[,1]!=0))))
}

runGLM<-function(Y,X,a,b,c){
  range          <-  c(a:b)[-c]
  cordata        <-  cor(do.call(cbind,X[range,]))
  hc             <-  hclust(dist(cordata))
  varord         <-  (cutree(hc, k = 10))
  sortedlist     <-  sort(abs(cor(cbind(Y[range],X[range,]))[1,-1]))
  sel            <-  c()
  for(i in 1:10){
    allvar       <-  varord[varord==i]
    selected     <-  sort(sortedlist[names(allvar)], decreasing=T)[1]
    sel          <-  c(sel,names(selected))
  }
  data           <-  X[,sel]
  data1          <-  data.frame(cbind(Y[range],
                                      scale(data[range,])))
  names(data1)   <-  c("resp",names(data))
  lassout        <-  glm(resp~.,family = "poisson",data=data1)
  mm             <-  apply(data[range,],2,mean)
  sd             <-  apply(data[range,],2,sd)
  newdata1       <-  data.frame(cbind(Y[c],t(apply(data[c,],1,
                                                   function(x)(x-mm)/sd))))
  names(newdata1)<-  c("resp",names(data))
  pred           <-  predict(lassout,
                             newdata=data.frame(newdata1[,-1]),type="response")
  res            <-  data.frame(cbind(pred,Y[c]),mean(Y[range]))
  names(res)     <-  c("pred","real","average")
  scores         <-  getscores(res$pred,res$real,res$average,Y[range])
  return(list(res=res,scores=scores,sel=sel))
}

runGLM2<-function(Y,X,a,b,c){
  range          <-  c(a:b)[-c]
  data           <-  X
  data1          <-  data.frame(cbind(Y[range],
                                      scale(data[range,])))
  names(data1)   <-  c("resp",names(data))
  lassout        <-  glm(resp~.,family = "poisson",data=data1)
  mm             <-  apply(data[range,],2,mean)
  sd             <-  apply(data[range,],2,sd)
  newdata1       <-  data.frame(cbind(Y[c],t(apply(data[c,],1,
                                                   function(x)(x-mm)/sd))))
  names(newdata1)<-  c("resp",names(data))
  pred           <-  predict(lassout,newdata=data.frame(newdata1[,-1]),type="response")
  res            <-  data.frame(cbind(pred,Y[c]),mean(Y[range]))
  names(res)     <-  c("pred","real","average")
  scores         <-  getscores(res$pred,res$real,res$average,Y[range])
  return(list(res=res,scores=scores))
}


getmse<-function(pred,obs,average){
  score          <-  mean((obs-pred)^2)
  clima1         <-  mean((obs-average)^2)
  D             <-   clima1-score
  pD            <-   D/clima1
  return(list(D,pD))
}

##################################
###### validation  #########
##################################

validation<-function(Y,X,var.sel=TRUE){
  if(dim(X)[1]!=length(Y)){
    stop('The argument "Y" must have the same length as the number of rows of X')
  }  else {
    N<-dim(X)[1]
  }
  
  ##### Two CV methods:
  ##################################################
  ## SWCV -- for both cluster and LASSO
  ##################################################
  #### Work with w windows of 30 years each + forecasting the 31st. (forecast)
  wdef<-cbind(c(1:(N-30)),c(30:(N-1)))
  wd<-dim(wdef)[1];wd
  ## 44 windows of 30 years each
  if (!var.sel) {
    res1<-lapply(1:wd,function(w){
      try(runGLM(Y,X,wdef[w,1],wdef[w,2],wdef[w,2]+1))})
  } else {
    res1<-lapply(1:wd,function(w){
      try(runLASSO(Y,X,wdef[w,1],wdef[w,2],wdef[w,2]+1))})
  }
  aaa<-lapply(res1,function(x)x[[1]])
  res1proc<-data.frame(t(matrix(unlist(aaa),ncol=44)))
  names(res1proc)<-c("pred","obs","average")
  scores1<-do.call(rbind,lapply(res1,function(x)unlist(x[[2]])))
  
  ## Select random erased value from 1:N (n-fold)
  wall<-cbind(rep(1,N),rep(N,N),c(1:N))
  if (!var.sel) {
    res3<-lapply(1:74,function(w){
      try(runGLM(Y,X,wall[w,1],wall[w,2],wall[w,3]))
    })
  } else {
    res3<-lapply(1:74,function(w){
      try(runLASSO(Y,X,wall[w,1],wall[w,2],wall[w,3]))
    })
  }
  aaa<-lapply(res3,function(x)x[[1]])
  res2proc<-data.frame(t(matrix(unlist(aaa),ncol=74)))
  names(res2proc)<-c("pred","obs","average")
  scores2<-do.call(rbind,lapply(res3,function(x)unlist(x[[2]])))
  
  
  betas<-list(SWF=sort(table(unlist(lapply(res1,function(x)x$sel))), decreasing = T),
              CVnF=sort(table(unlist(lapply(res3,function(x)x$sel))), decreasing = T))
  
  result <- list(tabSWCV=scores1,tabRCV=scores2, resSWCV=res1proc,resRCV=res2proc,betas)
  return(result)
}

### FIT GLM AND LASSO MODEL ###
load(file="independentvariable.Rdata")
load(file="targetvariable.Rdata")

## Now, run it for all combinations: THIS TAKES TIME! 
for(i in 1:9){
  for(j in 1:9){
    Y<-yy0[[i]]
    X<-dataX[[j]]
    a[[i]][[j]]<-validation(Y,X,var.sel=TRUE)
    b[[i]][[j]]<-validation(Y,X,var.sel=FALSE)
  }
  print(i)
  print(j)
}

allresults<-list(a,b)
save(allresults, file="Output\lassoresult.Rdata")
