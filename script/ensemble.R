library(xgboost)
library(lightgbm)
library(caret)
library(randomForest)
library(lightgbm)
library(Matrix)
library(mxnet)

print("loading data")
tr<-read.csv("./data/trdata.csv",sep = ",")
te<-read.csv("./data/tedata.csv",sep = ",")
print("making class label")
trclass<-make.names(factor(tr$class,levels = c(0,1,2),labels=c(0,1,2)))

trdata<-subset(tr,select = -(class))
trlabel<-tr$class

tr<-subset(tr,select = -c(class))
te<-te

print("defining fit control")
fitControl <- trainControl(
  method = "none"
)

print("defining classifierfunction")

fxgb<-function(datatrain){
  xgbparams <- list(booster="gbtree",
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    nthread=4,
                    num_class=3,
                    eta = .02,
                    gamma = 1,
                    max_depth = 4,
                    min_child_weight = 1,
                    subsample = .7,
                    colsample_bytree = .5
  )
  
  xgb<-xgb.train(data = datatrain,
                 params = xgbparams,
                 print_every_n = 300,
                 nrounds = 4050
  )
  
  return (xgb)
}

flgbm<-function(datatrain){
  lgbmparams <- list(objective = "multiclass", 
                     metric = "multi_logloss",
                     num_class = 3
  )
  
  lgbm<-lgb.train( lgbmparams,
                   datatrain,
                   nrounds = 960, 
                   learning_rate=0.01,
                   num_leaves=101,
                   feature_fraction=0.9,
                   bagging_fraction=0.9,
                   bagging_freq=20,
                   is_sparse=FALSE,
                   boosting="gbdt", 
                   verbose = 0, 
                   num_threads=4
  )  
  return(lgbm)
}


frfFit<-function(class,data){
  rfgrid = expand.grid(.mtry=c(19))
  rfFit <- train(y=class,x=data, 
                 method = "rf", 
                 trControl = fitControl,
                 tuneGrid = rfgrid,
                 ntree = 300,
                 metric="logLoss",
                 verbose = TRUE
  ) 
  return (rfFit)
}

fsvmFit<-function(class,data){
  svmgrid <- expand.grid(C = c(0.01))
  svmFit <- train(y=class,x=data, 
                  method = "svmLinear",
                  trControl = fitControl,
                  tuneGrid = svmgrid,
                 # preProcess = c("center","scale"),
                  metric="logLoss"
  )
  return(svmFit)
}

flogisticFit<-function(class,data){
  logistgrid <- expand.grid(nIter= c(31))
  logistFit <- train(y=class,x=data, 
                     method = "LogitBoost", 
                     trControl = fitControl,
                     tuneGrid = logistgrid,
                     metric="logLoss"
  )
  return(logistFit)
}

fldaFit<-function(class,data){
  ldagrid <- expand.grid(dimen= c(3))
  ldaFit <- train(y=class,x=data, 
                  method = "lda2", 
                  trControl = fitControl,
                  tuneGrid = ldagrid,
                  metric="logLoss"
  )
  return(ldaFit)
}

fglmnetFit<-function(class,data){
  glmnetgrid <- expand.grid(.alpha=c(0.01), .lambda=c(0.01))
  glmnetFit <- train(y=class,x=data,  
                     method = "glmnet", 
                     trControl = fitControl,
                     tuneGrid = glmnetgrid,
                     metric="logLoss"
  )
  return (glmnetFit)
}

fgbmFit<-function(class,data){
  gbmgrid <- expand.grid(.n.trees=320, .interaction.depth=8, .shrinkage=0.04, .n.minobsinnode=3)
  gbmFit <- train(y=class,x=data,  
                  method = "gbm", 
                  trControl = fitControl,
                  tuneGrid = gbmgrid,
                  metric="logLoss"
  )
  return(gbmFit)
}

fc50Fit<-function(class,data){
  c50grid =expand.grid(.trials=100, .model="tree", .winnow=TRUE)
  c50Fit<- train(y=make.names(trlabel),x=trdata,
                 method="C5.0",
                 tuneGrid=c50grid,
                 trControl=fitControl,
                 metric="logLoss",
                 verbose=FALSE)
  return(c50Fit)
}


flgbmdt<-function(datatrain){
  lgbmdtparams <- list(objective = "multiclass",
                       metric = "multi_logloss",
                       num_class = 3
  )
  
  lgbmdt<-lgb.train(lgbmdtparams,
                    datatrain,
                    nrounds = 350,
                    learning_rate=0.1,
                    num_leaves=95,
                    feature_fraction=0.9
                    ,bagging_fraction=0.9
                    ,bagging_freq=20
                    ,is_sparse=FALSE,
                    boosting="dart",
                    verbose = 0,
                    num_threads=4
  )
  return(lgbmdt)
}

flgbmr<-function(datatrain){
  lgbmrparams <- list(objective = "regression_l2",
                       metric = "l2"
                       # num_class = 3
  )
  
  lgbmr<-lgb.train(lgbmrparams,
                    datatrain,
                    nrounds =171,
                    learning_rate=0.1,
                    num_leaves=95,
                    boosting="gbdt",
                    verbose = 0,
                    num_threads=4
  )
  return(lgbmr)
}

fadabagFit<-function(class,data){
  adabaggrid =expand.grid(.mfinal=30, .maxdepth=11)
  adabagFit<- train(y=make.names(trlabel),x=trdata,
                    method="AdaBag",
                    tuneGrid=adabaggrid,
                    trControl=fitControl,
                    metric="logLoss",
                    verbose=FALSE)
  return(adabagFit)
}


fxgbdt<-function(datatrain){
  xgbdtparams <- list(booster="dart",
                      objective="multi:softprob",
                      eval_metric="mlogloss",
                      nthread=4,
                      num_class=3,
                      eta = .04,
                      gamma = 1,
                      max_depth = 4,
                      min_child_weight = 1,
                      subsample = .7,
                      colsample_bytree = .5
  )
  
  xgbdt<-xgb.train(data = datatrain,
                   params = xgbdtparams,
                   print_every_n = 300,
                   nrounds = 700
  )
  
  return (xgbdt)
}

print("stage 1")

fitting<-function(iscv=TRUE,trdat,trlab,trclas,tedat){
  
  idxseq <- c(1:nrow(trdat))
  print("data format processing")
  
  trspar <- Matrix(as.matrix(trdat), sparse=TRUE)
  tespar <- Matrix(as.matrix(tedat), sparse=TRUE)
  
  dtrgbm <- lgb.Dataset(trspar, label=trlab)
  dtegbm <- lgb.Dataset(tespar)
  
  dtrxgb <- xgb.DMatrix(data=trspar, label=trlab)
  dtexgb <- xgb.DMatrix(data=tespar)
  
  if(iscv){
    
    trpred_xgb<-matrix(nrow = nrow(dtrxgb), ncol = 3)
    trpred_lgbm<-matrix(nrow = nrow(dtrgbm), ncol = 3)
    trpred_rf<-matrix(nrow = nrow(trdat), ncol = 3)
    trpred_svm<-matrix(nrow = nrow(trdat), ncol = 3)
    trpred_logist<-matrix(nrow = nrow(trdat), ncol = 3)
    trpred_lda<-matrix(nrow = nrow(trdat), ncol = 3)
    trpred_glmnet<-matrix(nrow = nrow(trdat), ncol = 3)
    trpred_gbm<-matrix(nrow = nrow(trdat), ncol = 3)
    trpred_c50<-matrix(nrow = nrow(trdat), ncol = 3)
    trpred_lgbmdt<-matrix(nrow = nrow(dtrgbm), ncol = 3)
    trpred_lgbmr<-matrix(nrow = nrow(dtrgbm), ncol = 1)
    trpred_xgbdt<-matrix(nrow = nrow(dtrxgb), ncol = 3)
    trpred_adabag<-matrix(nrow = nrow(trdat), ncol = 3)
    
    tepred_xgb<-matrix(0,nrow = nrow(dtexgb), ncol = 3)
    tepred_lgbm<-matrix(0,nrow = nrow(dtegbm), ncol = 3)
    tepred_rf<-matrix(0,nrow = nrow(tedat), ncol = 3)
    tepred_svm<-matrix(nrow = nrow(tedat), ncol = 3)
    tepred_logist<-matrix(0,nrow = nrow(tedat), ncol = 3)
    tepred_lda<-matrix(nrow = nrow(tedat), ncol = 3)
    tepred_glmnet<-matrix(nrow = nrow(tedat), ncol = 3)
    tepred_gbm<-matrix(0,nrow = nrow(tedat), ncol = 3)
    tepred_c50<-matrix(0,nrow = nrow(tedat), ncol = 3)
    tepred_lgbmdt<-matrix(0,nrow = nrow(dtegbm), ncol = 3)
    tepred_lgbmr<-matrix(nrow = nrow(dtegbm), ncol = 1)
    tepred_xgbdt<-matrix(0,nrow = nrow(dtexgb), ncol = 3)
    tepred_adabag<-matrix(0,nrow = nrow(tedat), ncol = 3)
    
    print("creating cv folds")
    set.seed(322)
    cvFolds <- createFolds(factor(trlabel,levels = unique(trlabel)), k=5, returnTrain=TRUE)
    print("fitting in cv")
    for(rnd in c(1:5)){
      cat("round = ", rnd, "\n")
      
      foldidx<-cvFolds[[rnd]]
      restidx<-setdiff(idxseq,foldidx)
      class<-trclas[foldidx]
      traindata<-trdat[foldidx,]
      preddata<-trdat[restidx,]
      
      print("fitting xgboost")
      xgb<-fxgb(dtrxgb[foldidx,])
      trpred_xgb[restidx,]<-t(matrix(predict(xgb, dtrxgb[restidx,]),
                                     nrow = 3,ncol = length(restidx)))
      tepred_xgb <-tepred_xgb+t(matrix(predict(xgb, dtexgb),
                                       nrow = 3,ncol = nrow(dtexgb)))
      gc()
      print("fitting lightgbm")
      lgbm<-flgbm(lightgbm::slice(dtrgbm,foldidx))
      trpred_lgbm[restidx,]<-t(matrix(predict(lgbm, trspar[restidx,]),
                                      nrow = 3,ncol = length(restidx)))
      tepred_lgbm<-tepred_lgbm+t(matrix(predict(lgbm, tespar),
                                        nrow = 3,ncol = nrow(dtexgb)))
      gc()
      print("fitting rf")
      rfFit<-frfFit(class,traindata)
      trpred_rf[restidx,]<-as.matrix(predict(rfFit, preddata,type = "prob"))
      tepred_rf<-tepred_rf+as.matrix(predict(rfFit, tedat,type = "prob"))
      gc()
      print("fitting svm")
      svmFit<-fsvmFit(class,traindata)
      trpred_svm[restidx,]<-as.matrix(predict(svmFit, preddata,type = "prob"))
      tepred_svm<-tepred_svm+as.matrix(predict(svmFit, tedat,type = "prob"))
      gc()
      print("fitting logistic")
      logistFit<-flogisticFit(class,traindata)
      trpred_logist[restidx,]<-as.matrix(predict(logistFit, preddata,type = "prob"))
      tepred_logist<-tepred_logist+as.matrix(predict(logistFit, tedat,type = "prob"))
      gc()
      print("fitting lda")
      ldaFit<-fldaFit(class, traindata)
      trpred_lda[restidx,]<-as.matrix(predict(ldaFit, preddata,type = "prob"))
      gc()
      print("fitting glmnet")
      glmnetFit<-fglmnetFit(class, traindata)
      trpred_glmnet[restidx,]<-as.matrix(predict(glmnetFit, preddata,type = "prob"))
      tepred_glmnet<-tepred_glmnet+as.matrix(predict(glmnetFit, tedat,type = "prob"))
      gc()
      print("fitting gbm")
      gbmFit<-fgbmFit(class,traindata)
      trpred_gbm[restidx,]<-as.matrix(predict(gbmFit, preddata,type = "prob"))
      tepred_gbm<-tepred_gbm+as.matrix(predict(gbmFit, tedat,type = "prob"))
      gc()
      print("fitting c50")
      c50Fit<-fc50Fit(class,traindata)
      trpred_c50[restidx,]<-as.matrix(predict(c50Fit, preddata,type = "prob"))
      tepred_c50<-tepred_c50+as.matrix(predict(c50Fit, tedat,type = "prob"))
      gc()

      print("fitting lightgbmdt")
      lgbmdt<-flgbmdt(lightgbm::slice(dtrgbm,foldidx))
      trpred_lgbmdt[restidx,]<-t(matrix(predict(lgbmdt, trspar[restidx,]),
                                        nrow = 3,ncol = length(restidx)))
      tepred_lgbmdt<-tepred_lgbmdt+t(matrix(predict(lgbmdt, tespar),
                                            nrow = 3,ncol = nrow(dtegbm)))

      print("fitting lightgbmr")
      lgbmr<-flgbmr(lightgbm::slice(dtrgbm,foldidx))
      trpred_lgbmr[restidx,]<-t(matrix(predict(lgbmr, trspar[restidx,]),
                                        nrow = 1,ncol = length(restidx)))
      tepred_lgbmr<-tepred_lgbmr+t(matrix(predict(lgbmr, tespar),
                                            nrow = 1,ncol = nrow(dtegbm)))
      gc()
      print("fitting xgbdt")
      xgbdt<-fxgbdt(dtrxgb[foldidx,])
      trpred_xgbdt[restidx,]<-t(matrix(predict(xgbdt, dtrxgb[restidx,]),
                                       nrow = 3,ncol = length(restidx)))
      tepred_xgbdt <-tepred_xgbdt+t(matrix(predict(xgbdt, dtexgb),
                                           nrow = 3,ncol = nrow(dtexgb)))
      gc()
      print("fitting adabag")
      adabagFit<-fadabagFit(class,traindata)
      trpred_adabag[restidx,]<-as.matrix(predict(adabagFit, preddata,type = "prob"))
      tepred_adabag<-tepred_adabag+as.matrix(predict(adabagFit, tedat,type = "prob"))
      
    }
    print("saving stage1 train prediction")
    
    write.csv(trpred_xgb,"./data/stacking/stage1_tr_xgb.csv",sep = ",",col.names = FALSE,row.names = FALSE)
    write.csv(trpred_lgbm,"./data/stacking/stage1_tr_lgbm.csv",sep = ",",col.names = FALSE,row.names = FALSE)
    write.csv(trpred_rf,"./data/stacking/stage1_tr_rf.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(trpred_svm,"./data/stacking/stage1_tr_svm.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_logist,"./data/stacking/stage1_te_logist.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(trpred_glmnet,"./data/stacking/stage1_tr_glmnet.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(trpred_gbm,"./data/stacking/stage1_tr_gbm.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(trpred_c50,"./data/stacking/stage1_tr_c50.csv",sep = ",",col.names = FALSE,row.names = FALSE)
    write.csv(trpred_lgbmdt,"./data/stacking/stage1_tr_lgbmdt.csv",sep = ",",col.names = FALSE,row.names = FALSE)
    
    
    stage1_tr_pred<-cbind(
                          trpred_xgb
                          ,trpred_lgbm
                          ,trpred_rf
                          ,trpred_logist
                          ,trpred_gbm
                          ,trpred_c50
                          ,trpred_gbm
                          ,trpred_lgbmdt
                          ,trpred_lgbmr
                          ,trpred_xgbdt
                          ,trpred_svm
                          ,trpred_glmnet
                          ,trpred_adabag
                          )

    stage1_te_pred<-cbind(
                          trpred_xgb/5.0
                          ,trpred_lgbm/5.0
                          ,trpred_rf/5.0
                          ,trpred_logist/5.0
                          ,trpred_gbm/5.0
                          ,trpred_c50/5.0
                          ,trpred_gbm/5.0
                          ,trpred_lgbmdt/5.0
                          ,trpred_lgbmr/5.0
                          ,trpred_xgbdt/5.0
                          ,trpred_svm/5.0
                          ,trpred_glmnet/5.0
                          ,trpred_adabag/5.0
                          )
    write.table(stage1_tr_pred,"./data/stacking/stage1_trpred12.csv",sep = ",",row.names = FALSE)
    write.table(stage1_te_pred,"./data/stacking/stage1_tepred12.csv",sep = ",",row.names = FALSE)
    remove(trpred_xgb,tr_pred_lgbm,trpred_rf,trpred_svm,
           trpred_logist,trpred_lda,trpred_glmnet,trpred_gbm)
    gc()
    return(list("tr"=stage1_tr_pred,"te"=stage1_te_pred))
  }
  else{
    print("fitting on whole dataset")
    class<-trclas
    traindata<-trdat
    preddata<-tedat
    
    print("fitting xgboost")

    xgb<-fxgb(dtrxgb)
    tepred_xgb<-t(matrix(predict(xgb, dtexgb),
                                 ncol = 3,nrow = nrow(preddata)))
    gc()
    print("fiitting lightgbm")
    lgbm<-flgbm(dtrgbm)
    tepred_lgbm<-t(matrix(predict(lgbm, tespar),
                                ncol = 3, nrow = nrow(preddata)))
    gc()
    print("fitting rf")
    rfFit<-frfFit(class,traindata)
    tepred_rf<-as.matrix(predict(rfFit, preddata,type = "prob"))
    gc()
    print("fitting svm")
    svmFit<-fsvmFit(class,traindata)
    tepred_svm<-as.matrix(predict(svmFit, preddata,type = "prob"))
    gc()
    print("fitting logistic")
    logistFit<-flogisticFit(class,traindata)
    tepred_logist<-as.matrix(predict(logistFit, preddata,type = "prob"))
    gc()
    print("fitting lda")
    ldaFit<-fldaFit(class, traindata)
    tepred_lda<-as.matrix(predict(ldaFit, preddata,type = "prob"))
    gc()
    print("fitting glmnet")
    glmnetFit<-fglmnetFit(class, traindata)
    tepred_glmnet<-as.matrix(predict(glmnetFit, preddata,type = "prob"))
    print("fitting gbm")
    gbmFit<-fgbmFit(class,traindata)
    tepred_gbm<-as.matrix(predict(gbmFit, preddata,type = "prob"))
    gc()
    print("saving stage1 test prediction")
    print("fitting lgbmdt")
    lgbmdt<-flgbmdt(dtrgbm)
    tepred_lgbmdt<-t(matrix(predict(lgbmdt, tespar),
                            nrow = 3,ncol = nrow(preddata)))
    gc()
    write.csv(tepred_xgb,"./data/stacking/stage1_te_xgb.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_lgbm,"./data/stacking/stage1_te_lgbm.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_rf,"./data/stacking/stage1_te_rf.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_svm,"./data/stacking/stage1_te_svm.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_logist,"./data/stacking/stage1_te_logist.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_lda,"./data/stacking/stage1__te_lda.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_glmnet,"./data/stacking/stage1_te_glmnet.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_gbm,"./data/stacking/stage1_te_gbm.csv",sep = ",",col.names = c("low","middle","high"),row.names = FALSE)
    write.csv(tepred_lgbmdt,"./data/stacking/stage1_te_lgbmdt.csv",sep = ",",col.names = FALSE,row.names = FALSE)
    stage1_te_pred<-cbind(tepred_xgb,tepred_lgbm,tepred_rf,
                          tepred_svm,tepred_logist,tepred_lda,tepred_glmnet,tepred_gbm)
    remove(trpred_xgb,tr_pred_lgbm,trpred_rf,trpred_svm,
           trpred_logist,trpred_lda,trpred_glmnet,trpred_gbm)
    gc()
    return(stage1_te_pred)
  }
}


telist<-fitting(iscv=TRUE,trdat=trdata,trlab=trlabel,trclas=trclass,tedat=te)
write.table(telist$tr,"./data/stacking/stage1_trpred.csv",sep = ",",row.names = FALSE)
write.table(telist$te,"./data/stacking/stage1_tepred.csv",sep = ",",row.names = FALSE)






