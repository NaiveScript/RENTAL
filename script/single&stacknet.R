# stacktest<-read.csv("/home/strn/Rental/StackNet-master/example/twosigma_kaggle/saved/stacknetwithfork/test_stacknet.csv",sep = ",",header=FALSE)
# stacklisting<-stacktest$V1
stackpred<-read.csv("/home/strn/Rental/StackNet-master/example/twosigma_kaggle/submission_0.507253516775.csv",
                      sep = ",",col.names = c("listing_id","low","medium","high"))
# colnames(stackpred)<-c("high","medium","low")
# # stackpred<-read.csv("./subfile/stacksubmission.csv",sep=",")
# stackpred<-cbind(listing_id=stacklisting,stackpred)
# # fwrite(stackpred, "./subfile/stacksubmission.csv")
# # stackpred<-read.csv("./StackNet-master/example/twosigma_kaggle/sigma_stack_pred.csv",
# #                       sep = ",")
# 
singlepred<-read.csv("./subfile/submission.csv",sep=",")
# singlelisting<-singlepred$listing_id
singlepred1<-read.csv("./subfile/submissiongbm.csv",sep=",")


mergetable<-merge(stackpred,singlepred,by="listing_id")
mergetable<-merge(mergetable,singlepred1,by="listing_id")
averagepred<-data.table(listing_id=mergetable$listing_id,
                        high=(0.3*mergetable$high.x+0.7*mergetable$high.y+0.0*mergetable$high),
                        medium=(0.3*mergetable$medium.x+0.7*mergetable$medium.y+0.0*mergetable$medium),
                        low=(0.3*mergetable$low.x+0.7*mergetable$low.y+0.0*mergetable$low))

# singlepred<-read.csv("./subfile/submission.csv",sep=",")
# singlepred1<-read.table("/home/strn/Rental/StackNet-master/example/twosigma_kaggle/cv1/tepred.csv",
#                       sep=",",col.names = c("low","medium","high"))
# 
# singlepred$high<-(singlepred$high*0.7+singlepred1$high*0.3)
# singlepred$medium<-(singlepred$medium*0.7+singlepred1$medium*0.3)
# singlepred$low<-(singlepred$low*0.7+singlepred1$low*0.3)

fwrite(averagepred, "./subfile/ensemblestack.csv")
remove(stacktest,stackpred,stacklisting,singlepred,singlelisting,mergetable,averagepred);gc()
