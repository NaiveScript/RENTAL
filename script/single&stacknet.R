stackpred<-read.csv("/home/strn/Rental/StackNet-master/example/twosigma_kaggle/submission_0.507253516775.csv",
                      sep = ",",col.names = c("listing_id","low","medium","high"))


singlepred<-read.csv("./subfile/submission.csv",sep=",")

singlepred1<-read.csv("./subfile/submissiongbm.csv",sep=",")


mergetable<-merge(stackpred,singlepred,by="listing_id")
mergetable<-merge(mergetable,singlepred1,by="listing_id")
averagepred<-data.table(listing_id=mergetable$listing_id,
                        high=(0.3*mergetable$high.x+0.7*mergetable$high.y+0.0*mergetable$high),
                        medium=(0.3*mergetable$medium.x+0.7*mergetable$medium.y+0.0*mergetable$medium),
                        low=(0.3*mergetable$low.x+0.7*mergetable$low.y+0.0*mergetable$low))

fwrite(averagepred, "./subfile/ensemblestack.csv")
remove(stacktest,stackpred,stacklisting,singlepred,singlelisting,mergetable,averagepred);gc()
