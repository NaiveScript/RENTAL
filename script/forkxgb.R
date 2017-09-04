# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 
# Load packages and data
packages <- c("jsonlite", "dplyr", "purrr")
purrr::walk(packages, library, character.only = TRUE, warn.conflicts = FALSE)

library(data.table)
library(xgboost)
library(caret)
library(stringr)
library(quanteda)
library(lubridate)
library(stringr)
library(Hmisc)
library(Matrix)
library(text2vec)

catNWayAvgCV <- function(data, varList, y, pred0, filter, k, f, g=1, lambda=NULL, r_k, cv=NULL){
  # It is probably best to sort your dataset first by filter and then by ID (or index)
  n <- length(varList)
  varNames <- paste0("v",seq(n))
  ind <- unlist(cv, use.names=FALSE)
  oof <- NULL
  if (length(cv) > 0){
    for (i in 1:length(cv)){
      sub1 <- data.table(v1=data[,varList,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
      sub1 <- sub1[sub1$filt==TRUE,]
      sub1[,filt:=NULL]
      colnames(sub1) <- c(varNames,"y","pred0")
      sub2 <- sub1[cv[[i]],]
      sub1 <- sub1[-cv[[i]],]
      sum1 <- sub1[,list(sumy=sum(y), avgY=mean(y), cnt=length(y)), by=varNames]
      tmp1 <- merge(sub2, sum1, by = varNames, all.x=TRUE, sort=FALSE)
      set(tmp1, i=which(is.na(tmp1[,cnt])), j="cnt", value=0)
      set(tmp1, i=which(is.na(tmp1[,sumy])), j="sumy", value=0)
      if(!is.null(lambda)) tmp1[beta:=lambda] else tmp1[,beta:= 1/(g+exp((tmp1[,cnt] - k)/f))]
      tmp1[,adj_avg:=((1-beta)*avgY+beta*pred0)]
      set(tmp1, i=which(is.na(tmp1[["avgY"]])), j="avgY", value=tmp1[is.na(tmp1[["avgY"]]), pred0])
      set(tmp1, i=which(is.na(tmp1[["adj_avg"]])), j="adj_avg", value=tmp1[is.na(tmp1[["adj_avg"]]), pred0])
      set(tmp1, i=NULL, j="adj_avg", value=tmp1$adj_avg*(1+(runif(nrow(sub2))-0.5)*r_k))
      oof <- c(oof, tmp1$adj_avg)
    }
  }
  oofInd <- data.frame(ind, oof)
  oofInd <- oofInd[order(oofInd$ind),]
  sub1 <- data.table(v1=data[,varList,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
  colnames(sub1) <- c(varNames,"y","pred0","filt")
  sub2 <- sub1[sub1$filt==F,]
  sub1 <- sub1[sub1$filt==T,]
  sum1 <- sub1[,list(sumy=sum(y), avgY=mean(y), cnt=length(y)), by=varNames]
  tmp1 <- merge(sub2, sum1, by = varNames, all.x=TRUE, sort=FALSE)
  tmp1$cnt[is.na(tmp1$cnt)] <- 0
  tmp1$sumy[is.na(tmp1$sumy)] <- 0
  if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(g+exp((tmp1$cnt - k)/f))
  tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgY + tmp1$beta*tmp1$pred0
  tmp1$avgY[is.na(tmp1$avgY)] <- tmp1$pred0[is.na(tmp1$avgY)]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
  # Combine train and test into one vector
  return(c(oofInd$oof, tmp1$adj_avg))
}


# Load training set
print("loading training set")
t1 <- fromJSON("./data/train.json")
t1_feats <- data.table(listing_id=rep(unlist(t1$listing_id), lapply(t1$features, length)), features=unlist(t1$features))
t1_photos <- data.table(listing_id=rep(unlist(t1$listing_id), lapply(t1$photos, length)), features=unlist(t1$photos))
vars <- setdiff(names(t1), c("photos", "features"))
t1<- map_at(t1, vars, unlist) %>% as.data.table(.)
t1[,":="(filter=0)]

# create 5 fold CV
set.seed(321)
cvFoldsList <- createFolds(t1$interest_level, k=5, list=TRUE, returnTrain=FALSE)

# Convert classes to integers for xgboost
class <- data.table(interest_level=c("low", "medium", "high"), class=c(0,1,2))
t1 <- merge(t1, class, by="interest_level", all.x=TRUE, sort=F)

# Load test set
print("loading test set")
s1 <- fromJSON("./data/test.json")
s1_feats <- data.table(listing_id=rep(unlist(s1$listing_id), lapply(s1$features, length)), features=unlist(s1$features))
s1_photos <- data.table(listing_id=rep(unlist(s1$listing_id), lapply(s1$photos, length)), features=unlist(s1$photos))
vars <- setdiff(names(s1), c("photos", "features"))
s1<- map_at(s1, vars, unlist) %>% as.data.table(.)
s1[,":="(interest_level="-1",
         class=-1,
         filter=2)]

ts1 <- rbind(t1, s1)
rm(t1, s1);gc()
ts1_feats <- rbind(t1_feats, s1_feats)
rm(t1_feats, s1_feats);gc()
ts1_photos <- rbind(t1_photos, s1_photos)
rm(t1_photos, s1_photos);gc()

ts1[,":="(created=as.POSIXct(created)
          ,dummy="A"
          ,low=as.integer(interest_level=="low")
          ,medium=as.integer(interest_level=="medium")
          ,high=as.integer(interest_level=="high")
          ,display_address=trimws(tolower(display_address))
          ,street_address=trimws(tolower(street_address)))]
ts1[, ":="(pred0_low=sum(interest_level=="low")/sum(filter==0),
           pred0_medium=sum(interest_level=="medium")/sum(filter==0),
           pred0_high=sum(interest_level=="high")/sum(filter==0))]

# merge Feature column
ts1_feats[,features:=gsub(" ", "_", paste0("feature_",trimws(char_tolower(features))))]
feats_summ <- ts1_feats[,.N, by=features]
ts1_feats_cast <- dcast.data.table(ts1_feats[!features %in% feats_summ[N<8, features]], listing_id ~ features, 
                                   fun.aggregate = function(x) as.integer(length(x) > 0), value.var = "features")


ts1 <- merge(ts1, ts1_feats_cast, by="listing_id", all.x=TRUE, sort=FALSE)
rm(ts1_feats_cast);gc()

# Photo counts
ts1_photos_summ <- ts1_photos[,.(photo_count=.N), by=listing_id]
ts1 <- merge(ts1, ts1_photos_summ, by="listing_id", all.x=TRUE, sort=FALSE)
rm(ts1_photos, ts1_photos_summ);gc()

# Convert building_ids and manager_ids with only 1 observation into a separate group
build_count <- ts1[,.(.N), by=building_id]
manag_count <- ts1[,.(.N), by=manager_id]
add_count <- ts1[,.(.N), by=display_address]
set(ts1, i=which(ts1[["building_id"]] %in% build_count[N==1, building_id]), j="building_id", value="-1")
set(ts1, i=which(ts1[["manager_id"]] %in% manag_count[N==1, manager_id]), j="manager_id", value="-1")
set(ts1, i=which(ts1[["display_address"]] %in% add_count[N==1, display_address]), j="display_address", value="-1")

# Mean target encode high cardinality variables
print("target encoding")
highCard <- c(
  "building_id",
  "manager_id"
)
for (col in 1:length(highCard)){
  # ts1[,paste0(highCard[col],"_mean_low"):=catNWayAvgCV(ts1, varList=c("dummy",highCard[col]), y="low", pred0="pred0_low", filter=ts1[["filter"]]==0, k=10, f=2, r_k=0.02, cv=cvFoldsList)]
  ts1[,paste0(highCard[col],"_mean_med"):=catNWayAvgCV(ts1, varList=c("dummy",highCard[col]), y="medium", pred0="pred0_medium", filter=ts1$filter==0, k=5, f=1, r_k=0.01, cv=cvFoldsList)]
  ts1[,paste0(highCard[col],"_mean_high"):=catNWayAvgCV(ts1, varList=c("dummy",highCard[col]), y="high", pred0="pred0_high", filter=ts1$filter==0, k=5, f=1, r_k=0.01, cv=cvFoldsList)]
}

#ts1$featurelen<-as.numeric(lapply(ts1$features,length))
# Create some date and other features
print("creating some more features")
ts1[,":="(building_id=as.integer(as.factor(building_id))
          ,display_address=as.integer(as.factor(display_address))
          ,manager_id=as.integer(as.factor(manager_id))
          ,street_address=as.integer(as.factor(street_address))
          ,desc_wordcount=str_count(description)
          ,pricePerBed=ifelse(!is.finite(price/bedrooms),-1, price/bedrooms)
          ,pricePerBath=ifelse(!is.finite(price/bathrooms),-1, price/bathrooms)
          ,pricePerRoom=ifelse(!is.finite(price/(bedrooms+bathrooms)),-1, price/(bedrooms+bathrooms))
          ,pricePerPhoto=ifelse(!is.finite(price/photo_count),-1, price/photo_count)
          ,bedPerBath=ifelse(!is.finite(bedrooms/bathrooms), -1, bedrooms/bathrooms)
          ,bedBathDiff=bedrooms-bathrooms
          ,bedBathSum=bedrooms+bathrooms
          ,bedsPerc=ifelse(!is.finite(bedrooms/(bedrooms+bathrooms)), -1, bedrooms/(bedrooms+bathrooms)))
    ]


## my code here,add more featrues

print("add more and more features")
tsmoref<-data.table(listing_id=ts1$listing_id)

#description cate,num,upper,lower,other

descate<-function(describe){
  describemat<-matrix(0,nrow = length(ts1$photos),ncol = 3)
  substring="0|1|2|3|4|5|6|7|8|9"
  substring1="!|&|@|="
  substring2="nice|awesome|beautiful|bright|top|actual|granite|amazing|heart|ultra|best|any|perfect|
  fantastic|splendid|spacious|luxury|spacious|charming"
  for (i in c(1:length(describe))){
        sdescribe<-setdiff(strsplit(describe[i]," "), c("",","))
        if(length(sdescribe)!=0){
          sdescribe<-sdescribe[[1]]
          for (j in c(1:length(sdescribe))){
            if (length(sdescribe[j])!=0){
                 if (grepl(substring,sdescribe[j])){
                   describemat[i,1]<-describemat[i,1]+1
                 }
                if (grepl(substring1,sdescribe[j])){
                  describemat[i,2]<-describemat[i,2]+1
                }
                if (grepl(substring2,sdescribe[j])){
                  describemat[i,3]<-describemat[i,3]+1
                }
            }
            else{
              1
              }
          }
        }
        else{
          1
        }
  }
  return (describemat)
}

# 1.make slice for price
print("create price slice")
# ts1$pricelog<log(ts1$price)
# ts1$priceexp<-exp(ts1$price)

quant<-quantile(ts1$price,c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
tspriceslice<-ts1$price
for (iter in c(1:(length(quant)-1))){
  tspriceslice[tspriceslice>=quant[iter]&tspriceslice<=quant[iter+1]]<-iter
}
tsmoref$priceslice<-factor(tspriceslice,levels = unique(tspriceslice))  

print("make circle feature")
# 2.add circle feature in a city ,calculate dis to center
# find center
# removeunnorm
lg<-ts1$longitude
la<-ts1$latitude
# la[la<40|la>41]<-median(la)
# lg[lg<(-74.5)|lg>(-72)]<-median(lg)

cl<-kmeans(data.table(lg,la),centers = 1,iter.max = 10,nstart=10)
clong<-cl$centers[1]
clat<-cl$centers[2]

gcd.slc <- function(long1, lat1, long2, lat2) {
  R <- 6371 # Earth mean radius [km]
  d <- acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2) * cos(long2-long1)) * R
  return(d) # Distance in km
}

tsmoref$dcircle<-gcd.slc(ts1$longitude*pi/180,ts1$latitude*pi/180,clong*pi/180,clat*pi/180)

print("make price revalue by dis to center")
# price revalue by distance to center
# tsmoref$pricerevalue<-2*log(ts1$price)*tanh(tsmoref$dcircle)
tsmoref$pricecrevalue<-log(ts1$price)/tanh(tsmoref$dcircle) #almost same 0.527813
remove(tspriceslice);gc()

listing_id_test <- ts1[filter %in% c(2), listing_id]
labels <- ts1[filter %in% c(0), class]

print("make word number slice")
#3.slice for description length,we found interest 2 is higher than others
wordcnt<-ts1$desc_wordcount

quant<-quantile(wordcnt[labels==1],c(0,0.25,0.5,0.75,1))

wordcnt[wordcnt<quant[2]|wordcnt>quant[4]]<-0
wordcnt[wordcnt>=quant[2]&wordcnt<=quant[4]]<-1

tsmoref$wordslice<-factor(wordcnt,levels = unique(wordcnt)) #achieve 0.5278 
# 
featurelen<-sapply(ts1$features,length)
quant<-quantile(featurelen,seq(0,1,by=0.1))
tsfeaturelen<-featurelen
for (iter in c(1:(length(quant)-1))){
  tsfeaturelen[tsfeaturelen>=quant[iter]&tsfeaturelen<=quant[iter+1]]<-iter
}

tsmoref$featrueslice<-tsfeaturelen # 0.523048

print("make manager skill slice")
# 4.manager skill coding
trmag<-ts1[filter==0, c("listing_id","manager_id","class"), with=FALSE]
tremag<-subset(ts1,select = c(listing_id,manager_id))

tremagpoints<-tremag$manager_id
tremagcnts<-tremag$manager_id
tremagratio<-tremag$manager_id

for(mag in unique(tremag$manager_id)){
  selectmag<-trmag[trmag$manager_id==mag]
  ratio<-nrow(selectmag)/nrow(tremag[tremag$manager_id==mag])
  tremagratio[tremag$manager_id==mag]<-ratio
  tremagcnts[tremag$manager_id==mag]<-nrow(tremag[tremag$manager_id==mag])
  
  if (mag %in% unique(trmag$manager_id) && ratio>0.0){
      nh<-length(which(selectmag$class==2))
      nm<-length(which(selectmag$class==1))
      points<-(nh*3+nm*1)/length(selectmag$listing_id)
      tremagpoints[tremag$manager_id==mag]<-points
      #tremagcnt[tremag$manager_id==mag]<-ratio
  }
  else{
    tremagpoints[tremag$manager_id==mag]<-0 #?
    #tremagcnt[tremag$manager_id==mag]<-ratio
  }
}
remove(selectmag)

#manager center coding
tsmag<-subset(ts1,select = c(listing_id,manager_id,longitude,latitude,class))
tsmag$magcenter<-tsmag$manager_id
tsmag$magbetter<-tsmag$manager_id

sbetterplace<-tsmag[tsmag$class>0]
lgbetter<-sbetterplace$longitude
labetter<-sbetterplace$latitude
clbetter<-kmeans(data.table(lg,la),centers = 1,iter.max = 20,nstart=10)
clongbetter<-cl$centers[1]
clatbetter<-cl$centers[2]
tsmag$tobetter<-tsmag$manager_id

tsmoref$tobetter<-gcd.slc(ts1$longitude*pi/180,ts1$latitude*pi/180,clongbetter*pi/180,clatbetter*pi/180) #0.524465

# price revalue by distance to better
#tsmoref$pricebrevalue<-log(ts1$price)/tanh(tsmoref$tobetter) #
#remove(tspriceslice);gc()
remove(sbetterplace)

for (mag in unique(tsmag$manager_id)){
  selectmag<-tsmag[tsmag$manager_id==mag]
  lg<-selectmag$longitude
  la<-selectmag$latitude
  cl<-kmeans(data.table(lg,la),centers = 1,iter.max = 20,nstart=10)
  clong<-cl$centers[1]
  clat<-cl$centers[2]
  if(selectmag$longitude!=clong||selectmag$latitude!=clat){
    tsmag$magcenter[tsmag$manager_id==mag]<-
      gcd.slc(selectmag$longitude*pi/180,selectmag$latitude*pi/180,clong*pi/180,clat*pi/180)
  }
  else{
    tsmag$magcenter[tsmag$manager_id==mag]<-0  # 0.524629
  }
  if(clong!=clongbetter||clat!=clatbetter){
    tsmag$magbetter[tsmag$manager_id==mag]<-
      gcd.slc(clong*pi/180,clat*pi/180,clongbetter*pi/180,clatbetter*pi/180)
  }
  else{
    tsmag$magbetter[tsmag$manager_id==mag]<-0  # 0.523894
  }
}     

tsmoref$tsmagcenter<-tsmag$magcenter
tsmoref$tsmagbetter<-tsmag$magbetter

remove(trmag,tremag,tsmag);gc()
# slicing
quant<-quantile(tremagpoints,seq(0,1,by=0.05))
tremagpnt<-tremagpoints
for (iter in c(1:(length(quant)-1))){
  tremagpnt[tremagpnt>=quant[iter]&tremagpnt<=quant[iter+1]]<-iter
}

 #?does this help?
# add time stamp
listingimgtime<-read.csv("./data/listing_image_time.csv",sep=",")
colnames(listingimgtime)[1]<-"listing_id"
ts1<-merge(ts1,listingimgtime,by="listing_id", all.x=TRUE, sort=FALSE) 


# time feature
time<-strsplit(as.character(ts1$created)," ")
time1<-sapply(time,"[",1)
startdate<-head(time1[order(format(as.Date(time1),"%y%m%d"))],1)
enddate<-tail(time1[order(format(as.Date(time1),"%y%m%d"))],1)
dayvec<-as.integer(as.Date(time1)-as.Date(startdate))

tsmoref$weekday<-as.numeric(dayvec%%7) #week day
# week<-as.numeric(dayvec/7) #week
tsmoref$monthday<-as.numeric(factor(dayvec%%30))#month day
tsmoref$month<-as.numeric(format(as.POSIXct(as.character(ts1$created),"%Y-%m-%d %H:%M:%S"),"%m")) #month # 0.5226
tsmoref$hour<-as.numeric(format(as.POSIXct(as.character(ts1$created),"%Y-%m-%d %H:%M:%S"),"%H")) #hour  # 0.523346
# fminute<-as.integer(format(as.POSIXct(as.character(data$created),"%Y-%m-%d %H:%M:%S"),"%M")) #minute

# measure month trend


pricebedbath<-data.table(listing_id=ts1$listing_id,pricev=ts1$price,bed=ts1$bedrooms,building= ts1$building_id,
                         mag=ts1$manager_id,bath=ts1$bathrooms,word=as.numeric(tsmoref$wordslice),mon=tsmoref$month,
                         magpnt=tremagpnt,street=ts1$street_address,photolen=ts1$photo_count
                         ,wordlen=ts1$desc_wordcount
                         # ,build=ts1$building_id
                         # ,imgt=imgt
                         # ,imgts=imgtslice
                         )
#room=(1+pmax(0, pmin( ts1$bedrooms, 4))+0.5*pmax(0, pmin( ts1$bathrooms, 2))
setDT(pricebedbath)[, priceofbedbath := pricev/median(pricev), by = c("bed","bath")] #0.526865
setDT(pricebedbath)[, priceofbed := pricev/median(pricev), by = c("bed")]   #0.526507
setDT(pricebedbath)[, priceofword:= pricev/median(pricev), by = c("word")]  #0.526499
setDT(pricebedbath)[, priceofmagpnt:= pricev/median(pricev), by = c("magpnt")] #0.526565
setDT(pricebedbath)[, priceofmag:= pricev/median(pricev), by = c("mag")]  #0.519617
setDT(pricebedbath)[, priceofbuild:= pricev/median(pricev), by = c("building")] #0.519731

setDT(pricebedbath)[, magstreet:= .N, by = c("street","mag")] # 0.523422
setDT(pricebedbath)[, magmon:= .N, by = c("mon","mag")]

setDT(pricebedbath)[, magbuild:= .N, by = c("building","mag")]
setDT(pricebedbath)[, buildstreet:= .N, by = c("building","street")]
setDT(pricebedbath)[, magN:= .N, by = c("mag")] 
setDT(pricebedbath)[, buildingN:= .N, by = c("building")] 
setDT(pricebedbath)[, streetN:= .N, by = c("street")] #0.5201

setDT(pricebedbath)[, magstreetm:= magstreet/magN]
setDT(pricebedbath)[, magbuildm:= magbuild/magN]   # 0.520503
setDT(pricebedbath)[, buildstreetm:= buildstreet/buildingN]
setDT(pricebedbath)[, magmonm:= magmon/magN]
setDT(pricebedbath)[, pricemagN:= pricev/median(pricev), by = c("magN")]
setDT(pricebedbath)[, pricebuildN:= pricev/median(pricev), by = c("buildingN")]
setDT(pricebedbath)[, pricestreetN:= pricev/median(pricev), by = c("streetN")]


ts1<-merge(ts1,subset(pricebedbath,select=c(listing_id,priceofbedbath,priceofbed,priceofword
                                            ,priceofmag
                                            ,priceofbuild
                                            ,priceofmagpnt
                                            # ,magbuild
                                            ,magstreet
                                            ,magN
                                            ,buildingN
                                            ,streetN
                                            ,magstreetm
                                            ,magbuildm
                                            ,buildstreetm
                                            ,magmonm
                                            #,pricemagN
                                            #,pricebuildN
                                            #,pricestreetN
                                            # ,photoofbuild
                                            # ,wordofbuild
                                            # ,bedofbuildmag
                                            # ,bedofstreetmag
                                            )),
                                            by="listing_id", all.x=TRUE, sort=FALSE)

remove(pricebedbath,tremagpnt,tremagpoints);gc()

# 4.add line dis feature
tsmoref$dline<-ts1$longitude/ts1$latitude
tsmoref$dline1<-ts1$longitude*ts1$latitude

# description featrue
descorpus<-corpus(as.character(factor(ts1$description,levels = unique(ts1$description))))
desdfm<-dfm(descorpus,remove = c("br","li","apartment","bedroom","1","room","building","2","p","york","call",
                                 "studio","bond","real","one","eaul","city","steps","street","apartments","windows",
                                 "bath","west","3","floor","dishwasher","area","two","access","bedrooms","close","see",
                                 " 24","can","href","view","will","rent","storage","transportation","ul","east",
                                 "pictures","including","c","w","brick","trains","amenities","walk","bed","please",
                                 "kagglemanager@renthop.com","laundry","bathroom","appliances","find","listing",
                                 "wood","shops","roof","tons","blocks","queen","ceilings","information","schedule",
                                 "train","brand","equal","size","market","sized","24","block","_blank","marble",
                                 "contact","opportunity","supports","ocation","located","text","steel","park","home",
                                 "subway","deck","midtown","service","pets","4","apt","s","bathrooms","time",
                                 "appointment","neighborhood","counter","included","photos","deal",
                                 stopwords("english")),stem = FALSE,removePunct = TRUE)
#dessparse<-tfidf(desdfm,normalize = TRUE)
dessparse<-desdfm
dessparse<-dfm_select(dessparse,features=names(topfeatures(dessparse,80))) ##0.527813

tsmoref$desvalue<-rowSums(matrix(as.numeric(dessparse),nrow = length(ts1$listing_id),byrow=FALSE)) # 0.525615 0.524763

# one hot coding for factors
dmy <- dummyVars(" ~ .", data = tsmoref)
trsf <- data.frame(predict(dmy, newdata = tsmoref))

ts1<-merge(ts1,trsf,by="listing_id", all.x=TRUE, sort=FALSE)


# fill in missing values with -1
print("fill in missing values")
for (col in 1:ncol(ts1)){
  set(ts1, i=which(is.na(ts1[[col]])), j=col, value=-1)
}

# don't want to sparse description
print("get variable names")
varnames <- setdiff(colnames(ts1), c("photos","pred0_high", "pred0_low","pred0_medium",
                                     "description", "features","interest_level","dummy","filter", "created", "class", "low","medium","high","street"))
# Convert dataset to sparse format
print("converting data to sparse format")
t1_sparse <- Matrix(as.matrix(ts1[filter==0, varnames, with=FALSE]), sparse=TRUE)
s1_sparse <- Matrix(as.matrix(ts1[filter==2, varnames, with=FALSE]), sparse=TRUE)


print("converting data into xgb format")
dtrain <- xgb.DMatrix(data=t1_sparse, label=labels)
dtest <- xgb.DMatrix(data=s1_sparse)

param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              nthread=4,
              num_class=3,
              eta = .02,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .5 # 0.5->0.7
)

set.seed(201609)
(tme <- Sys.time())
xgb2cv <- xgb.cv(data = dtrain,
                params = param,
                nrounds = 50000,
                maximize=FALSE,
                prediction = TRUE,
                folds = cvFoldsList,
                # nfold = 5,
                print_every_n = 50,
                early_stopping_round=200,
                missing=-1)
Sys.time() - tme

set.seed(2)
watch <- list(dtrain=dtrain)
xgb2 <- xgb.train(data = dtrain,
                  params = param,
                  watchlist=watch,
                  print_every_n = 50,
                  nrounds = 4242
                  #nrounds = xgb2cv$best_ntreelimit

)

# train in cv
set.seed(321)
cvFolds <- createFolds(labels, k=5, returnTrain=TRUE)

cnt<-1
for(fold in cvFolds){
  cat("round = ", cnt, "\n")
  dtrainfold<-dtrain[fold,]
  watch <- list(dtrain=dtrainfold)
  xgbfold <- xgb.train(data = dtrain,
                    params = param,
                    watchlist=watch,
                    nrounds = xgb2cv$best_ntreelimit,
                    print_every_n = 100
                    # nrounds = 5300
  )
  spredfold <- as.data.table(t(matrix(predict(xgbfold, dtest), nrow=3, ncol=nrow(dtest))))
  if (cnt==1){
    sPred<- spredfold
  }
  else{
    sPred<-sPred+spredfold
  }
  cnt<-cnt+1
}
sPred<-sPred/(cnt-1)
#sPreds <- as.data.table(t(matrix(predict(xgb2, dtest), nrow=3, ncol=nrow(dtest))))
colnames(sPred) <- class$interest_level
fwrite(data.table(listing_id=listing_id_test, sPred[,list(high,medium,low)]), "./subfile/submission.csv")  # submissiongbm.csv 

# importanceraw<-xgb.importance(feature_names = colnames(dtrain),xgb2)
# xgb.plot.importance(importance_matrix = importanceraw[1:50])
# top60<-importanceraw$Feature[1:60]
# selectfeatures<-subset(ts1,select = top60)
# write.table(selectfeatures,"./stacking/middlepredicting/sfeatrues.csv",sep = ",",col.names = FALSE,row.names = FALSE)

# sPreds <- as.data.table(t(matrix(predict(xgb2, dtest), nrow=3, ncol=nrow(dtest))))
# colnames(sPreds) <- class$interest_level
# fwrite(data.table(listing_id=listing_id_test, sPreds[,list(high,medium,low)]), "./subfile/submission.csv") # 0.51272