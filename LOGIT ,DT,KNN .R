library(ISLR)
setwd("D:/DSA4211 project")

Bookings <- read.csv(file = 'hotel_bookings.csv')
data<-Bookings

#attach(data)

#Remove reservation_status as it is same as is_canceled
data <-subset(data, select=-c(reservation_status))

#check no na
for (col_no in 1:ncol(data)){
  print(colnames(data[col_no]))
  print(sum(is.na(data[col_no])))
}

#since children has na remove those rows
data<-data[!is.na(data$children),]



#now making ALL CATERGORICAL DATA into factors and convert years into number

data$hotel<-as.factor(data$hotel)
data$is_canceled<-as.factor(data$is_canceled)
data$arrival_date_year<-as.numeric(data$arrival_date_year)
data$arrival_date_week_number<-as.numeric(data$arrival_date_week_number)
data$arrival_date_day_of_month<-as.numeric(data$arrival_date_day_of_month)

data$meal<-as.factor(data$meal)
#country already handled
data$market_segment<-as.factor(data$market_segment)
data$distribution_channel<-as.factor(data$distribution_channel)
data$is_repeated_guest<-as.factor(data$is_repeated_guest)
data$reserved_room_type<-as.factor(data$reserved_room_type)
data$assigned_room_type<-as.factor(data$assigned_room_type)
data$deposit_type<-as.factor(data$deposit_type)

#agent and company already 
data$customer_type<-as.factor(data$customer_type)
#data$reservation_status<-as.factor(data$reservation_status)
data$arrival_date_month<-as.numeric(match(data$arrival_date_month,month.name))
data$reservation_status_date<-as.numeric(as.Date(data$reservation_status_date,format='%Y-%m-%d'))


#according to column description on kaggle Undefined and SC are same for meal thus reassign all Undefined to SC
data$meal[data$meal=="Undefined"]<-"SC"
data$meal<-factor(data$meal)

#remove undefined from market segment
data<-data[!(data$market_segment=="Undefined"),]
data$market_segment<-factor(data$market_segment)

#remove undefined from distribution channel
data<-data[!(data$distribution_channel=="Undefined"),]
data$distribution_channel<-factor(data$distribution_channel)

#can see which one has very high or low adr
#which(data$adr %in% data$adr[data$adr>4000])
#which(data$adr %in% data$adr[data$adr<=0])
#which(data$adr %in% data$adr[data$adr<0])

#set seed so same answers
set.seed(0)

#doing SIS lasso
library(varhandle)
library(fastDummies)
str(data)
data_x<-cbind(data[,-2])
data_y<-cbind(data[c(2)])
data_x<-dummy_cols(data_x,remove_most_frequent_dummy = TRUE, remove_selected_columns = TRUE)
data_x<-data.matrix(data_x, rownames.force = NA)
data_y<-data.matrix(data_y, rownames.force = NA)
library(SIS)
n<-31 #31 predictor
d<-floor(n/log(n))

#use this to skip long sis
#isis_p = c(1, 15, 16, 17, 204, 205, 230, 334, 919)

sis_lasso<-SIS(data_x,data_y,family="gaussian",penalty="lasso",tune="cv",nfolds=10, nsis=d,iter.max=5,seed=1)
sis_x<-cbind(data_x[,sis_lasso$ix])
sis_y<-cbind(data[c(2)])
sis_all<-cbind(sis_y,sis_x)

# Change column names to remove spaces for Tree function
colnames(sis_x)[6]
colnames(sis_x)[7]
colnames(sis_x)[9]
colnames(sis_x)[6] <-'market_segment_Offline_TA_TO'
colnames(sis_x)[7] <-'deposit_type_Non_Refund'
colnames(sis_x)[9] <- 'customer_type_Transient_Party'
colnames(sis_all)[7]
colnames(sis_all)[8]
colnames(sis_all)[10]
colnames(sis_all)[7] <-'market_segment_Offline_TA_TO'
colnames(sis_all)[8] <-'deposit_type_Non_Refund'
colnames(sis_all)[10] <- 'customer_type_Transient_Party'

#create index for 80 20 split
index = round(nrow(data)*0.2,digits=0)
test.indices = sample(1:nrow(data),index)


#xtrain ytrain xtest ytest
Ytrain = sis_y[-test.indices,]
Xtrain = sis_x[-test.indices,]
train = sis_all[-test.indices,]

Ytest = sis_y[test.indices,]
Xtest = sis_x[test.indices,]
test = sis_all[test.indices,]


#logit
library('ROCit')


logit1 <- glm(is_canceled ~ .,data=train, family=binomial)
summary(logit1)
#train results
phat<- fitted(logit1)
phat<-predict.glm(logit1,type="response")
roc.info<-roc(ytrain,phat,plot=TRUE,legacy.axes=TRUE)
roc.df<-data.frame(tpp=roc.info$sensitivities*100,tnr=roc.info$specificities*100,thresholds=roc.info$thresholds)
roc.df<-roc.df[roc.df$thresholds>0.3,]
roc.df<-roc.df[roc.df$thresholds<0.8,]
roc.df<-data.frame(roc.df)
#only get with threshld between 0.3 and 0.8 where the youden index is
length<-nrow(roc.df)
temp_best<-0
best<-0
for(i in 1:length){
  a<-roc.df[i,]
  temp<-0
  temp<-a[,1]+a[,2]
  if(temp_best<temp){
    temp_best<-temp
    best<-a[,3]
  }
  
  
}
best
logit.pred<-ifelse(phat>best,1,0)
table(logit.pred,ytrain)
#test_result
pphat<-predict.glm(logit1,newdata=data.frame(xtest),type="response")
pred.pred<-ifelse(pphat>best,1,0)
table(pred.pred,ytest)


#K-nearest neighbours
#sort(c(1:2,floor(95508^(1/(2:9)))))
library(class)
k_list = sort(c(1:2,floor(95508^(1/(2:9)))))
err = rep(0,10)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
for (i in 1:10){
  fold.indices=sample(1:nrow(train), size = (nrow(data)/11-i), replace = FALSE)
  train.set = sis_all[-fold.indices,]
  s = scale(train.set[,-1])
  train.set[, -1] <- s
  test.set = sis_all[fold.indices,]
  test.set[, -1] <- scale(test.set[,-1], center = attr(s,"scaled:center"), scale = attr(s,"scaled:scale"))
  for (k in 1:10){
    knn.pred <- knn(train.set,test.set,cl=train.set[,"is_canceled"],k=k_list[k])
    err[k] = err[k] + accuracy(table(knn.pred,test.set[,1]))
    cat(i,k,"\n")
  }
}

#err/10 = 99.97971 99.95482 99.94376 99.92993 99.91610 99.90412 99.87370 99.80180 99.55197 98.20604
best_k = k_list[which.max(err)]

train.set <- train
s = scale(train.set[,-1])
train.set[, -1] <- s
test.set <- test
test.set[, -1] <- scale(test.set[,-1], center = attr(s,"scaled:center"), scale = attr(s,"scaled:scale"))
knn.pred <- knn(train.set,test.set,cl=train.set[,"is_canceled"],k=best_k)
final_acc = accuracy(table(knn.pred,test.set[,1]))
print(final_acc)
print(table(knn.pred,test.set[,1]))
#final_acc = 99.98744
#confusion table = p0a0=15126, p0a1=3, p1a0=0, p1a1=8748


#Decision Tree
str(train)

library(tree)
fmE <- tree(is_canceled~., train,)
fmE
# 7 terminal nodes for fmE
summary(fmE)

library(maptree)
draw.tree(fmE)


n_obs = nrow(train)
# Need to specify mincut for gini tree
# Around 73 terminal nodes for fmG ; error rate = 0.2126
fmG <- tree(is_canceled~., train, split="gini",control=tree.control(nobs = n_obs, mincut = 1000))
fmG
summary(fmG)
draw.tree(fmG)

CV <- cv.tree(fmE, FUN=prune.misclass)
CV
#No need to prune fmE tree as CV error is lowest at size = 7
plot(CV$size, CV$dev, type="b", xlab="size of tree", ylab="CV error")



# Even though 'gini' leads to a slightly lower error rate, the tree using entropy is much easier to explain
# since it only has 7 terminal nodes compared to 73 when 'gini' is used.
# Training error
ypred <- predict(fmE, train, type="class")
table(ypred, ytrain)
mean(ypred!=ytrain)
# Test error
ypred <- predict(fmE, test, type="class")
table(ypred, ytest)
mean(ypred!=ytest)

(singletesterr <- mean(ypred!=ytest))


# Random Forest
p <- dim(train)[2]-1

library(randomForest)

# Code runs for around 10 mins
# Increasing end point of B does not show any improvement in test error, and 
# may produce memory issues in R.
# Test different values of p (p [bagging], p/2, sqrt(p)) 
Bval <- seq(from=100, to=500, by=100)
T <- length(Bval)
mval <- c(p, p/2, floor(sqrt(p)))
testerr_rf <- matrix(0, length(mval), T)  # store test set error
for (i in 1:length(mval)){
  m <- mval[i]
  for (t in 1:T){
    B <- Bval[t]
    rf <- randomForest(is_canceled~., data=train, mtry=m, ntree=B)
    ypred <- predict(rf, newdata=test)
    testerr_rf[i,t] <- mean(ypred!=ytest)
    cat(i, t, "\n")
  }
}

#Bagging (m=p) seems to produce the smallest error.
plot(Bval, testerr_rf[1,], type="l", ylim=c(0.15, 0.22),
     xlab="B", ylab="test error", main="Random forests")
points(Bval, testerr_rf[2,], type="l", col="red", lty=2)
points(Bval, testerr_rf[3,], type="l", col="blue", lty=2)
abline(h=singletesterr, col="seagreen")
legend("topright", legend=c("single tree", "m=p (Bagging)", "m=p/2", m="sqrt(p)"),
       col=c("seagreen", "black", "red", "blue"), lty=c(1,1,2,2))