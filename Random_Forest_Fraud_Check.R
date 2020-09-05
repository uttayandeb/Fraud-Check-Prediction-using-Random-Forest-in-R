###### Packages required #######

library(MASS)
library(randomForest)
library(caret)


###### Reading and understanding the data ######


set.seed(123)# to get the same result everytime


Fraud_Check <- read.csv(file.choose())
View(Fraud_Check)
names(Fraud_Check)
nrow(Fraud_Check)
ncol(Fraud_Check)
str(Fraud_Check)
summary(Fraud_Check)


###### exploratory data analysis and visualization ######

hist(Fraud_Check$Taxable.Income)


hist(Fraud_Check$Taxable.Income, main = "Sales of Companydata",xlim = c(0,100000),
     breaks=c(seq(40,60,80)), col = c("blue","red", "green","violet"))


mean(Fraud_Check$Taxable.Income)#[1] 55208.38

#let us take 30,000 as a scale to check either the taxable income is good or risky
Risky_Good = ifelse(Fraud_Check$Taxable.Income<= 30000, "Risky", "Good")
# if Taxable Income is less than or equal to 30000 then it is Risky else its Good.


FCtemp= data.frame(Fraud_Check,Risky_Good)## adding a extra column to the dataframe 
FC = FCtemp[,c(1:7)]
View(FC)
View(FCtemp)

str(FC)

table(FC$Risky_Good) #to find out the number of good and risky taxable income 
                    # 476 good customers and 124 risky customers
                    # Good Risky 
                    #  476   124



########### Data Partition or splitting of data ##########

set.seed(123)## for not getting repeatable outputs or results

ind <- sample(2, nrow(FC), replace = TRUE, prob = c(0.7,0.3))

train <- FC[ind==1,]
nrow(train)###[1] 414

test  <- FC[ind==2,]
nrow(test)##[1] 186


set.seed(213)
rf <- randomForest(Risky_Good~., data=train)
rf  # Description of the random forest with no of trees
# OOB estimate of  error rate: 0.24%
#Number of trees: 500

attributes(rf)



####### Prediction and Confusion Matrix on  training data ######

pred1 <- predict(rf, train)
head(pred1)

head(train$Risky_Good)


##### confusion amtrix
confusionMatrix(pred1, train$Risky_Good) 
####Accuracy : 1          
#95% CI : (0.9911, 1)
# 100 % accuracy on training data 


########### Prediction and confusion Matrix with test data  ##################

pred2 <- predict(rf, test)
confusionMatrix(pred2, test$Risky_Good) 
#Accuracy : 1  ,100% accuracy on the test data        
#95% CI : (0.9804, 1)






##### Error Rate in Random Forest Model(visualization) #####
plot(rf)



# Tune Random Forest Model mtry 
tune <- tuneRF(train[,-6], train[,6], stepFactor = 0.5, plot = TRUE, ntreeTry = 300,
               trace = TRUE, improve = 0.05)



rf1 <- randomForest(Risky_Good~., data=train, ntree = 200, mtry = 2, importance = TRUE,
                    proximity = TRUE)
rf1



pred1 <- predict(rf1, train)
confusionMatrix(pred1, train$Risky_Good)  # 100 % accuracy on training data 




# test data prediction using the Tuned RF1 model
pred2 <- predict(rf1, test)
confusionMatrix(pred2, test$Risky_Good) # 100 % accuracy on test data 
#Accuracy : 1          
#95% CI : (0.9804, 1)


hist(treesize(rf1), main = "No of Nodes for the trees", col = "green")





################ Variable Importance ################

varImpPlot(rf1)

# Mean Decrease Accuracy graph shows that how worst the model performs without each variable.
# say Taxable.Income is the most important variable for prediction.on looking at City population,it has no value.

# MeanDecrease gini graph shows how much by average the gini decreases if one of those nodes were 
# removed. Taxable.Income is very important and Urban is not that important.




varImpPlot(rf1 ,Sort = T, n.var = 5, main = "Top 5 -Variable Importance")

# Quantitative values 
importance(rf1)

# Partial Dependence Plot 
partialPlot(rf1, train, Taxable.Income, "Good")


# On that graph, i see that if the taxable Income is 30000 or greater,
# than they are good customers else those are risky customers.



###### Extract single tree from the forest 

tr1 <- getTree(rf1, 2, labelVar = TRUE)

# Multi Dimension scaling plot of proximity Matrix
MDSplot(rf1, FC$Risky_Good)
