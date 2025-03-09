## L5

library(RWeka) 
library(caret) 
library(rsample)
library(C50)
library(rpart)
library(rpart.plot)
library(pROC)
library(MASS)
library(dplyr)
library(kernlab)

## set your workin directory
setwd('C:\\Courses\\C2025\\699\\Slides-Spring2025\\L5')

#######################################################################
## decision tree

## Weka J48 (C4.5)
## RV dataset
df<-read.csv('RV.csv')
df <- data.frame(unclass(df), stringsAsFactors = TRUE)
# fit model
RV.J48 <- J48(Class ~ ., data=df)
plot(RV.J48)
# summarize the fit
summary(RV.J48)

## universal bank dataset
df<-read.csv('UniversalBank.csv')
head(df)

## Personal.Loan is the class attribute
##   1: accepted personal loan, 0: did not accept persona loan
df$Personal.Loan <- factor(df$Personal.Loan)
df <- df[, -1] # remove ID column
head(df)
table(df$Personal.Loan)

# split dataset
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = Personal.Loan)
train <- training(split)
test <- testing(split)

J48.tree <- J48(Personal.Loan ~ ., data=train)

# test 
pred <- predict(J48.tree, newdata = test, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = test$Personal.Loan,
                                         positive = '1')

performance_measures

##########################################################################
## C5.0
C5.tree <- C5.0(Personal.Loan ~ ., data = train)

# test 
pred <- predict(C5.tree, newdata = test, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = test$Personal.Loan,
                                         positive = '1')
performance_measures

###########################################################################
## rpart
rpart.tree <- rpart(Personal.Loan ~ ., data = train, 
                    method = "class", parms = list(split = "gini"))
# plot the tree
prp(rpart.tree, type = 1, extra = 1, under = TRUE)

# test 
pred <- predict(rpart.tree, newdata = test, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = test$Personal.Loan,
                                         positive = '1')
performance_measures

#############################################################################
## overfitting 
## ThoracicSurgery1 dataset

df<-read.csv('ThoracicSurgery1.csv')
df$diagnosis <- factor(df$diagnosis)
head(df)

# split dataset
set.seed(123)
split <- initial_split(df, prop = 0.66, strata = diagnosis)
train <- training(split)
test <- testing(split)
head(test)

# build full tree
rpart.tree <- rpart(diagnosis ~ ., data = train, method = "class", minsplit=1,
                    cp = 0, parms = list(split = "information"))
# plot the full tree (overfitted)
prp(rpart.tree)

# test on training dataset
pred <- predict(rpart.tree, newdata = train, type = "class")
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = train$diagnosis)
performance_measures

# test on test dataset
pred <- predict(rpart.tree, newdata = test, type = "class")
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$diagnosis)
performance_measures

# find best cp and create a pruned tree
best <- rpart.tree$cptable[which.min(rpart.tree$cptable[,"xerror"]), "CP"]
print(best) 
# create pruned tree using best cp
pruned.tree <- prune(rpart.tree, cp=best)
#plot the pruned tree
prp(pruned.tree) 

# test on training dataset
pred <- predict(pruned.tree, newdata = train, type = "class")
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = train$diagnosis)
performance_measures

# test on test dataset
pred <- predict(pruned.tree, newdata = test, type = "class")
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$diagnosis)
performance_measures


########################################################################
## logistic
## heart disease

df<-read.csv('heart_disease_reduced.csv')
head(df)
df$class <- factor(df$class)
head(df)

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

# run logistic regression
# use glm() (generalized linear model) with family = "binomial" 
logitModel <- glm(class ~ ., data = train, family = "binomial") 
# disable scientific notation
options(scipen=999)
summary(logitModel)

# use predict() with type = "response" to compute predicted probabilities. 
logitModel.pred <- predict(logitModel, test[, -7], type = "response")
logitModel.pred[1:10]

# predicted probabilities of first 10 tuples in the test dataset
data.frame(actual = test$class[1:10], predicted = logitModel.pred[1:10])

# performance measures on the test dataset

pred <- factor(ifelse(logitModel.pred >= 0.5, 2, 1))
pred
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$class,
                                         positive = '1')
performance_measures

## ROC
rocobj=roc(test$class, logitModel.pred) 
plot(rocobj, main ="ROC curve", print.auc=TRUE)

## variable importance
x <- varImp(logitModel)
x
imp <- data.frame(varImp(logitModel))
imp <- data.frame(names   = rownames(imp),
                  overall = imp$Overall)
imp[order(imp$overall,decreasing = T), ]

############################################################################
## SVM
## This example uses svmRadial function which is an implmentation of SVM
## that uses "radial basis" kernel function
## It has two tuning parameters, sigma and C
## sigma and C, combined, controls overfitting

df<-read.csv('drug_consumption_cannabis.csv')
head(df)
summary(df)
colnames(df)[which(names(df) == "C6")] <- "class"

# use only a subset of columns (to reduce running time)
df <- df[, c(2,3,9,11,14)]
df$class <- as.factor(df$class)
head(df)
dim(df)

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

set.seed(31)
train_control <- trainControl(method = "cv", 
                              summaryFunction = defaultSummary)
svmGrid <-  expand.grid(sigma = seq(0.01, 0.1, by = 0.01), C = seq(0.1, 1.0, by = 0.1))

svm.model <- train(class ~ ., data = train, method = "svmRadial",
                   preProc = c("center", "scale"),
                   trControl = train_control, tuneGrid = svmGrid)
svm.model
plot(svm.model)

pred <- predict(svm.model, test)
performance_measures <- confusionMatrix(pred, test$class)
performance_measures


#########################################################################
## nnet
## Fit single-hidden-layer neural network
## size: number of units in the hidden layer
## decay: parameter for weight decay. 

library(NeuralNetTools)

# use only two predictors
df5 <- read.csv('ThoracicSurgery_small.csv')
df5
df5$diagnosis <- as.factor(df5$diagnosis)

library(nnet)
set.seed(31)
nnet_model <- nnet(diagnosis ~ ., data = df5, 
                   size = 4, trace = FALSE)
nnet_model
plotnet(nnet_model)

# shows, for each node, the bias and weights of the final model
neuralweights(nnet_model)

# heart disease dataset
df <- read.csv('heart_disease.csv')
dim(df)
head(df)
df$class <- ifelse(df$class==1, 'Yes', 'No')
head(df)
df$class <- as.factor(df$class)

# min-max normalization to [0, 1]
df_scaled <- df %>% mutate(across(age:thal, ~ ( . - min(.)) / (max(.) - min(.))))
head(df_scaled)

set.seed(31)
split <- initial_split(df_scaled, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)
dim(train)
dim(test)
table(train$class)
table(test$class)

# build model with three units in hidden layer
nnet_model <- nnet(class ~ ., data = train, size = 1, trace = TRUE)
nnet_model
plotnet(nnet_model)

# test the model
pred <- predict(nnet_model, newdata = test, type = "class")
pred <- factor(pred, levels = levels(test$class))

performance_measures <- confusionMatrix(pred, test$class)
performance_measures

# parameter tuning

ctrl <- trainControl(method = "CV", number = 10,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

nnetGrid <- expand.grid(size = 1:13, decay = seq(0, 2, 0.2))

set.seed(31)
nnetFit <- train(x = train[, -14], 
                 y = train$class,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 100,
                 MaxNWts = 1000,
                 trControl = ctrl)
nnetFit
nnetFit$bestTune
plot(nnetFit)

plotnet(nnetFit$finalModel)

test_pred <- predict(nnetFit, newdata = test)
test_pred

performance_measures <- confusionMatrix(test_pred, test$class)
performance_measures

# drug consumption dataset
df<-read.csv('drug_consumption_cannabis.csv')
head(df)
summary(df)
colnames(df)[which(names(df) == "C6")] <- "class"
sapply(df, class)

df <- df[, c(2,3,9,11,14)]
dim(df)
head(df)
sapply(df, class)
df$class <- as.factor(df$class)
df$class <- ifelse(df$class ==  0, "zero", "one")
sapply(df, class)
df$class <- as.factor(df$class)
head(df)

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1))

set.seed(31)
nnetFit <- train(x = train[, -5], 
                 y = train$class,
                 method = "nnet",
                 metric = "ROC",                 
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 50,
                 MaxNWts = 1000,
                 trControl = ctrl)

nnetFit
plot(nnetFit)
nnetFit$bestTune

nnetCM <- confusionMatrix(nnetFit)
nnetCM

test_pred <- predict(nnetFit, newdata = test)
confusionMatrix(test_pred, test$class)

