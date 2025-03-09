## Lecture 4

library(caret)
library(MASS)
library(RWeka)
library(rpart)
library(e1071)
library(dplyr)
library(rsample)

# set your working directory
setwd('C:\\Courses\\C2025\\699\\Slides-Spring2025\\L4')

############################################################################
# parameter tuning

df<-read.csv('drug_consumption_cannabis.csv')
head(df)
summary(df)
colnames(df)[which(names(df) == "C6")] <- "class"
sapply(df, class)
df$class <- factor(df$class)
sapply(df, class)
df <- df[, -1]
head(df)
dim(df)

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

# look up parameters
modelLookup("J48")

set.seed(31)
# repeat 10-fold cross-validation 5 times
# train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
#                               summaryFunction = defaultSummary)

train_control <- trainControl(method = "cv", number = 10,
                              summaryFunction = defaultSummary)

model <- train(class ~ ., data = train, method = "J48", trControl = train_control)
model
plot(model)
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$class, positive = '1')

## use tuneLength
set.seed(31)
model <- train(class ~ ., data = train, method = "J48", trControl = train_control,
               tuneLength = 4)
model
plot(model)
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$class)

## use tuneGrid
J48Grid <-  expand.grid(C = seq(0.005, 0.015, 0.001), M = (1:4))
set.seed(31)
model <- train(class ~ ., data = train, method = "J48", trControl = train_control,
               tuneGrid = J48Grid)
model
plot(model)
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$class)

# rpart algorithm

modelLookup("rpart")
set.seed(31)
model <- train(class ~ ., data = train, method = "rpart", trControl = train_control,
               tuneLength = 10)
model
plot(model)
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$class)

############################################################################
# multiple regression
# define prediction performance function

predictionPerformance = function(actual, predicted){
  e = actual - predicted
  RMSE = sqrt(mean(e * e))
  MAE = mean(abs(e))
  MAPE = 100 * mean(abs(e / actual))
  ME = mean(e)
  MPE = 100 * mean(e / actual)
  measures = cbind("RMSE" = RMSE, "MAE" = MAE, "MAPE" = MAPE, "ME" = ME, "MPE" = MPE)
  return (measures)
}

###########################################################################
## Boston Housing data

df <- read.csv('BostonHousing_numeric.csv')
head(df)
dim(df)

set.seed(31)
split <- initial_split(df, prop = 0.66)
train <- training(split)
test <- testing(split)
dim(train)
dim(test)

fit <- lm(MEDV ~ ., data = train)
summary(fit)
pred <- predict(fit, test[, -13])
predictionPerformance(test$MEDV, pred)

# variable selection with stepAIC

stepAIC_result <- MASS::stepAIC(fit, direction="both")
stepAIC_result$coefficients

############################################################################
## linear regression
## concrete strength data
## downloaded from UCI ML Repository

df <- read.csv('Concrete_Data.csv')

dim(df)
head(df)
sapply(df, class)
summary(df)

set.seed(31)
split <- initial_split(df, prop = 0.66)
train <- training(split)
test <- testing(split)
dim(train)
dim(test)

## ordinary linear regression
ordinary_fit <- lm(Strength ~ ., data = train)
summary(ordinary_fit)

pred <- predict(ordinary_fit, test[, -9])
predictionPerformance(test$Strength, pred)

## regularization using glmnet
## ridge regression (L2)
## lasso regression (L1)

library(glmnet) 

df <- read.csv('Concrete_Data.csv')

y <- df[, 9] %>% as.matrix() 
head(y)
X <- df[, -9] %>% as.matrix() 

# ridge penalty regression, alpha = 0
# Fit model with fixed lambda value 0.01 (random guessing)
ridge_model <- glmnet(X, y, alpha = 0, lambda = 0.01, 
                      standardize = TRUE) 
y_hat <- predict(ridge_model, X) 

predictionPerformance(y, y_hat)


# Perform 10-fold cross-validation to find best lambda 
lambdas_to_try <- 10^seq(4, -3, by = -0.1) 

# Setting alpha = 0 implements ridge regression 
ridge_cv <- cv.glmnet(X, y, alpha = 0, 
                      lambda = lambdas_to_try, 
                      standardize = TRUE, nfolds = 10) 

plot(ridge_cv)
coefficients(ridge_cv)

# Best cross-validated lambda 
lambda_cv <- ridge_cv$lambda.min 
lambda_cv

# Fit final model with best lambda
best_ridge <- glmnet(X, y, alpha = 0, lambda = lambda_cv, 
                     standardize = TRUE) 
y_hat_cv <- predict(best_ridge, X) 

predictionPerformance(y, y_hat_cv)

# lasso penalty regression, alpha = 1
# Fit model with fixed lambda value 0.01 (random guessing)
lasso_model <- glmnet(X, y, alpha = 1, lambda = 0.01, 
                      standardize = TRUE) 

y_hat <- predict(lasso_model, X) 

predictionPerformance(y, y_hat)

# Perform 10-fold cross-validation to select best lambda 
lambdas_to_try <- 10^seq(4, -3, by = -0.1)

# Setting alpha = 1 implements lasso regression 
lasso_cv <- cv.glmnet(X, y, alpha = 1, 
                      lambda = lambdas_to_try, 
                      standardize = TRUE, nfolds = 10) 

plot(lasso_cv)
coefficients(lasso_cv)

# Best cross-validated lambda 
lambda_cv <- lasso_cv$lambda.min 
lambda_cv

# Fit final model
best_lasso <- glmnet(X, y, alpha = 1, lambda = lambda_cv, 
                     standardize = TRUE) 
y_hat_cv <- predict(best_lasso, X) 

predictionPerformance(y, y_hat_cv)

#############################################################################
## KNN
## drug concumption dataset, downloaded from UCI ML Repository and modified 

df<-read.csv('drug_consumption_cannabis.csv')
head(df)
## change C6 to class
colnames(df)[which(names(df) == "C6")] <- "class"
## change data type of class
df$class <- factor(df$class)
## remove ID column
df <- df[, -1]
head(df)
dim(df)

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

## 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10, 
                              summaryFunction = defaultSummary)

## use tuneLength
## consider 100 k values
knnModel <- train(class ~., data = train, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"), # standardize
                  tuneLength = 100)

knnModel
plot(knnModel)

test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$class)

## use tuneGrid
knnGrid <-  expand.grid(k = seq(1, 200, 2))
knnModel <- train(class ~., data = train, method = "knn",
                  trControl=train_control,
                  preProcess = c("center", "scale"),
                  tuneGrid = knnGrid)

knnModel
plot(knnModel)

test_pred <- predict(knnModel, newdata = test)
confusionMatrix(test_pred, test$class)

###########################################################################
## Naive Bayes
# read RV.csv into df
df <- read.csv('RV.csv')
df
df$Class <- factor(df$Class) 
# build a model
model_nb <- naiveBayes(Class ~ ., data = df)
model_nb # display the model 

# classify new customer
new_customer <- list(Income="middle", Marital="single", Age="young",
                     Housing="own")
pred <- predict(model_nb, new_customer, type = "raw")
pred
pred <- predict(model_nb, new_customer, type = "class")
pred

## flight delays data example
# read FlightDelays.csv data into df
df <- read.csv('FlightDelays.csv')
head(df)
table(df$Flight.Status)

# use only CRS_DEPT_TIME, CARRIER, DEST, ORIGIN, and DAY_WEEK to
# predict Flight.Status
cols <- c(1, 2, 4, 8, 10, 13)
task.df <- df[cols]
# convert DAY_WEEK and Flight.Status to factor
task.df$DAY_WEEK <- factor(task.df$DAY_WEEK)
task.df$Flight.Status <- factor(task.df$Flight.Status)
# convert CRS_DEPT_TIME to hour as factor
task.df$CRS_DEP_TIME <- factor(round(task.df$CRS_DEP_TIME/100))
table(task.df$Flight.Status)

set.seed(31)
split <- initial_split(task.df, prop = 0.66, strata = Flight.Status)
train <- training(split)
test <- testing(split)
dim(train)
dim(test)

# build a Naïve Bayes model from training dataset
delays_nb <- naiveBayes(Flight.Status ~ ., data=train)
# test on test dataset
pred <- predict(delays_nb, newdata = test, type = "class")
# pred
# produce performance measures
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$Flight.Status)
performance_measures
