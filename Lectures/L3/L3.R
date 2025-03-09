### Performance measures

library(rsample)
library(caret)
library(dplyr)
library(e1071)

## set working directory
setwd('C:/Courses/C2025/699/Slides-Spring2025/L3')

#############################################################################
## Performance Evaluation

## Numeric prediction performance

# define performance measure function
predictionMeasures = function(actual, predicted){
  e = actual - predicted
  MAE = mean(abs(e))
  ME = mean(e)
  MPE = 100 * mean(e / actual)
  MAPE = 100 * mean(abs(e / actual))
  RMSE = sqrt(mean(e * e))
  measures = cbind("MAE" = MAE, "ME" = ME, "MPE" = MPE, "MAPE" = MAPE, "RMSE" = RMSE)
  return (measures)
}

# energy efficiency data
# downloaded from UCI ML Repository

df <- read.csv('energy_efficiency.csv')
dim(df)
head(df)
summary(df)

# standardize

df1 <- df %>%
  select(-Y1) %>%
  scale() %>%
  as.data.frame() %>%
  cbind(Y1 = df$Y1)
summary(df1)

library(rsample)
set.seed(31)
split <- initial_split(df1, prop = 0.66)
train <- training(split)
test <- testing(split)
dim(train)
dim(test)

## ordinary linear regression
ordinary_fit <- lm(Y1 ~ ., data = train)
summary(ordinary_fit)

pred <- predict(ordinary_fit, test[, -8])
predictionMeasures(test[, 8], pred)

## Classification performance

## illustration of hold out method and cross-validation
## use Naive Bayes 

## holdout method example
## heart disease dataset, downloaded from UCI Machine Learning Repository
df<-read.csv('heart_disease.csv')
head(df)

## change the data type of class attribute to factor
df$class<-as.factor(df$class)

# stratified split 
# 66-34% split
set.seed(1)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

## Naive Bayes model

model_nb <- naiveBayes(class ~ ., data = train) # build mode from training data
pred <- predict(model_nb, newdata = test, type = "class") # test on test/unseen data
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$class, positive = '1')
performance_measures

# calculate performance measures from confusion matrix
## assume 1 is positive
cm <- performance_measures$table

tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e) # Kappa statistic
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
performance_measures

## cross-validation

set.seed(31)

train_control <- trainControl(method = "cv", number = 10)
model <- train(class ~ ., data = df, method = "nb", trControl = train_control,
               tuneLength = 1) 
model

### ROC Curve
## there are different ways of plotting ROC curves

df<-read.csv('autism_adult.csv')
df$Class.ASD<-as.factor(df$Class.ASD)
head(df)

# stratified split 
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = Class.ASD)
train <- training(split)
test <- testing(split)

## use ROCR

library(ROCR)
model_nb <- naiveBayes(Class.ASD ~ ., data = train)
pred <- predict(model_nb, newdata = test, type = "raw")
## convert pred to data frame to be used in prediction
pred <- data.frame(pred)
pred
pred_perf <- prediction(predictions = pred$YES, labels = test$Class.ASD)
roc <- performance(pred_perf, measure="tpr", x.measure = "fpr")
plot(roc, main = "ROC")
abline(a=0, b= 1)
roc_area = performance(pred_perf, measure = "auc")
roc_area@y.values

## use pROC

library(pROC)
model_nb <- naiveBayes(Class.ASD ~ ., data = train)
pred <- predict(model_nb, newdata = test, type = "raw")
head(pred)

YES_prob <- pred[, "YES"]
rocCurve <- roc(test$Class.ASD, YES_prob)
plot(rocCurve, legacy.axes = TRUE)
auc(rocCurve)

## plot multiple models; Naive Bayes and J48 decision tree
## J48 is in RWeka package
library(RWeka)

## use prerec
library(precrec)

## ThoracicSurgery dataset
## downloaded from UCI ML Repository and modified

df<-read.csv('ThoracicSurgery1.csv')
df$diagnosis<-as.factor(df$diagnosis)
head(df)

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = diagnosis)
train <- training(split)
test <- testing(split)

model_nb <- naiveBayes(diagnosis ~ ., data = train) 
pred_nb <- predict(model_nb, newdata = test, type = "class") # get classes
score_nb <- as.numeric(pred_nb)
model_J48 <- J48(diagnosis ~ ., data = train)
pred_J48 <- predict(model_J48, newdata = test)
score_J48 <- as.numeric(pred_J48)
scores = join_scores(score_nb, score_J48)

msmdat <- mmdata(scores, test$diagnosis, modnames = c("Naive Bayes", "J48"))
mscurves <- evalmod(msmdat)
library(ggplot2)
library(patchwork)
autoplot(mscurves, "ROC")
auc(mscurves)

###########################################################################
## imbalanced data
###########################################################################

## bank dataset 
## downloaded from UCI ML Repository and modified
## 'y' is the class attribute
## class distribuion 
##    yes: 521 instances (11.5%)
##    no: 4000 instances (88.5%)

df <- read.csv('bank.csv')
head(df)
df$y <- as.factor(df$y)
dim(df)
table(df$y)

set.seed(31)
split <- initial_split(df, prop = 0.667, strata = y)
train <- training(split)
dim(train)
table(train$y)
test <- testing(split)
dim(test)
table(test$y)

## build Naive Bayes model from the initial, imbalanced training dataset

nb.model <- naiveBayes(y ~ ., data=train)
# test on test dataset
pred <- predict(nb.model, newdata = test, type = "class")
# produce performance measures
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$y,
                                         positive = 'yes')
## note the pediction accuricy of class yes (sensitivity)
## and the prediction accuracy for class no (specificity)
performance_measures

############################################################################
library(ROSE)

## Under-sampling
## create undersampled training dataset
train_under = ovun.sample(y~., data=train, p=0.5, seed=33, method='under')$data
dim(train_under)
table(train_under$y) # yes: 347, no: 356

## build Naive Bayes model from undersampled training dataset
nb.model.under <- naiveBayes(y ~ ., data=train_under)
## test on test dataset, which has initial imbalanced class distributin
pred <- predict(nb.model.under, newdata = test, type = "class")
# produce performance measures
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$y,
                                         positive = 'yes')
## note the pediction accuricy of class yes (sensitivity)
## and the prediction accuracy for class no (specificity)
performance_measures

## Over-sampling
## create oversampled training dataset
train_over = ovun.sample(y~., data=train, p=0.5, seed=33, method='over')$data
dim(train_over)
table(train_over$y) # yes: 2660, no:2668

## build Naive Bayes model from oversampled training dataset
nb.model.over <- naiveBayes(y ~ ., data=train_over)
## test on test dataset, which has initial imbalanced class distributin
pred <- predict(nb.model.over, newdata = test, type = "class")
# produce performance measures
performance_measures  <- confusionMatrix(data=pred, 
                                         reference = test$y,
                                         positive = 'yes')
## note the pediction accuricy of class yes (sensitivity)
## and the prediction accuracy for class no (specificity)
performance_measures


