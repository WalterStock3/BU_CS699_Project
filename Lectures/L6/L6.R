#### L6
library(caret)
library(MASS)

## set your workin directory
setwd('C:\\Courses\\C2025\\699\\Slides-Spring2025\\L6')

##########################################################################
## discriminant analysis
## drug consumption cannabis dataset

df<-read.csv('drug_consumption_cannabis.csv')
colnames(df)[which(names(df) == "C6")] <- "class"
df$class <- factor(df$class)
df <- df[, -1]
# standardize
preProcValues <- preProcess(df[, -13], method = c("center", "scale"))
df <- predict(preProcValues, df)

library(rsample)
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

# lda
ldaModel <- lda(class ~ ., data = train) 
ldaModel
pred <- predict(ldaModel, test)
performance_measures  <- confusionMatrix(data=pred$class, 
                                         reference = test$class)
performance_measures

# qdq
qdaModel <- qda(class ~ ., data = train) 
pred <- predict(qdaModel, test)
performance_measures  <- confusionMatrix(data=pred$class, 
                                         reference = test$class)
performance_measures

############################################################################
## boosting

# AdaBoost.M1 (method = 'AdaBoost.M1')
# For classification using packages adabag with tuning parameters:
#   Number of Trees (mfinal, numeric)
#   Max Tree Depth (maxdepth, numeric)
#   Coefficient Type (coeflearn, character) 

library(adabag)

modelLookup("AdaBoost.M1")

df<-read.csv('drug_consumption_cannabis.csv')
colnames(df)[which(names(df) == "C6")] <- "class"
df$class <- ifelse(df$class ==  0, "N", "Y")
df$class <- as.factor(df$class)
df <- df[, -1]
head(df)
dim(df)

library(rsample)
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)
table(train$class)
table(test$class)

ctrl <- trainControl(method = "CV", number = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
boostingGrid <- expand.grid(mfinal = seq(50, 500, 50), maxdepth = seq(1,3),
                          coeflearn = "Freund")

## use default parameters
boosting_model <- train(class ~ ., data = train, method = "AdaBoost.M1", 
                        metric = "ROC", tuneGrid = boostingGrid, trControl = ctrl)

boosting_model
plot(boosting_model)

pred <- predict(boosting_model, test)
cm <- caret::confusionMatrix(pred, test$class)
cm

############################################################################
## random forest

modelLookup("rf")
df<-read.csv('drug_consumption_cannabis.csv')
colnames(df)[which(names(df) == "C6")] <- "class"
df$class <- ifelse(df$class ==  0, "N", "Y")
df$class <- as.factor(df$class)
df <- df[, -1]
head(df)
dim(df)

library(rsample)
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

mtryValues <- seq(2, ncol(df)-1, by = 1)

set.seed(31)
rfFit <- caret::train(x = train[, -13], 
                      y = train$class,
                      method = "rf",
                      ntree = 100,
                      tuneGrid = data.frame(mtry = mtryValues),
                      importance = TRUE,
                      metric = "ROC",
                      trControl = ctrl)
rfFit
plot(rfFit)

## variable importance
imp <- varImp(rfFit)
imp

pred <- predict(rfFit, test)
cm <- caret::confusionMatrix(pred, test$class)
cm

library(pROC)
rfRoc <- roc(response = rfFit$pred$obs,
             predictor = rfFit$pred$Y,
             levels = rev(levels(rfFit$pred$obs)))
plot(rfRoc, main ="ROC curve", print.auc=TRUE)

###########################################################################
## gradient boosting machine

library(gbm)
modelLookup("gbm")
df<-read.csv('drug_consumption_cannabis.csv')
dim(df)
colnames(df)[which(names(df) == "C6")] <- "class"
df$class <- ifelse(df$class ==  0, "N", "Y")
df$class <- as.factor(df$class)
df <- df[, -1]

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)

ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10)*10,
                       shrinkage = c(.001, .01),
                       n.minobsinnode = 3)

set.seed(31)
gbmFit <- caret::train(x = train[, -13], 
                       y = train$class,
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)
gbmFit
plot(gbmFit)

pred <- predict(gbmFit, test)
cm <- caret::confusionMatrix(pred, test$class)
cm

############################################################################
## XGBoost
library(xgboost)

## boston housing dataset
df<-read.csv('BostonHousing_2.csv')
dim(df)
head(df)
colnames(df)[13] <- 'class'
df$class <- ifelse(df$class == 0, 'N', 'Y')
df$class <- factor(df$class)
sapply(df, class)
table(df$class)

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)
dim(train)
dim(test)

xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary
)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7)
)

set.seed(123)
xgbModel <- caret::train(x = train[, -13], 
                         y = train$class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         verbosity = 0,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test)
cm <- caret::confusionMatrix(pred, test$class)
cm

#############################################################################
## drug consumption dataset

df<-read.csv('drug_consumption_cannabis.csv')
colnames(df)[which(names(df) == "C6")] <- "class"
df$class <- ifelse(df$class ==  0, "N", "Y")
df$class <- as.factor(df$class)
df <- df[, -1]

set.seed(31)
split <- initial_split(df, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)
dim(test)

set.seed(123)
xgbModel <- caret::train(x = train[, -13], 
                         y = train$class,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         verbosity = 0,
                         trControl = xgb_control)
xgbModel
plot(xgbModel)

pred <- predict(xgbModel, test)
cm <- caret::confusionMatrix(pred, test$class)
cm

