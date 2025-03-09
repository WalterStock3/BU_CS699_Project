library(FSelector)
library(Boruta)
library(RWeka)
library(caret)
library(rsample)
library(rpart)

## set your working directory
setwd('C:/Courses/C2025/699/Slides-Spring2025/L2') 

## chi-square test
## read heart disease dataset into df
df <- read.csv('heart_disease_L2.csv') 
head(df)
ct <- table(df$chest_pain, df$class)
ct
chisq.test(ct)

## dimension reduction
## near zero variance
## read Frogs_MFCCs (Anruan Calls) dataset into df
df <- read.csv('Frogs_MFCCs.csv')
dim(df)
head(df)
nearZeroVar(df, names = TRUE)

## collinearity
corr <- cor(df[c(1:22)])
highCorr <- findCorrelation(corr, cutoff = 0.7, names = TRUE)
length(highCorr)
highCorr

## feature selection
df <- read.csv('bone_marrow.csv')
head(df)
dim(df)
sapply(df, class)

# cfs
subset <- cfs(survival_status ~., df)
subset

# Boruta
bone.marrow.boruta <- Boruta(survival_status~.,data=df)
bone.marrow.boruta
getSelectedAttributes(bone.marrow.boruta)
important_tentative <- names(bone.marrow.boruta$finalDecision[bone.marrow.boruta$finalDecision %in% 
                                                                c("Confirmed", "Tentative")]) 
important_tentative

# information gain
info.gain <- information.gain(survival_status ~ ., df)
info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
names(info.gain) <- c("Attribute", "Info Gain")
sorted.info.gain <- info.gain[order(-info.gain$"Info Gain"), ]
sorted.info.gain
# top 5
sorted.info.gain[1:5, ]

# small PCA example
ex.df <- read.csv("small_pca_ex.csv")
ex.df

# perform principal components anslysis on the dataset
# should standardize variables but, here, not standardized for illustration purpose
pc <- prcomp(ex.df, center = TRUE, scale = TRUE)
summary(pc)

# mapping matrix
m <- pc$rotation
m

# mapping: rotation matrix is multiplied to original dataset (ex.df * m)
# new dataset with PC's as new variables is created
new.df <- predict(pc, ex.df) 
new.df
cor(new.df)


## PCA Example with rpart decision tree
# read crime-data-2.csv
# downloaded from UCI ML Repository and modified for this example
df <- read.csv('crime-data-2.csv')
dim(df)
head(df)
table(df$Class)

df$Class <- factor(df$Class)
sapply(df, class)

# remove community name and state
df <- df[, -c(1,2)]
head(df)

# split dataset
set.seed(31)
split <- initial_split(df, prop = 0.66, strata = Class)
training <- training(split)
test <- testing(split)
dim(training)
dim(test)

# build rpart decision tree model using all attributes
rpart.model <- rpart(Class ~ . , data=training)
# test the model on the test dataset
pred <- predict(rpart.model, newdata = test, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = test$Class,
                                         positive = '1')
performance_measures

# apply PCA on the training dataset
pc <- prcomp(training[, -79], center = TRUE, scale = TRUE) # exclude class attribute
summary(pc)

# first map (project) original attributes to new attributes created by PCA
tr <- predict(pc, training)
tr <- data.frame(tr, training[79])
ts <- predict(pc, test)
ts <- data.frame(ts, test[79])

head(tr)
head(ts)

# Build model using first 17 components, test, get confusion matrix and accuracy
rpart.model <- rpart(Class~., data=tr[c(1:17, 79)])
pred <- predict(rpart.model, newdata = ts, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = ts$Class,
                                         positive = '1')
performance_measures

## sampling
x <- c(1:20)
x
srswor <- sample(x, 10, replace=FALSE)
srswor
srswr <- sample(x, 15, replace=TRUE)
srswr



