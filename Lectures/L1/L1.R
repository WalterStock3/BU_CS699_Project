## Cirrhosis Patient Survival Prediction
# Target: Status, D (death), C (censored), CL (censored due to liver transplantation).
# Downloaded from UCI ML Repository and modified

library(tidyr)
library(dplyr)

######################################################################################################
# 

# set working directory
setwd("~/Source/BU_CS699_Project/Lectures/L1")

df <- read.csv('cirrhosis-1.csv')
dim(df)
head(df)

# Status has three values
# make it binary: D => 1, C or CL => 0 and remove Status

df$class <- ifelse(df$Status == 'D', 1, 0)
df <- df[, -1]
head(df)

# convert class to factor
df$class <- as.factor(df$class)
sapply(df, class)

# convert integer variables to numeric
df_numeric <- df %>% mutate_at(c('Age',
                           'Cholesterol', 
                           'Copper',
                           'Tryglicerides',
                           'Platelets',
                           'Stage'),
                           as.numeric)
sapply(df_numeric, class)
dim(df_numeric)

# handle missing values

temp <- df_numeric
dim(temp)

sum(is.na(temp)) # number of missing values in the whole dataset
sapply(temp, function(x) sum(is.na(x))) # missing values in individual columns

# 106 observations have missing values in multiple columns
# remove these observations

df_no_missing <- temp %>% drop_na(Drug)
dim(df_no_missing)
sapply(df_no_missing, class)
sum(is.na(df_no_missing)) 
sapply(df_no_missing, function(x) sum(is.na(x)))

# replace remaining missing values with median

df_no_missing <- df_no_missing %>%
  mutate(across(where(is.numeric), 
                ~ replace(., is.na(.), median(., na.rm = TRUE))))
sapply(df_no_missing, function(x) sum(is.na(x)))
sapply(df_no_missing, class)
dim(df_no_missing)

write.csv(df_no_missing, 'cirrhosis-no-missing-values.csv', row.names = FALSE)

# outliers
# plot boxpots of variables that seem to have extreme values

temp <- df_no_missing
dim(temp)

library(ggplot2)
library(gridExtra)
bil_box <- ggplot(temp, aes( x = "", y = Bilirubin)) + geom_boxplot()
chol_box <- ggplot(temp, aes( x = "", y = Cholesterol)) + geom_boxplot()
copper_box <- ggplot(temp, aes( x = "", y = Copper)) + geom_boxplot()
alk_box <- ggplot(temp, aes( x = "", y = Alk_Phos)) + geom_boxplot()
sgot_box <- ggplot(temp, aes( x = "", y = SGOT)) + geom_boxplot()
try_box <- ggplot(temp, aes( x = "", y = Tryglicerides)) + geom_boxplot()
plat_box <- ggplot(temp, aes( x = "", y = Platelets)) + geom_boxplot()
pro_box <- ggplot(temp, aes( x = "", y = Prothrombin)) + geom_boxplot()

grid.arrange(bil_box, chol_box,  copper_box, alk_box, 
             sgot_box, try_box, plat_box, pro_box,
             ncol = 4, nrow = 2)

# detect and remove outliers using the IQR method with lager outlier factor 6.0
# to be conservative (not to lose too many observations)

outlier_factor <- 6.0

cols <- c('Bilirubin', 'Cholesterol', 'Copper', 'Alk_Phos',
          'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin')

upper_bound <- function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  high_threshold = Q3 + outlier_factor * IQR
  return (high_threshold)
}

bil_ub <- upper_bound(temp$Bilirubin)
chol_ub <- upper_bound(temp$Cholesterol)
copper_ub <- upper_bound(temp$Copper)
alk_ub <- upper_bound(temp$Alk_Phos)
sgot_ub <- upper_bound(temp$SGOT)
try_ub <- upper_bound(temp$Tryglicerides)
plat_ub <- upper_bound(temp$Platelets)
pro_ub <- upper_bound(temp$Prothrombin)

cat(bil_ub, chol_ub, copper_ub, alk_ub, sgot_ub, try_ub, plat_ub, pro_ub)

df_no_outlier <- temp %>% filter(Bilirubin <= bil_ub & Cholesterol <= chol_ub &
                          Copper <= copper_ub & Alk_Phos <= alk_ub &
                          SGOT <= sgot_ub & Tryglicerides <= try_ub &
                          Platelets <= plat_ub & Prothrombin <= pro_ub)

dim(df_no_outlier)

df_processed <- df_no_outlier
write.csv(df_processed, 'processed_data.csv', row.names = FALSE)

library(caret)

# split into train and test sets
library(rsample)
set.seed(31)
split <- initial_split(df_processed, prop = 0.66, strata = class)
train <- training(split)
test <- testing(split)
dim(train)
dim(test)
table(train$class)
table(test$class)

# build rpart decision tree from train set
library(rpart)
library(rpart.plot)
model.tree <- rpart(class ~ ., data = train)
prp(model.tree, type = 1, extra = 1, under = TRUE, 
    split.font = 1, varlen = -10)

# test on test set
pred <- predict(model.tree, newdata = test, type = "class")
performance_measures  <- confusionMatrix(data=pred,
                                         reference = test$class,
                                         positive = '1')
performance_measures

## visualization

library(ggplot2)

df_processed <- read.csv('processed_data.csv')
temp <- df_processed
dim(temp)
head(temp)

# scatterplot (Tryglicerides vs. Cholesterol)
ggplot(temp, aes(x = Tryglicerides, y = Cholesterol)) + geom_point()

# barchart (Sex count)
ggplot(temp) + geom_bar(aes(x = Sex))

# histogram (Cholesterol)
ggplot(temp) + geom_histogram(aes(x = Cholesterol), binwidth = 100, 
                            col = "black", fill = "green")


# boxplot
ggplot(temp, aes(y = Cholesterol, x = class)) + geom_boxplot()

# heatmap with neumeric variables
library(gplots)

neumeric.df <- temp %>% select(where(is.numeric))
head(neumeric.df)

heatmap.2(cor(neumeric.df), Rowv = FALSE, Colv = FALSE,
          dendrogram = "none", cellnote = round(cor(neumeric.df), 2),
          notecol = "black", key = FALSE, trace = "none",
          margins = c(10,10))

## correlation example with iris dataset

iris.df<-read.csv('iris.csv')
head(iris.df)
iris.df$class <- as.factor(iris.df$class)
table(iris.df$class)
sapply(iris.df, class)
barplot(prop.table(table(iris.df$class)), col = rainbow(3), 
        ylim = c(0, 0.4), main = "Class Distribution")
cor(iris.df[c(1:4)])

# correlation heatmap
library(ggcorrplot)
correlation_matrix <- cor(iris.df[c(1:4)])
ggcorrplot(correlation_matrix)

# pair-wise scatterplot

pairs(iris.df[c(1:4)], 
      bg = rainbow(3)[iris.df$class],
      col = rainbow(3)[iris.df$class])

