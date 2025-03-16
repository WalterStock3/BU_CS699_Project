# Project Goal (Lecture 1): Generate a model to predict the likelihood of a
# person having difficulty living independently.

# Dataset (Lecture 1): Part of the 2023 American Community Survey modified by
# the class professor for this project.

# Target Variable: The Class variable represents difficulty living independently
# and is binary (Yes/No).

# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(caret)
library(rsample)
library(ROSE)

in_limit_missing_col_percent <- 0.01
in_limit_missing_row_percent <- 0.01
in_select1_cor_threshold <- 0.75

################################################################################
## 1 Preprocessing - Project Step 1
################################################################################

loc <- "~/Source/BU_CS699_Project/CS699_Provided_Artifacts/"
data_file <- "project_data.csv"
df <- read.csv(paste(loc, data_file, sep = ""))

# 4318  117
print(paste("df - dim:", dim(df)[1], ",", dim(df)[2]))
print(paste("df - total missing values:", sum(is.na(df)))) # 141635

# Update df Class to be a binary factor variable.
df$Class <- ifelse(df$Class == "Yes", 1, 0)
df$Class <- as.factor(df$Class)

# Load the PUMS data dictionary
data_dict_loc <- "~/Source/BU_CS699_Project/CS699_Added_Artifacts/"
data_dict_file <- "PUMS_Data_Dictionary_2023.csv"
data_dict_df <- read.csv(paste(data_dict_loc, data_dict_file, sep = ""))

data_dict_names <- data_dict_df %>%
  filter(Record_Type_Name_or_Val == "NAME") %>%
  select(Code = Record_Name, Name = Value_All, Description = Value_Description)

data_dict_vals <- data_dict_df %>%
  filter(Record_Type_Name_or_Val == "VAL") %>%
  select(Code = Record_Name, Value = Value_All, Description = Value_Description)

### Remove columns with no info - iterative - purposefully including here before
#   row removal
#    * ADJINC - Adjustment factor for income and earnings dollar amounts
#    * RACNH - Native Hawaiian and Other Pacific Islander - all 0
#    * RT - Record Type - all are P for person records
#    * SERIALNO - Serial Number
#    * SPORDER - Person Number - Spouse Order

df_processing <- df %>% select(-c("STATE", "REGION", "DIVISION",
                                  "ADJINC", "RACNH", "RT",
                                  "SERIALNO", "SPORDER"))

print(paste("df_processing - post specific column removal dim:",
            dim(df_processing)[1], ",", dim(df_processing)[2]))

# Columns
print(paste("df_processing - missing column percent limit:",
            in_limit_missing_col_percent))

missing_values_col_count <- sapply(df_processing, function(x) sum(is.na(x)))
missing_values_col_percent <- (missing_values_col_count / nrow(df_processing))

df_processing_filt_columns <- df_processing %>%
  select(which(missing_values_col_percent <= in_limit_missing_col_percent))

print(paste("df_processing - post_column_filt - dim:",
            dim(df_processing_filt_columns)[1], ",",
            dim(df_processing_filt_columns)[2]))

# Rows
print(paste("df_processing - missing row values percent limit:",
            in_limit_missing_row_percent))

df_processing_filt_rows <- df_processing_filt_columns %>%
  mutate(calc_missing_values_row_count = rowSums(is.na(.))) %>%
  mutate(calc_missing_values_row_percent = (calc_missing_values_row_count /
                                              ncol(df_processing_filt_columns)))

df_processing_filt_rows <- df_processing_filt_rows %>%
  filter(calc_missing_values_row_percent <= in_limit_missing_row_percent)

print(paste("df_processing - post_row_filt - dim:",
            dim(df_processing_filt_rows)[1], ",",
            dim(df_processing_filt_rows)[2]))
print(paste("df_processing - total missing values:",
            sum(is.na(df_processing_filt_rows))))

df_processing_filt_rows <- df_processing_filt_rows %>%
  select(-calc_missing_values_row_count, -calc_missing_values_row_percent)

# Retrieve the class of all columns in the dataset
column_classes <- sapply(df_processing_filt_rows, class)

### Outliers
# Create boxplots for each numeric variable in the dataset
integer_columns <- df_processing_filt_rows %>% select(where(is.integer))

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(names(integer_columns), function(col) {
  ggplot(integer_columns, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

# Arrange boxplots in a grid
boxplots <- boxplots[order(names(integer_columns))]
grid.arrange(grobs = boxplots, ncol = 10)

df_processed <- df_processing_filt_rows

################################################################################
## Split - Project Step 2
################################################################################

set.seed(123)

split <- initial_split(df_processed, prop = 0.7, strata = "Class")
train <- training(split)
test <- testing(split)

print(paste("training dataset - dim:", dim(train)[1], ",", dim(train)[2]))
print(paste("testing dataset - dim:", dim(test)[1], ",", dim(test)[2]))
print(paste("training dataset - class distribution:",
            table(train$Class)[1], ",", table(train$Class)[2]))
print(paste("testing dataset - class distribution:",
            table(test$Class)[1], ",", table(test$Class)[2]))

################################################################################
## 3 Create Balanced Training Dataset - Project Step 3
################################################################################

#-------------------------------------------------------------------------------
### 3.1 Create Balanced Training Dataset - Method 1 - Project Step 3
#-------------------------------------------------------------------------------

# Simply undersampling
df_balanced1 <- downSample(x = train[, -which(names(train) %in% "Class")],
                           y = train$Class)

print(paste("training balanced 1 dataset - dim:", dim(df_balanced1)[1],
            ",", dim(df_balanced1)[2]))

print(paste("training balanced 1 dataset - class distribution:",
            table(df_balanced1$Class)[1], ",",
            table(df_balanced1$Class)[2]))

#-------------------------------------------------------------------------------
### 3 Create Balanced Training Dataset - Method 2 - Project Step 3
#-------------------------------------------------------------------------------

#TODO: Complete balanced dataet 2 possibly with SMOTE

df_balanced2 <- train

################################################################################
## 4 Select Attributes - Project Step 4
################################################################################

#    Data Reduction - Dimension Reduction - Lecture 2 - Data Reduction Slides
#    * Remove irrelevant attributes
#    * Remove duplicate attributes
#    * Remove zero-variance attributes
#    * Remove attributes to avoid collinearity
#    * Feature selection

#-------------------------------------------------------------------------------
### 4-1 Select Attributes - Method 1 - Project Step 4
#-------------------------------------------------------------------------------

#### 4-1-1 Select Attributes - Method 1 - balanced dataset 1
#-------------------------------------------------------------------------------

# Collinearity

df_balanced1_select1 <- df_balanced1

repeat {
  df_numeric <- df_balanced1_select1 %>%
    mutate(across(where(is.integer), as.numeric))

  # Check for collinearity using a correlation matrix
  correlation_matrix_full <- cor(df_numeric %>% select(-Class))
  correlation_matrix <- correlation_matrix_full

  # Identify the two variables that are most correlated
  correlation_matrix[upper.tri(correlation_matrix, diag = TRUE)] <- NA
  most_correlated_location <- which(abs(correlation_matrix) ==
                                      max(abs(correlation_matrix),
                                          na.rm = TRUE), arr.ind = TRUE)
  most_correlated_vars <- colnames(correlation_matrix)[most_correlated_location]
  most_correlated_correlation <- correlation_matrix[most_correlated_location]

  # Break the loop if the highest correlation is less than a threshold.
  if (abs(most_correlated_correlation) <= in_select1_cor_threshold) {
    break
  }

  print(paste("Most correlated:", most_correlated_vars[1],
              "and", most_correlated_vars[2],
              "at", most_correlated_correlation))

  # Sum the correlations to decide which one to remove
  row_to_sum1 <- abs(correlation_matrix_full[most_correlated_vars[1],
                                             , drop = FALSE])
  row_sum1 <- sum(row_to_sum1, na.rm = TRUE)

  row_to_sum2 <- abs(correlation_matrix_full[most_correlated_vars[2],
                                             , drop = FALSE])
  row_sum2 <- sum(row_to_sum2, na.rm = TRUE)

  print(paste("Variable:", most_correlated_vars[1], "Row Sum:", row_sum1))
  print(paste("Variable:", most_correlated_vars[2], "Row Sum:", row_sum2))

  # Remove the variable with the highest sum of correlations
  highly_correlated <- ifelse(row_sum1 > row_sum2, most_correlated_vars[1],
                              most_correlated_vars[2])
  df_balanced1_select1 <- df_numeric %>%
    select(-all_of(highly_correlated))

  print(paste("Removed variable:", highly_correlated))
}

#### 4-1-2 Select Attributes - Method 1 - balanced dataset 2
#-------------------------------------------------------------------------------

df_balanced2_select1 <- df_balanced2

#-------------------------------------------------------------------------------
### 4-2 Select Attributes - Method 2 - Project Step 4
#-------------------------------------------------------------------------------

#### 4-2-1 Select Attributes - Method 2 - balanced dataset 1
#-------------------------------------------------------------------------------

df_balanced1_select2 <- df_balanced1

#### 4-2-2 Select Attributes - Method 2 - balanced dataset 2
#-------------------------------------------------------------------------------
df_balanced2_select2 <- df_balanced2

#-------------------------------------------------------------------------------
### 4-3 Select Attributes - Method 3 - Project Step 4
#-------------------------------------------------------------------------------

#### 4-3-1 Select Attributes - Method 3 - balanced dataset 1
#-------------------------------------------------------------------------------
df_balanced1_select3 <- df_balanced1

#### 4-3-2 Select Attributes - Method 3 - balanced dataset 2
#-------------------------------------------------------------------------------
df_balanced2_select3 <- df_balanced2

################################################################################
## 5 Models - Project Step 5
################################################################################

## Models - Project Step 5
# Lecture 1 - Classification Slide:
# * Decision trees,
# * Naive Bayesian classification,
# * Support vector machines,
# * K-nearest neighbors,
# * Neural networks.

#-------------------------------------------------------------------------------
### 5-1 Balanced Training Dataset - Model 1 Logistic Regression - Project Step 5
#-------------------------------------------------------------------------------

# Logistic Regression Model
logistic_model <- glm(Class ~ ., data = df_balanced1_select1, family = binomial)

# Summary of the model
summary(logistic_model)

# Predict on the test dataset
test_predictions <- predict(logistic_model, newdata = test, type = "response")

# Convert probabilities to binary predictions
test_predicted_class <- ifelse(test_predictions > 0.5, 1, 0)

# Confusion matrix
confusion_matrix <- table(Predicted = test_predicted_class, Actual = test$Class)
print(confusion_matrix)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy:", accuracy))

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         Actual
#Predicted   0   1
#        0 923  24
#        1 286  63

# TPR No
true_positive0 <- confusion_matrix[1, 1]
false_negative0 <- confusion_matrix[2, 1]
tpr0 <- true_positive0 / (true_positive0 + false_negative0)
print(paste("Class No True Positive Rate (TPR):", tpr0))

# TPR Yes
true_positive1 <- confusion_matrix[2, 2]
false_negative1 <- confusion_matrix[1, 2]
tpr1 <- true_positive1 / (true_positive1 + false_negative1)
print(paste("Class Yes True Positive Rate (TPR):", tpr1))

# Weighted Average TPR
weighted_average_tpr <- (tpr1 + tpr0) / 2
print(paste("Weighted Average True Positive Rate (TPR):", weighted_average_tpr))

# FPR No
false_positive0 <- confusion_matrix[1, 2]
true_negative0 <- confusion_matrix[2, 2]
fpr0 <- false_positive0 / (false_positive0 + true_negative0)
print(paste("Class No False Positive Rate (FPR):", fpr0))

# FPR Yes
false_positive1 <- confusion_matrix[2, 1]
true_negative1 <- confusion_matrix[1, 1]
fpr1 <- false_positive1 / (false_positive1 + true_negative1)
print(paste("Class Yes False Positive Rate (FPR):", fpr1))

# Weighted Average FPR
weighted_average_fpr <- (fpr1 + fpr0) / 2
print(paste("Weighted Average False Positive Rate (FPR):",
            weighted_average_fpr))

# Precision No
precision0 <- true_positive0 / (true_positive0 + false_positive0)
print(paste("Class No Precision:", precision0))

# Precision Yes
precision1 <- true_positive1 / (true_positive1 + false_positive1)
print(paste("Class Yes Precision:", precision1))

# Weighted Average Precision
weighted_average_precision <- (precision1 + precision0) / 2
print(paste("Weighted Average Precision:", weighted_average_precision))

# Recall No
recall0 <- true_positive0 / (true_positive0 + false_negative0)
print(paste("Class No Recall:", recall0))

# Recall Yes
recall1 <- true_positive1 / (true_positive1 + false_negative1)
print(paste("Class Yes Recall:", recall1))

# Weighted Average Recall
weighted_average_recall <- (recall1 + recall0) / 2
print(paste("Weighted Average Recall:", weighted_average_recall))

# F-measure No
f_measure0 <- (2 * precision0 * recall0) / (precision0 + recall0)
print(paste("Class No F-measure:", f_measure0))

# F-measure Yes
f_measure1 <- (2 * precision1 * recall1) / (precision1 + recall1)
print(paste("Class Yes F-measure:", f_measure1))

# Weighted Average F-measure
weighted_average_f_measure <- (f_measure1 + f_measure0) / 2
print(paste("Weighted Average F-measure:", weighted_average_f_measure))

# ROC
roc <- roc(test$Class, test_predictions)
roc_auc <- auc(roc)
print(paste("ROC AUC:", roc_auc))

# MCC
mcc <- (true_positive0 * true_positive1 - false_positive0 * false_negative1) /
  sqrt((true_positive0 + false_positive0) * (true_positive0 + false_negative1) *
         (true_positive1 + false_negative0) *
         (true_positive1 + false_negative1))
print(paste("Matthews Correlation Coefficient (MCC):", mcc))

# Kappa
kappa <- (accuracy - sum(rowSums(confusion_matrix) / sum(confusion_matrix)) /
            (1 - sum(rowSums(confusion_matrix) / sum(confusion_matrix))))
print(paste("Kappa:", kappa))

#-------------------------------------------------------------------------------
### 5-2 Balanced Training Dataset - Model 2 K-Nearest Neighbors - Project Step 5
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
### 5-3 Balanced Training Dataset - Model 3 Decision Tree - Project Step 5
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
### 5-4 Balanced Training Dataset - Model 4 Random Forest - Project Step 5
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
### 5-5 Balanced Training Dataset - Model 5 Support Vect Machine - Proj Step 5
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
### 5-6 Balanced Training Dataset - Model 6 Gradient Booston - Project Step 5
#-------------------------------------------------------------------------------
