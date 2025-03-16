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
in_limit_missing_col_percent <- 0.05
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
in_limit_missing_row_percent <- 0.01
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
#    Split the dataset into training and testing datasets.

library(caret)
library(rsample)
set.seed(123)

split <- initial_split(df_processed, prop = 0.7, strata = "Class")
train <- training(split)
test <- testing(split)

print(paste("training dataset - dim:", dim(train)[1], ",", dim(train)[2]))
print(paste("testing dataset - dim:", dim(test)[1], ",", dim(test)[2]))
print(paste("train dataset - class distribution:",
            table(train$Class)[1], ",", table(train$Class)[2]))
print(paste("test dataset - class distribution:",
            table(test$Class)[1], ",", table(test$Class)[2]))

################################################################################
## 3 Create Balanced Training Dataset - Project Step 3
################################################################################

### 3 Create Balanced Training Dataset - Method 1 - Project Step 3

df_balanced1 <- train

### 3 Create Balanced Training Dataset - Method 2 - Project Step 3

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

### 4-1 Select Attributes - Method 1 - Project Step 4

# Check covariances
# Convert integer columns to numeric
df_numeric <- df_balanced1 %>%
  mutate(across(where(is.integer), as.numeric))

# Check for collinearity using a correlation matrix
correlation_matrix_full <- cor(df_numeric %>% select(-Class))
correlation_matrix <- correlation_matrix_full

# Identify the two variables that are most correlated
correlation_matrix[upper.tri(correlation_matrix, diag = TRUE)] <- NA
most_correlated_location <- which(abs(correlation_matrix) ==
                                    max(abs(correlation_matrix),
                                        na.rm = TRUE), arr.ind = TRUE)
most_correlated_vars <- colnames(correlation_matrix)[most_correlated]
most_correlated_correlation <- correlation_matrix[most_correlated_location]
print(paste("Most correlated:", most_correlated_vars[1],
            "and", most_correlated_vars[2], "at", most_correlated_correlation))

# Sum the correlations to decide which one to remove.
row_to_sum1 <- correlation_matrix_full[most_correlated_vars[1], ]
row_sum1 <- sum(row_to_sum1, na.rm = TRUE)

row_to_sum2 <- correlation_matrix_full[most_correlated_vars[2], ]
row_sum2 <- sum(row_to_sum2, na.rm = TRUE)

# Remove the variable with the highest sum of correlations
highly_correlated <- ifelse(row_sum1 > row_sum2, most_correlated_vars[1],
                            most_correlated_vars[2])

# Remove the highly correlated variable from df_balanced1
df_balanced1 <- df_numeric %>%
  select(-all_of(highly_correlated))

# Update df_balanced1 with reduced attributes
df_balanced1 <- df_numeric

# Optionally, remove highly correlated attributes
df_balanced1 <- df_numeric %>%
    select(-all_of(names(df_numeric)[highly_correlated]))

#### 4-1-1 Select Attributes - Method 1 - Balanced Dataset 1
df_balanced1_select1 <- df_balanced1 %>%
  select(Class, ANC1P, ANC2P) # Selecting Class and AGE as the additional column

#### 4-1-2 Select Attributes - Method 1 - Balanced Dataset 2
df_balanced2_select1 <- df_balanced2

### 4-2 Select Attributes - Method 2 - Project Step 4

#### 4-2-1 Select Attributes - Method 2 - Balanced Dataset 1
df_balanced1_select2 <- df_balanced1

#### 4-2-2 Select Attributes - Method 2 - Balanced Dataset 1
df_balanced2_select2 <- df_balanced2

### 4-3 Select Attributes - Method 3 - Project Step 4
#     ----------------------------------

df_balanced1_select3 <- df_balanced1

df_balanced2_select3 <- df_balanced2

## Models - Project Step 5
# Lecture 1 - Classification Slide:
# * Decision trees,
# * Naive Bayesian classification,
# * Support vector machines,
# * K-nearest neighbors,
# * Neural networks.

### Balanced Training Dataset - Model 1 Logistic Regression - Project Step 5

### Balanced Training Dataset - Model 2 K-Nearest Neighbors - Project Step 5

train_control <- trainControl(method = "cv",
                              number = 10) # 10-fold cross-validation

knn_model <- train(Class ~ ., data = df_balanced1_select1,
                   method = "knn",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneLength = 10)

print(knn_model)
plot(knn_model)

### Balanced Training Dataset - Model 3 Decision Tree - Project Step 5

### Balanced Training Dataset - Model 4 Random Forest - Project Step 5

### Balanced Training Dataset - Model 5 Support Vector Machine - Project Step 5

### Balanced Training Dataset - Model 6 Gradient Booston - Project Step 5
