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

print("df - dim")
dim(df) # 4318  117

# Update df Class to be a binary factor variable.
df$Class <- ifelse(df$Class == 'Yes', 1, 0)
df$Class <- as.factor(df$Class)

# Load the PUMS data dictionary
data_dict_loc <- "~/Source/BU_CS699_Project/CS699_Added_Artifacts/"
data_dict_file <- "PUMS_Data_Dictionary_2023.csv"
data_dict_df <- read.csv(paste(data_dict_loc, data_dict_file, sep = ""))

data_dict_names <- data_dict_df %>%
  filter(Record_Type_Name_or_Val == "Name") %>%
  select(Name = Record_Type_Name_or_Val, Value = Value_All)

data_dict_values

# Preview the data dictionary
head(data_dict_df)

# Example: Filter variables based on descriptions
# Assuming the data dictionary has columns 'Variable' and 'Description'
selected_vars <- data_dict %>%
  filter(grepl("income|employment", Description, ignore.case = TRUE)) %>%
  pull(Variable)

# Use the selected variables in your dataset
df_selected <- df %>%
  select(all_of(selected_vars))

### Remove columns with no info - iterative - purposefully including here
# All values are equal:
# ADJINC - Adjustment factor for income and earnings dollar amounts - all same
# RACNH - Native Hawaiian and Other Pacific Islander - all 0
# RT - Record Type - all are P for person records
df <- df %>%
  select(-c("STATE", "REGION", "DIVISION", "ADJINC", "RACNH", "RT"))

print("df - post no-info-column removal dim")
dim(df)

# Convert integer variables to numeric
df_numeric <- df %>% mutate(across(where(is.integer), as.numeric))

### Handle missing values

print("df - missing values")
sum(is.na(df_numeric)) # 141635

# Columns
missing_values_col_count <- sapply(df_numeric, function(x) sum(is.na(x)))
missing_values_col_percent <- (missing_values_col_count / nrow(df_numeric))

# input
in_limit_missing_col_percent <- 0.1

print("missing column percent limit")
in_limit_missing_col_percent

df_numeric_filt1_columns <- df_numeric %>%
  select(which(missing_values_col_percent <= in_limit_missing_col_percent))

print("df_filt1_columns - dim")
dim(df_numeric_filt1_columns)

# Rows - purposefully starting with the filtered dataset
df_numeric_filt2_rows <- df_numeric_filt1_columns %>%
  mutate(calc_missing_values_row_count = rowSums(is.na(.))) %>%
  mutate(calc_missing_values_row_percent = (calc_missing_values_row_count /
                                              ncol(df_numeric)))

# input
print("missing row percent limit")
in_limit_missing_row_percent <- 0.1

df_numeric_filt2_rows <- df_numeric_filt2_rows %>%
  filter(calc_missing_values_row_percent <= in_limit_missing_row_percent)

print("df_filt2_rows - dim")
dim(df_numeric_filt2_rows)

df_numeric_filt <- df_numeric_filt2_rows

df_numeric_filt_no_missing <- df_numeric_filt %>%
  mutate(
    across(where(is.numeric), ~ replace(., is.na(.), median(., na.rm = TRUE)))
  )

print("df_filt_no_missing - dim")
dim(df_numeric_filt_no_missing)

print("df_filt_no_missing - missing values")
sum(is.na(df_numeric_filt_no_missing)) # 141635

### Outliers

# Create boxplots for each numeric variable in the dataset
numeric_columns <- df_numeric_filt_no_missing %>% select(where(is.numeric))

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(names(numeric_columns), function(col) {
  ggplot(df_numeric_filt_no_missing, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(), 
          axis.ticks.x = element_blank())
})

# Arrange boxplots in a grid
boxplots <- boxplots[order(names(numeric_columns))]
grid.arrange(grobs = boxplots, ncol = 10)

df_processed <- df_numeric_filt_no_missing

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

dim(train) # 3022  65
dim(test) # 1296  65
table(train$Class)
table(test$Class)

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

#### balanced dataset 1
df_balanced1_select1 <- df_balanced1 %>%
  select(Class, ANC1P, ANC2P) # Selecting Class and AGE as the additional column

#### balanced dataset 2

df_balanced2_select1 <- df_balanced2

# Identify zero or near zero variance variables
nzv <- nearZeroVar(df_balanced2, saveMetrics = TRUE)
nzv
##### Near Zero Variance Variables
# Keeping HINS data as it is a binary variable.
# Keeeping MIL as military may be relevant classification.
# Keep NWAV as available for work may be relevant classification.

# Filter out variables with zero or near zero variance
df_balanced2_select1 <- df_balanced2 %>%
    select(-which(nzv$nzv))

### 4-2 Select Attributes - Method 2 - Project Step 4


df_balanced1_select2 <- df_balanced1 %>% 

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

df_balanced1_select1_model1

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

