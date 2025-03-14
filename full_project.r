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

## Preprocessing - Project Step 1
loc <- "~/Source/BU_CS699_Project/CS699_Provided_Artifacts/"
data_file <- "project_data.csv"
df <- read.csv(paste(loc, data_file, sep = ""))

print("df - dim")
dim(df) # 4318  117

# Update df Class to be a binary factor variable.
df$Class <- ifelse(df$Class == 'Yes', 1, 0)
df$Class <- as.factor(df$Class)

### Remove columns with no info - iterative - purposefully including here
df <- df %>%
  select(-c("STATE", "REGION", "DIVISION"))

print("df - post no-info-column removal dim")
dim(df) # 4318  117

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
    geom_boxplot()
})

# Arrange boxplots in a grid
grid.arrange(grobs = boxplots, ncol = 10)

## Split - Project Step 2

## Training Dataset - Create Balanced Training Dataset - Project Step 3

### Training Dataset - Create Balanced Training Dataset - Method 1 - Project Step 3

### Training Dataset - Create Balanced Training Dataset - Method 2 - Project Step 3

## Balance the Training Dataset - Select Attributes - Project Step 4

### Balance the Training Dataset - Select Attributes - Method 1 - Project Step 4

### Balance the Training Dataset - Select Attributes - Method 2 - Project Step 4

### Balance the Training Dataset - Select Attributes - Method 3 - Project Step 4

## Balanced Training Dataset - Models - Project Step 5

### Balanced Training Dataset - Model 1 - Project Step 5

### Balanced Training Dataset - Model 2 - Project Step 5

### Balanced Training Dataset - Model 3 - Project Step 5

### Balanced Training Dataset - Model 4 - Project Step 5

### Balanced Training Dataset - Model 5 - Project Step 5

### Balanced Training Dataset - Model 6 - Project Step 5

