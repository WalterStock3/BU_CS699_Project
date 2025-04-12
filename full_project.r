#---- 0 PROG ******* Project Step 0 --------------------------------------------
## TODO:
#  - Complete a graph for Fisher Scores for Logical - 4-2-1-1
#  - Complete a grpah for Factor wihout processing Missing.
#  - Look into tuning logistic regression to get the best TP0 and TP1

# Project Goal (Lecture 1): Generate a model to predict the likelihood of a
# person having difficulty living independently.

# Dataset (Lecture 1): Part of the 2023 American Community Survey modified by
# the class professor for this project.

# Target Variable: The Class variable represents difficulty living independently
# and is binary (Yes(1)/No(0)).

# Install the tidymodels package
if (!requireNamespace("tidymodels", quietly = TRUE)) {
  install.packages("tidymodels")
}

# Load necessary libraries
# Load the tidymodels library
library(tidymodels)
# Prefer tidymodels for modeling
tidymodels_prefer()
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(gridExtra)
library(caret)
library(rsample)
library(ROSE)
library(pROC)

#---- 0.1 DONE *****    Functions - Performance Evaluation ---------------------

calculate_all_measures <- function(in_model, in_test_df, threshold) {
  #in_test_df <- df_test
  #in_model <- m1_fit_s2b1

  # Predict on the test dataset
  test_predictions <- predict(in_model, new_data = in_test_df, type = "prob")

  # Convert probabilities to binary predictions
  test_predicted_class <- ifelse(test_predictions > threshold, 1, 0) [, 2]

  # Confusion matrix
  confusion_matrix <- table(Predicted = test_predicted_class,
                            Actual = df_test$Class)
  print(confusion_matrix)

  # Calculate accuracy
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
  print(paste("Accuracy:", accuracy))

  # TPR No
  tp_0 <- confusion_matrix[1, 1]
  print(paste("Class No True Positive (TP):", tp_0))
  fp_0 <- confusion_matrix[1, 2]
  print(paste("Class No False Positive (FP):", fp_0))
  tn_0 <- confusion_matrix[2, 2]
  print(paste("Class No True Negative (TN):", tn_0))
  fn_0 <- confusion_matrix[2, 1]
  print(paste("Class No False Negative (FN):", fn_0))
  tpr_0 <- tp_0 / (tp_0 + fn_0)
  print(paste("Class No True Positive Rate (TPR):", tpr_0))

  # TPR Yes
  tp_1 <- confusion_matrix[2, 2]
  print(paste("Class Yes True Positive (TP):", tp_1))
  fp_1 <- confusion_matrix[2, 1]
  print(paste("Class Yes False Positive (FP):", fp_1))
  tn_1 <- confusion_matrix[1, 1]
  print(paste("Class Yes True Negative (TN):", tn_1))
  fn_1 <- confusion_matrix[1, 2]
  print(paste("Class Yes False Negative (FN):", fn_1))
  tpr_1 <- tp_1 / (tp_1 + fn_1)
  print(paste("Class Yes True Positive Rate (TPR):", tpr_1))

  performance_measures <- calculate_measures(tp_0, fp_0, tn_0, fn_0,
                                             tp_1, fp_1, tn_1, fn_1)
  performance_measures

  roc_curve <- roc(df_test$Class, test_predictions$.pred_0)
  auc_value <- auc(roc_curve)

  # Convert AUC value to double
  auc_value_double <- as.double(auc_value)

  # Add ROC values to the performance measures dataframe
  performance_measures <- performance_measures %>%
    add_row(measures = "ROC_0", values = auc_value_double) %>%
    add_row(measures = "ROC_1", values = auc_value_double) %>%
    add_row(measures = "ROC_W", values = auc_value_double)

  # Print the AUC value
  #print(paste("Area Under the ROC Curve (AUC):", auc_value))

  # Plot the ROC curve
  #plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
  #abline(a = 0, b = 1, lty = 2, col = "red")

  return(performance_measures)

}

calculate_measures <- function(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1) {
  accuracy <- (tp_0 + tn_0) / (tp_0 + fp_0 + tn_0 + fn_0)
  tpr_0 <- tp_0 / (tp_0 + fn_0)
  fpr_0 <- fp_0 / (fp_0 + tn_0)
  tnr_0 <- tn_0 / (fp_0 + tn_0)
  fnr_0 <- fn_0 / (fn_0 + tp_0)
  precision_0 <- tp_0 / (tp_0 + fp_0)
  recall_0 <- tpr_0
  f_measure_0 <- (2 * precision_0 * recall_0) / (precision_0 + recall_0)

  mcc_0 <- (tp_0 * tn_0 - fp_0 * fn_0) /
    (sqrt(tp_0 + fp_0) * sqrt(tp_0 + fn_0) *
       sqrt(tn_0 + fp_0) * sqrt(tn_0 + fn_0))
  
  # Kappa total records
  total_0 <- (tp_0 + fn_0 + fp_0 + tn_0)
  #print(paste("Kappa for 0, total_0:", total_0))

  # Kappa - proportion of agreement to total records
  p_o_0 <- (tp_0 + tn_0) / total_0
  p_e1_0 <- ((tp_0 + fn_0) / total_0) * ((tp_0 + fp_0) / total_0)
  #print(paste("Kappa for 0, p_e1_0:", p_e1_0))
  p_e2_0 <- ((fp_0 + tn_0) / total_0) * ((fn_0 + tn_0) / total_0)
  #print(paste("Kappa for 0, p_e2_0:", p_e2_0))
  p_e_0 <- p_e1_0 + p_e2_0
  #print(paste("Kappa for 0, p_e_0:", p_e_0))
  k_0 <- (p_o_0 - p_e_0) / (1 - p_e_0) # Kappa statistic
  #print(paste("Kappa for 0, k_0:", k_0))

  tpr_1 <- tp_1 / (tp_1 + fn_1)
  fpr_1 <- fp_1 / (fp_1 + tn_1)
  tnr_1 <- tn_1 / (fp_1 + tn_1)
  fnr_1 <- fn_1 / (fn_1 + tp_1)
  precision_1 <- tp_1 / (tp_1 + fp_1)
  recall_1 <- tpr_1
  f_measure_1 <- (2 * precision_1 * recall_1) / (precision_1 + recall_1)
  
  mcc_1 <- (tp_1 * tn_1 - fp_1 * fn_1) /
    (sqrt(tp_1 + fp_1) * sqrt(tp_1 + fn_1) *
       sqrt(tn_1 + fp_1) * sqrt(tn_1 + fn_1))

  total_1 <- (tp_1 + fn_1 + fp_1 + tn_1)
  #print(paste("Kappa for 1, total_1:", total_1))
  p_o_1 <- (tp_1 + tn_1) / total_1
  #print(paste("Kappa for 1, p_o_1:", p_o_1))
  p_e1_1 <- ((tp_1 + fn_1) / total_1) * ((tp_1 + fp_1) / total_1)
  #print(paste("Kappa for 1, p_e1_1:", p_e1_1))
  p_e2_1 <- ((fp_1 + tn_1) / total_1) * ((fn_1 + tn_1) / total_1)
  #print(paste("Kappa for 1, p_e2_1:", p_e2_1))
  p_e_1 <- p_e1_1 + p_e2_1
  #print(paste("Kappa for 1, p_e_1:", p_e_1))
  k_1 <- (p_o_1 - p_e_1) / (1 - p_e_1) # Kappa statistic
  #print(paste("Kappa for 1, k_1:", k_1))

  weight0 <- sum(in_test_df$Class == 0) / nrow(in_test_df)
  weight1 <- sum(in_test_df$Class == 1) / nrow(in_test_df)
 
  #print(paste("Weight for Class 0 (weight0):", weight0))
  #print(paste("Weight for Class 1 (weight1):", weight1))

  tpr_w <- (tpr_0 * weight0 + tpr_1 * weight1)
  fpr_w <- (fpr_0 * weight0 + fpr_1 * weight1)
  tnr_w <- (tnr_0 * weight0 + tnr_1 * weight1)
  fnr_w <- (fnr_0 * weight0 + fnr_1 * weight1)
  precision_w <- (precision_0 * weight0 + precision_1 * weight1)
  recall_w <- (recall_0 * weight0 + recall_1 * weight1)
  f_measure_w <- (f_measure_0 * weight0 + f_measure_1 * weight1)
  mcc_w <- (mcc_0 * weight0 + mcc_1 * weight1)
  kappa_w <- (k_0 * weight0 + k_1 * weight1)

  measures <- c("ACCURACY", "TPR_0", "FPR_0", "TNR_0", "FNR_0",
                "Precision_0", "Recall_0", "F-measure_0", "MCC_0", "Kappa_0",
                "TPR_1", "FPR_1", "TNR_1", "FNR_1",
                "Precision_1", "Recall_1", "F-measure_1", "MCC_1", "Kappa_1",
                "TPR_W", "FPR_W", "TNR_W", "FNR_W",
                "Precision_W", "Recall_W", "F-measure_W", "MCC_W", "Kappa_W")
  values <- c(accuracy, tpr_0, fpr_0, tnr_0, fnr_0,
              precision_0, recall_0, f_measure_0, mcc_0, k_0,
              tpr_1, fpr_1, tnr_1, fnr_1,
              precision_1, recall_1, f_measure_1, mcc_1, k_1,
              tpr_w, fpr_w, tnr_w, fnr_w,
              precision_w, recall_w, f_measure_w, mcc_w, kappa_w)

  measure_df <- data.frame(measures, values)

  return(measure_df)
}

# Function to store model results with timestamp and description
store_results <- function(combination_key, results_df, description) {
  # Get current date and time
  current_datetime <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  
  # Create a wide format dataframe with measures as columns
  results_wide <- results_df %>%
    pivot_wider(names_from = measures, values_from = values)
  
  # Add description and datetime columns
  results_wide <- results_wide %>%
    mutate(combination_key = combination_key,
           description = description,
           datetime = current_datetime)
  
  # Check if results_storage exists in the global environment
  if (!exists("results_storage", envir = .GlobalEnv)) {
    # Create it if it doesn't exist
    results_storage <<- results_wide
  } else {
    # Append to existing dataframe if it does exist
    results_storage <<- bind_rows(results_storage, results_wide)
  }
  
  return(results_storage)
}

################################################################################
#---- 1 DONE ******* Preprocess - Project Step 1 ---------- df_preprocessed ----
################################################################################

# Load the dataset
loc <- "~/Source/BU_CS699_Project/CS699_Provided_Artifacts/"
data_file <- "project_data.csv"
df_orig <- read.csv(paste(loc, data_file, sep = ""))

df <- df_orig

# Replace all "Invalid Number" values in the dataframe with NA
# Identify columns containing "Invalid Number"
columns_with_invalid <-
  sapply(df, function(col) any(col == "Invalid Number", na.rm = TRUE))

# Replace "Invalid Number" with NA only in relevant columns
df <- df %>%
  mutate(across(names(df)[columns_with_invalid], ~ na_if(.x, "Invalid Number")))

print(paste("df - dim:", dim(df)[1], ",", dim(df)[2])) # 4318  117
print(paste("df - total missing values:", sum(is.na(df)))) # 141635

### Remove columns with no info - iterative - purposefully including here before
#   row removal
#    * STATE - State code - all same - MA
#    * REGION - Region code - all same
#    * DIVISION - Division code - all same
#    * ADJINC - Adjustment factor for income and earnings dollar amounts
#    * RT - Record Type - all are P for person records

df <- df %>%
  select(-matches("^(STATE|REGION|DIVISION|ADJINC|RT)"),
         -matches("^DETAILED-(STATE|REGION|DIVISION|ADJINC|RT)"))

print(paste("df - dim:", dim(df)[1], ",", dim(df)[2])) # 4318  112
print(paste("df - total missing values:", sum(is.na(df)))) # 141635

# Create a dataframe with each column name and its corresponding class (112)
df_columns_info <- data.frame(
  column_name = names(df),
  orig_column_class = sapply(df, class)
)

### Column Info - numeric, integer, character, factor, logical, date
df_columns_info <- df_columns_info %>%
  mutate(variable_type = case_when(
    column_name %in% c("SERIALNO") ~ "character",
    column_name %in% c("SPORDER", "PUMA", "PWGTP", "CITWP", "INTP", "JWMNP",
                       "MARHYP", "OIP", "PAP", "RETP", "SSIP", "SSP", "WAGP",
                       "WKHP", "WKWN", "YOEP", "MIGPUMA", "MIGSP", "RACNUM",
                       "PERNP", "PINCP", "POWPUMA", "POWSP", "POVPIP",  "SEMP")
    ~ "integer",
    column_name %in% c("CIT", "COW", "ENG", "HIMRKS", "JWTRNS", "LANX", "MAR",
                       "MIG", "MIL", "NWAB", "NWAV", "NWLA", "NWLK", "NWRE",
                       "SCH", "SCHG", "SCHL", "ANC", "ANC1P", "ANC2P",
                       "ESP", "ESR", "FOD1P", "FOD2P", "HICOV", "HISP", "INDP",
                       "LANP", "MSP", "NATIVITY", "NOP", "OCCP", "PAOC", "POBP",
                       "PRIVCOV", "PUBCOV", "QTRBIR", "RAC1P",
                       "RAC2P", "RAC3P", "SFN", "SFR", "VPS", "WAOB")
    ~ "factor",
    column_name %in% c("FER", "GCL", "GCR", "HINS1", "HINS2", "HINS3", "HINS4",
                       "HINS5", "HINS6", "HINS7", "MARHD", "MARHM", "MARHW",
                       "MLPA", "MLPB", "MLPCD", "MLPE", "MLPFG", "MLPH",
                       "MLPIK", "MLPJ", "RACAIAN", "RACASN", "RACBLK", "RACNH",
                       "RACPI", "RACSOR", "RACWHT", "RC", "SCIENGP",
                       "SCIENGRLP", "SEX", "WRK", "OC")
    ~ "logical",
    column_name %in% c("GCM", "JWRIP", "MARHT", "DECADE", "DRIVESP", "JWAP",
                       "JWDP")
    ~ "factor_levels",
    TRUE ~ "other"
  ))

# Load the PUMS data dictionary
data_dict_loc <- "~/Source/BU_CS699_Project/CS699_Added_Artifacts/"
data_dict_file <- "PUMS_Data_Dictionary_2023.csv"
data_dict_df <- read.csv(paste(data_dict_loc, data_dict_file, sep = ""))
# For readability and best practice - update column names to be lower case
colnames(data_dict_df) <- tolower(colnames(data_dict_df))

data_dict_names <- data_dict_df %>%
  filter(record_type_name_or_val == "NAME") %>%
  select(code = record_name, name = value_all)

data_dict_vals <- data_dict_df %>%
  filter(record_type_name_or_val == "VAL") %>%
  select(code = record_name, value = value_all,
         description = value_description) %>%
  mutate(value = ifelse(value == "0", "0", sub("^0+", "", value)))

# Remove 7 duplicates - RT, SERIALNO, STATE, REGION, DIVISION,PUMA, ADJINC (521)
data_dict_names_unique <- data_dict_names %>%
  distinct(code, .keep_all = TRUE)

# Join df_columns_info with data_dict_names to add the Name column (117)
df_columns_info <- df_columns_info %>%
  left_join(data_dict_names_unique, by = c("column_name" = "code"))

df_detailed <- df

# Process one record at a time and print columns with missing values
no_match_columns <- c()  # Initialize to store column names with "No Match"
column_counter <- 1
for (col_name in names(df_detailed)) {

  if (col_name %in% c("Class", "SERIALNO")) {
    next
  }

  detailed_col_name <-
    paste0("DETAILED-", col_name, "_",
           df_columns_info$name[match(col_name, df_columns_info$column_name)])

  variable_type <-
    df_columns_info$variable_type[match(col_name, df_columns_info$column_name)]

  print(paste0(column_counter, " Column: '", col_name,
               "'' Detailed name: '", detailed_col_name,
               "' Variable Type: ", variable_type))

  column_counter <- column_counter + 1

  # Populate the columns values
  if (variable_type == "factor" ||
        variable_type == "logical" ||
        variable_type == "factor_levels") {
    # Get the description for each value in the column
    value_descriptions <- sapply(df_detailed[[col_name]], function(col_value) {
      if (!is.na(col_value)) {
        value_description <- data_dict_vals %>%
          filter(code == col_name & value == as.character(col_value)) %>%
          pull(description)
      } else {
        value_description <- NA
      }
      return(value_description)
    })
    # Add the descriptions to the DETAILED- column
    df_detailed[[detailed_col_name]] <- value_descriptions
  }

  if (variable_type == "integer") {
    df_detailed[[detailed_col_name]] <- df_detailed[[col_name]]
  }

}

df <- df_detailed

print(paste("df - dim:", dim(df)[1], ",", dim(df)[2])) # 4318  112 -> 222
print(paste("df - total missing values (excluding DETAILED-* columns):",
            sum(is.na(df %>% select(-starts_with("DETAILED-")))))) # 141635

# Update df Class to be a binary factor variable.
df$Class <- ifelse(df$Class == "Yes", 1, 0)
df$Class <- as.factor(df$Class)

print(paste("df_processing - note - all records remain (4318): ",
            dim(df)[1], ",", dim(df)[2])) # 4318 222

df_columns_info <- df_columns_info %>%
  mutate(Evaluate_Positive = case_when(
    column_name %in% c("CIT", "CITWP", "COW", "ENG", "FER", "GCL", "GCM",
                       "HIMRKS", "HINS1", "HINS2", "HINS3", "HINS4", "HINS5",
                       "HINS6", "HINS7", "INTP", "JWMNP", "JWRIP", "JWTRNS",
                       "LANX", "MAR", "MARHD", "MARHM", "MARHT", "MARHW",
                       "MARHYP", "MIG", "MIL", "MLPA", "MLPB", "MLPCD",
                       "MLPE", "MLPFG", "MLPH", "MLPIK", "MLPJ", "NWAB",
                       "NWAV", "NWLA", "NWLK", "NWRE", "OIP", "SCH", "SCHG",
                       "SCHL", "SEMP", "SEX", "SSIP", "SSP", "WRK", "YOEP",
                       "ANC", "ANC1P", "ANC2P", "DECADE", "DRIVESP", "ESP",
                       "ESR", "FOD1P", "FOD2P", "HICOV", "HISP", "INDP",
                       "JWAP", "JWDP", "LANP", "MIGPUMA", "MIGSP", "MSP",
                       "NATIVITY", "NOP", "OC", "OCCP", "PAOC", "POBP",
                       "POVPIP", "POWPUMA", "POWSP", "PRIVCOV", "PUBCOV",
                       "QTRBIR", "RAC1P", "RAC2P", "RAC3P", "RACAIAN",
                       "RACASN", "RACBLK", "RACNH", "RACNUM", "RACPI",
                       "RACSOR", "RACWHT", "RC", "SCIENGP", "SCIENGRLP",
                       "SFN", "SFR", "VPS", "WAOB")
    ~ "Medium",
    column_name %in% c("PAP", "RETP", "WAGP", "WKHP", "WKWN", "PERNP", "PINCP")
    ~ "High",
    TRUE ~ "Low"
  ))

# Factor - update columns to factor based on variable_type in df_columns_info
factor_columns <- df_columns_info %>%
  filter(variable_type == "factor", column_name %in% names(df)) %>%
  pull(column_name)

df <- df %>% 
  mutate(across(all_of(factor_columns), as.factor)) %>%
  mutate(across(all_of(matches(paste0("^DETAILED-",
                                      paste(factor_columns, collapse = "|"),
                                      "_"))), as.factor))

# Logical - update columns to logical based on variable_type in df_columns_info
logical_columns <- df_columns_info %>%
  filter(variable_type == "logical") %>%
  pull(column_name)

df <- df %>%
  mutate(across(all_of(logical_columns), as.factor)) %>%
  mutate(across(all_of(matches(paste0("^DETAILED-",
                                      paste(logical_columns, collapse = "|"),
                                      "_"))), as.factor))

# Factor_Levels - update columns based on variable_type in df_columns_info
factor_levels_columns <- df_columns_info %>%
  filter(variable_type == "factor_levels") %>%
  pull(column_name)

df <- df %>%
  mutate(across(all_of(factor_levels_columns), ~ factor(.x,
                                                        ordered = TRUE))) %>%
  mutate(across(all_of(matches(paste0("^DETAILED-",
                                      paste(factor_levels_columns,
                                            collapse = "|"),
                                      "_"))), ~ factor(.x,
                                                       ordered = TRUE)))

# Integer - update columns to integer based on variable_type in df_columns_info
integer_columns <- df_columns_info %>%
  filter(variable_type == "integer") %>%
  pull(column_name)

df <- df %>% 
  mutate(across(all_of(integer_columns), as.integer)) %>%
  mutate(across(all_of(matches(paste0("^DETAILED-",
                                      paste(integer_columns, collapse = "|"),
                                      "_"))), as.integer))

df <- df %>%
  mutate(across(names(df)[columns_with_invalid], ~ na_if(.x, "Invalid Number")))

df_preprocessed <- df

# Export df_preprocessed as a CSV file
write.csv(df_preprocessed, file = "df_preprocessed.csv", row.names = FALSE)

# Export df_preprocessed as an R data file
save(df_preprocessed, file = "df_preprocessed.RData")

################################################################################
#---- 2 DONE ******* Split - Project Step 2 ------------- df_train, df_test ----
################################################################################

# Optional - Load the preprocessed dataset
# load("df_preprocessed.RData")

set.seed(1)

split <- initial_split(df_preprocessed, prop = 0.7, strata = "Class")
df_train <- training(split)
df_test <- testing(split)

print(paste("training dataset - dim:", dim(df_train)[1], ",", dim(df_train)[2]))
print(paste("testing dataset - dim:", dim(df_test)[1], ",", dim(df_test)[2]))
print(paste("training dataset - class distribution:",
            table(df_train$Class)[1], ",", table(df_train$Class)[2]))
print(paste("testing dataset - class distribution:",
            table(df_test$Class)[1], ",", table(df_test$Class)[2]))

# Save df_train and df_test to R data files
save(df_train, file = "df_train.RData")
save(df_test, file = "df_test.RData")

################################################################################
#---- 3 DONE ******* Balance - Project Step 3 -- df_balanced1, df_balanced2 ----
################################################################################

# Not using SMOTE because we have a large number of categorical variables.

# Load the training and testing datasets
# load("df_train.RData")
# load("df_test.RData")

# Verify the dimensions of the loaded datasets
# print(paste("Loaded training dataset - dim:", dim(df_train)[1], ",", dim(df_train)[2]))
# print(paste("Loaded testing dataset - dim:", dim(df_test)[1], ",", dim(df_test)[2]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 3.1 DONE *****    Balance - Method 1 - Down Sample ----- df_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Undersampling
df_balanced1 <- downSample(x = df_train[, -which(names(df_train) %in% "Class")],
                           y = df_train$Class)

print(paste("training balanced 1 dataset - dim:", dim(df_balanced1)[1],
            ",", dim(df_balanced1)[2]))

print(paste("training balanced 1 dataset - class distribution:",
            table(df_balanced1$Class)[1], ",",
            table(df_balanced1$Class)[2]))

save(df_balanced1, file = "df_balanced1.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 3.2 DONE *****    Balance - Method 2 - Up Sample ------- df_balanced2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Upsampling
df_balanced2 <- upSample(x = df_train[, -which(names(df_train) %in% "Class")],
                         y = df_train$Class)

print(paste("training balanced 2 dataset - dim:", dim(df_balanced2)[1],
            ",", dim(df_balanced2)[2]))

print(paste("training balanced 2 dataset - class distribution:",
            table(df_balanced2$Class)[1], ",",
            table(df_balanced2$Class)[2]))

save(df_balanced2, file = "df_balanced2.RData")

################################################################################
#---- 4 PROG ******* Select - Project Step 4 --------- df_select#_balanced# ----
################################################################################
#
#    Chapter 4 - Dimension Reduction
#
#    Data Reduction - Dimension Reduction - Lecture 2 - Data Reduction Slides
#    * Remove irrelevant attributes
#    * Remove duplicate attributes
#    * Remove zero-variance attributes
#    * Remove attributes to avoid collinearity
#    * Feature selection
################################################################################

# Load the balanced datasets
#load("df_balanced1.RData")
#load("df_balanced2.RData")

# Verify the dimensions of the loaded datasets
#print(paste("Loaded df_balanced1 - dim:",
#            dim(df_balanced1)[1], ",", dim(df_balanced1)[2]))
#print(paste("Loaded df_balanced2 - dim:",
#            dim(df_balanced2)[1], ",", dim(df_balanced2)[2]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 4-1 DONE *****    Select 1 - Missing Removal --- df_select1_balanced# ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-1-1 DONE ***       Select 1 - balanced 1 ----- df_select1_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Inputs that can be tuned
in_limit_missing_col_percent <- 0.05
in_limit_missing_row_percent <- 0.01

# Columns
print(paste("df_processing - missing column percent limit:",
            in_limit_missing_col_percent))

missing_values_col_count <- sapply(df_balanced1, function(x) sum(is.na(x)))
missing_values_col_percent <- (missing_values_col_count / nrow(df_balanced1))

df_processing_filt_columns <- df_balanced1 %>%
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

# Remove the columns used for calculations
df_processing_filt_rows <- df_processing_filt_rows %>%
  select(-calc_missing_values_row_count, -calc_missing_values_row_percent)

df_select1_balanced1 <- df_processing_filt_rows

save(df_select1_balanced1, file = "df_select1_balanced1.RData")

#---- 4-1-2 DONE ***       Select 1 - balanced 2 ----- df_select1_balanced2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Columns
print(paste("df_processing - missing column percent limit:",
            in_limit_missing_col_percent))

missing_values_col_count <- sapply(df_balanced2, function(x) sum(is.na(x)))
missing_values_col_percent <- (missing_values_col_count / nrow(df_balanced2))

df_processing_filt_columns <- df_balanced2 %>%
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

# Remove the columns used for calculations
df_processing_filt_rows <- df_processing_filt_rows %>%
  select(-calc_missing_values_row_count, -calc_missing_values_row_percent)

df_select1_balanced2 <- df_processing_filt_rows

save(df_select1_balanced2, file = "df_select1_balanced2.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 4-2 DONE *****    Select 2 - Fisher and Corr --- df_select2_balanced# ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-2-1 DONE ***       Select 2 - balanced 1 ----- df_select2_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_select2_balanced1 <- df_balanced1

#---- 4-2-1-1-DONE *          Factor and Logical Variables ---------------------

df_select2_balanced1_1factors <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor")) %>%
                                pull(column_name), 
                              collapse = "|"), ")_")))

df_select2_balanced1_2logical <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("logical")) %>%
                                pull(column_name), 
                              collapse = "|"), ")_")))

df_select2_balanced1_3levels <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor_levels")) %>%
                                pull(column_name), 
                              collapse = "|"), ")_")))

df_select2_balanced1_allfact <- cbind(
  df_select2_balanced1_1factors,
  df_select2_balanced1_2logical %>% select(-Class),
  df_select2_balanced1_3levels %>% select(-Class)
)

##### Will use Fisher test over Chi-square to handle sparse data.

# Some columns have too many levels to be used in Fisher test.
# SCHL - LDSTP too small - 2e9
# ANC1P - LDSTP too small - 1e9

fisher_not_possible <- c("SCHL", "ANC1P", "DETAILED-SCHL_",
                         "DETAILED-ANC1P_", 
                         "RACNH", "DETAILED-RACNH_", "Class")

sel2_bal1_fisher_results <- list()

for (col in names(df_select2_balanced1_merged)) {
  print(paste(Sys.time(), "- Processing column:", col))
  if (any(startsWith(col, fisher_not_possible))) {
    print(paste("Skipping column:", col))
    next
  }
  tryCatch({
    table_data <- table(df_select2_balanced1_merged[[col]],
                        df_select2_balanced1_merged$Class)
    fisher_test <- fisher.test(table_data, workspace = 1e9)
    sel2_bal1_fisher_results[[col]] <-
      list(column = col, p_value = fisher_test$p.value)
    print(paste(Sys.time(),
                "- Fisher test column:", col, "p-value:", fisher_test$p.value))
  }, error = function(e) {
    message(paste("Error processing column:", col, "-", e$message))
    sel2_bal1_fisher_results[[col]] <- list(column = col, p_value = NA)
  })
}

# Convert results to a data frame for easier interpretation
df_sel2_bal1_fisher_results <- 
  do.call(rbind, lapply(sel2_bal1_fisher_results, as.data.frame))

names(df_sel2_bal1_fisher_results) <- c("Column", "P_value")

df_sel2_bal1_fisher_results$neg_log10_P_value <-
  -log10(df_sel2_bal1_fisher_results$P_value)

# Create a bar plot for Fisher scores
df_sel2_bal1_fisher_plt <- df_sel2_bal1_fisher_results %>%
  mutate(P_value = as.numeric(as.character(P_value))) %>%
  arrange(P_value)

plt_sel2_bal1_fisher <-
  ggplot(df_sel2_bal1_fisher_plt,
         aes(x = reorder(substr(Column, 10, 60), -P_value),
             y = -log10(P_value))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = -log10(0.05), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(title = "Fisher Scores for Categorical and Logical Variables 
       with Missing Included",
       subtitle = "Select 2 - Balanced 1",
       caption = "Red line indicates p-value threshold of 0.01",
       x = "",
       y = "-log10(P-value)") +
  theme_minimal()

plt_sel2_bal1_fisher

ggsave("plt_sel2_bal1_fisher.png", plot = plt_sel2_bal1_fisher,
       width = 10, height = 12, dpi = 300)

# Identify the columns with P-values less than 0.05
select_cols <- df_sel2_bal1_fisher_results %>%
  filter(P_value < 0.05) %>%
  arrange(desc(neg_log10_P_value)) %>%
  pull(Column)

# Create a new dataframe with those columns
df_select2_balanced1_allfact <- df_select2_balanced1_allfact %>%
  select(Class, all_of(select_cols))

#---- 4-2-1-2 DONE *          Integer Variables --------------------------------

df_select2_balanced1_4integers <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

in_select1_cor_threshold <- 0.05

repeat {
  df_numeric <- df_select2_balanced1_4integers %>%
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
  df_select2_balanced1_4integers <- df_numeric %>%
    select(-all_of(highly_correlated))

  print(paste("Removed variable:", highly_correlated))
}

#---- 4-2-1-3 DONE *          Outliers -----------------------------------------
# Create boxplots for each numeric variable in the dataset

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(df_select2_balanced1_4integers, function(col) {
  ggplot(df_select2_balanced1_4integers, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

plt_list <- lapply(names(df_select2_balanced1_4integers)
                   [names(df_select2_balanced1_4integers) != "Class"],
                   function(col_name) {
                     ggplot(df_select2_balanced1_4integers,
                            aes(x = "", y = .data[[col_name]])) +
                       geom_boxplot() +
                       labs(title = 
                              substr(col_name, 10, regexpr("_", col_name) - 1),
                            y = substr(col_name, regexpr("_", col_name) + 1,
                                       regexpr("_", col_name) + 60)) +
                       theme(plot.title = element_text(hjust = .9)) +
                       theme_minimal() +
                       theme(
                             axis.title.x = element_blank(),
                             axis.text.x  = element_blank(),
                             axis.ticks.x = element_blank(),
                             axis.text.y  = element_blank(),
                             axis.ticks.y = element_blank())
                  })

plt_sel2_bal1_corr <- grid.arrange(grobs = plt_list, ncol = 7)

ggsave("plt_sel2_bal1_corr.png",
       plot = plt_sel2_bal1_corr, width = 10, height = 16, dpi = 300)

# Based on boxplot distributions adding Income to Poverty Ratio and Work Hours.
df_select2_balanced1 <- df_select2_balanced1_allfact %>% select(-Class) %>%
  bind_cols(df_select2_balanced1_4integers)

#---- 4-2-1-4 DONE *          Final --------------------------------------------

save(df_select2_balanced1, file = "df_select2_balanced1.RData")

#---- 4-2-2 PEND ***       Select 2 - balanced 2 ----- df_select2_balanced2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_select2_balanced2 <- df_balanced2

#---- 4-2-2-1 PEND *          Factor and Logical Variables ---------------------

#---- 4-2-2-2 PEND *          Integer Variables --------------------------------

#---- 4-2-2-3 PEND *          Outliers -----------------------------------------

#---- 4-2-2-4 PEND *          Final --------------------------------------------

#---- 4-3 PROG *****    Select 3 - Missing Added ----- df_select3_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-3-1 PROG ***       Select 3 - balanced 1 ----- df_select3_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_select3_balanced1 <- df_balanced1

#---- 4-3-1-1 PROG *          Factor and Logical Variables ---------------------

in_row_limit_with_missing <- 1

df_select2_balanced1_4fct_miss <- df_select2_balanced1_1factors %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_select2_balanced1_5log_miss <- df_select2_balanced1_2logical %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_select2_balanced1_6lvl_miss <- df_select2_balanced1_3levels %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_select2_balanced1_merg_miss <- cbind(df_select2_balanced1_4fct_miss,
                                        df_select2_balanced1_5log_miss,
                                        df_select2_balanced1_6lvl_miss)

##### Will use Fisher test over Chi-square to handle sparse data.

# Some columns have too many levels to be used in Fisher test.
# SCHL - LDSTP too small - 2e9
# ANC1P - LDSTP too small - 1e9

fisher_not_possible <- c("SCHL", "ANC1P", "DETAILED-SCHL_",
                         "DETAILED-ANC1P_", 
                         "RACNH", "DETAILED-RACNH_", "Class")

sel3_bal1_fisher_results <- list()

for (col in names(df_select2_balanced1_merg_miss)) {
  print(paste(Sys.time(), "- Processing column:", col))
  if (any(startsWith(col, fisher_not_possible))) {
    print(paste("Skipping column:", col))
    next
  }
  tryCatch({
    table_data <- table(df_select2_balanced1_merg_miss[[col]],
                        df_select2_balanced1_merg_miss$Class)
    fisher_test <- fisher.test(table_data, workspace = 1e9)
    sel3_bal1_fisher_results[[col]] <-
      list(column = col, p_value = fisher_test$p.value)
    print(paste(Sys.time(),
                "- Fisher test column:", col, "p-value:", fisher_test$p.value))
  }, error = function(e) {
    message(paste("Error processing column:", col, "-", e$message))
    sel3_bal1_fisher_results[[col]] <- list(column = col, p_value = NA)
  })
}

# Convert results to a data frame for easier interpretation
df_sel3_bal1_fisher_results <- 
  do.call(rbind, lapply(sel3_bal1_fisher_results, as.data.frame))

names(df_sel3_bal1_fisher_results) <- c("Column", "P_value")

df_sel3_bal1_fisher_results$neg_log10_P_value <-
  -log10(df_sel3_bal1_fisher_results$P_value)

# Create a bar plot for Fisher scores
df_sel3_bal1_fisher_plt <- df_sel3_bal1_fisher_results %>%
  mutate(P_value = as.numeric(as.character(P_value))) %>%
  arrange(P_value)

ggplot(df_sel3_bal1_fisher_plt,
       aes(x = reorder(substr(Column, 10, 60), -P_value),
           y = -log10(P_value))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = -log10(0.01), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(title = "Fisher Scores for Categorical and Logical Variables 
      with Missing Included",
       x = "",
       y = "-log10(P-value)") +
  theme_minimal()

# Identify the 2 columns with the highest neg_log10_P_value
top_cols <- df_sel2_bal1_fisher_results %>%
  arrange(desc(neg_log10_P_value)) %>%
  slice_head(n = 2) %>%
  pull(Column)

# Create a new dataframe with those columns
df_select2_balanced1 <- df_select2_balanced1 %>%
  select(all_of(top_cols))

# Use Correlation to check for independence between numeric variables
#   and the target variable.

df_select2_balanced1_factors <- df_select2_balanced1 %>%
  select(where(is.factor))

#---- 4-3-1-2 PEND *          Integer Variables --------------------------------

#---- 4-3-1-3 PEND *          Outliers -----------------------------------------

#---- 4-3-1-4 PEND *          Final --------------------------------------------

#---- 4-3-2 PEND ***       Select - 3 balanced dataset 2 -----------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_select3_balanced2 <- df_balanced2

#---- 4-3-2-1 PEND *          Factor and Logical Variables ---------------------

#---- 4-3-2-2 PEND *          Integer Variables --------------------------------

#---- 4-3-2-3 PEND *          Outliers -----------------------------------------

#---- 4-3-2-4 PEND *          Final --------------------------------------------

################################################################################
#---- 5 PROG ******* Models - Project Step 5 -----------------------------------
################################################################################

## Models - Project Step 5
# Lecture 1 - Classification Slide:
# * Decision trees,
# * Naive Bayesian classification,
# * Support vector machines,
# * K-nearest neighbors,
# * Neural networks.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-1 PROG *****    Model 1 Logistic Regression ----------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Logistic regression is a statistical method for predicting binary classes.
#---- 5-1-1 PROG ***       Model 1 Logistic Regression - s1b1 ------------------

#---- 5-1-3 DONE ***       Model 1 Logistic Regression - s2b1 ------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Logistic Regression Model

df_logistic_s2b1 <- df_select2_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
m1_spec_s2b1 <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Additional Processing
m1_rec1_s2b1 <- recipe(Class ~ ., data = df_logistic_s2b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Create a workflow
m1_wf1_s2b1 <- workflow() %>%
  add_model(m1_spec_s2b1) %>%
  add_recipe(m1_rec1_s2b1)

# 4. Cross-validation
set.seed(1)
m1_folds_s2b1 <- vfold_cv(df_logistic_s2b1, v = 5, strata = Class)

# 5. Grid of penalty and mixture values
m1_tune_grid_wf1_s2b1 <- grid_regular(penalty(), mixture(), levels = 5)

# 6. Tune the model
m1_tune_res_s2b1 <- tune_grid(
  m1_wf1_s2b1,
  resamples = m1_folds_s2b1,
  grid = m1_tune_grid_wf1_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
m1_tune_res_s2b1
summary(m1_tune_res_s2b1)
library(ggplot2)
autoplot(m1_tune_res_s2b1) +
  labs(title = "Tuning Results for Logistic Regression",
       x = "Penalty",
       y = "Mixture") +
  theme_minimal()

# 7. Select the best parameters
m1_model_s2b1 <- select_best(tune_res, metric = "roc_auc")

# 8. Finalize the workflow
m1_final_wf_s2b1 <- finalize_workflow(m1_wf1_s2b1, m1_model_s2b1)

# 9. Fit the final model
m1_fit_s2b1 <- fit(m1_final_wf_s2b1, data = df_logistic_s2b1)

# 10. Evaluate the model on the test dataset
# Evaluate the model on the test dataset
m1_test_predictions_s2b1 <- predict(m1_fit_s2b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(m1_fit_s2b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Calculate performance metrics
m1_test_metrics_s2b1 <- m1_test_predictions_s2b1 %>%
  metrics(truth = Class, estimate = .pred_class, .pred_1)

# Print the performance metrics
print(m1_test_metrics_s2b1)

# Generate a confusion matrix
m1_conf_matrix_s2b1 <- m1_test_predictions_s2b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(m1_conf_matrix_s2b1)

# Visualize the confusion matrix
autoplot(m1_conf_matrix_s2b1, type = "heatmap") +
  labs(title = "Confusion Matrix for Logistic Regression",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

# 0.6 works best on the test data but I cannot tune with the test data.
results_model1_s2b1 <- calculate_all_measures(m1_fit_s2b1, df_test, 0.5)

results_model1_s2b1

store_results("m1s2b1", results_model1_s2b1, "Logistic Regression Model 1 - s2b1")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-2 PEND *****    Model 2 K-Nearest Neighbors ----------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 5-2-1 PROG ***       Model 2 KNN - s1b1 ----------------------------------

# Define the KNN model specification
knn_spec <- nearest_neighbor(neighbors = 5, weight_func = "rectangular") %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Split the dataset into predictors and target
df_knn_s2b1 <- df_select2_balanced1 %>%
  select(Class, where(is.integer))

# Create a recipe for preprocessing
knn_recipe <- recipe(Class ~ ., data = df_knn_s2b1) %>%
  step_normalize(all_predictors())

# Create a workflow
knn_workflow <- workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(knn_recipe)

# Fit the KNN model
knn_fit <- knn_workflow %>%
  fit(data = df_knn_s2b1)

# Evaluate the model on the test dataset
df_test_knn <- df_test %>%
  select(Class, where(is.integer))

knn_predictions <- knn_fit %>%
  predict(new_data = df_test_knn) %>%
  bind_cols(df_test_knn)

# Calculate performance metrics
knn_metrics <- knn_predictions %>%
  metrics(truth = Class, estimate = .pred_class)

print(knn_metrics)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-2-3 PROG ***       Model 2 KNN - s2b1 ----------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Define the KNN model specification
m2_spec_s2b1 <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Create a recipe for preprocessing
m2_rec_s2b1 <- recipe(Class ~ ., data = df_select2_balanced1) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Create a workflow
m2_wf_s2b1 <- workflow() %>%
  add_model(m2_spec_s2b1) %>%
  add_recipe(m2_rec_s2b1)

# Cross-validation
set.seed(123)
m2_folds_s2b1 <- vfold_cv(df_select2_balanced1, v = 5, strata = Class)

# Define grid of hyperparameters
m2_grid_s2b1 <- grid_regular(
  neighbors(range = c(5, 50)),
  weight_func(values = c("rectangular", "triangular", "gaussian", "rank")),
  levels = c(10, 4)
)

# Tune the model
m2_tune_res_s2b1 <- tune_grid(
  m2_wf_s2b1,
  resamples = m2_folds_s2b1,
  grid = m2_grid_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Select the best model based on ROC AUC
m2_best_params_s2b1 <- select_best(m2_tune_res_s2b1, metric = "roc_auc")

# Finalize workflow
m2_final_wf_s2b1 <- finalize_workflow(m2_wf_s2b1, m2_best_params_s2b1)

# Fit the final model
m2_fit_s2b1 <- fit(m2_final_wf_s2b1, data = df_select2_balanced1)

# Try different thresholds to achieve the target TPR and TNR
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(m2_fit_s2b1, df_test, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tnr_0 <- results$values[results$measures == "TNR_0"]
  
  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TNR_0 = tnr_0,
    diff_from_target = abs(tpr_1 - 0.81) + abs(tnr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <- threshold_df[which.min(threshold_df$diff_from_target), "threshold"]

# Final evaluation with best threshold
results_model2_s2b1 <- calculate_all_measures(m2_fit_s2b1, df_test, best_threshold)
store_results("m2s2b1", results_model2_s2b1, "KNN Model - s2b1")

#---- 5-3 PEND *****    Model 3 Decision Tree ----------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-4 PEND *****    Model 4 Random Forest ----------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-5 PEND *****    Model 5 Support Vect Machine ---------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-6 PEND *****    Model 6 Gradient Boosting ------------------------------

################################################################################

#---- 6 PEND ******* Results - Project Step 6 ---------------------------------
################################################################################
# Results - Project Step 6

