#---- 0 DONE ******* Project Step 0 --------------------------------------------
## TODO:
#  - Complete a graph for Fisher Scores for Logical - 4-2-1-1
#  - Complete a grpah for Factor wihout processing Missing.
#  - Look into tuning logistic regression to get the best TP0 and TP1
#  - Look into filtering down the select to only include Details

# Project Goal (Lecture 1): Generate a model to predict the likelihood of a
# person having difficulty living independently.

# Dataset (Lecture 1): Part of the 2023 American Community Survey modified by
# the class professor for this project.

# Target Variable: The Class variable represents difficulty living independently
# and is binary (Yes(1)/No(0)).

# Load necessary libraries
library(tidymodels)
tidymodels_prefer()
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(gridExtra) # nolint
library(caret)
library(rsample)
library(ROSE) # nolint
library(pROC) # nolint
library(ranger)
library(xgboost)

# Set up parallel processing - do this once at the start of your script
library(future)
library(future.apply) # nolint
library(doFuture) # nolint

# Register doFuture as the parallel backend
registerDoFuture()


#---- 0.1 DONE *****    Functions - Full Performance Evaluation ----------------

#results_m2_s1b1 <- calculate_all_measures(fit_m2_s1b1, df_test, best_threshold) # nolint
#results_m1_s1b1 <- calculate_all_measures(final_fit_m1_s1b1, df_test, 0.5) # nolint

calculate_all_measures <- function(in_model, in_test_df, threshold) {
  #in_test_df <- df_test # nolint
  #in_model <- fit_m2_s1b1 # nolint
  #threshold <- best_threshold # nolint

  #in_test_df <- df_test # nolint
  #in_model <- final_fit_m1_s1b1 # nolint
  #threshold <- 0.5 # nolint

  # Predict on the test dataset
  test_predictions <- predict(in_model, new_data = in_test_df, type = "prob")

  # Convert probabilities to binary predictions
  test_predicted_class <- ifelse(test_predictions > threshold, 1, 0) [, 2]

  # Confusion matrix
  confusion_matrix <- table(Predicted = test_predicted_class,
                            Actual = in_test_df$Class)
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

  weight0 <- sum(in_test_df$Class == 0) / nrow(in_test_df)
  weight1 <- sum(in_test_df$Class == 1) / nrow(in_test_df)

  performance_measures <- calculate_measures(tp_0, fp_0, tn_0, fn_0,
                                             tp_1, fp_1, tn_1, fn_1,
                                             weight0, weight1)
  performance_measures

  roc_curve <- roc(in_test_df$Class, test_predictions$.pred_0)
  auc_value <- auc(roc_curve)

  # Convert AUC value to double
  auc_value_double <- as.double(auc_value)

  # Add ROC values to the performance measures dataframe
  performance_measures <- performance_measures %>% # nolint
    add_row(measures = "ROC_0", values = auc_value_double) %>% # nolint
    add_row(measures = "ROC_1", values = auc_value_double) %>%
    add_row(measures = "ROC_W", values = auc_value_double)

  # Print the AUC value
  #print(paste("Area Under the ROC Curve (AUC):", auc_value)) # nolint

  # Plot the ROC curve
  #plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2) # nolint
  #abline(a = 0, b = 1, lty = 2, col = "red") # nolint

  return(performance_measures)

}

#---- 0.2 DONE *****    Functions - Calc Performance Measures ------------------

calculate_measures <- function(tp_0, fp_0, tn_0, fn_0,
                               tp_1, fp_1, tn_1, fn_1,
                               weight0, weight1) {
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
  #print(paste("Kappa for 0, total_0:", total_0)) # nolint

  # Kappa - proportion of agreement to total records
  p_o_0 <- (tp_0 + tn_0) / total_0
  p_e1_0 <- ((tp_0 + fn_0) / total_0) * ((tp_0 + fp_0) / total_0)
  #print(paste("Kappa for 0, p_e1_0:", p_e1_0)) # nolint
  p_e2_0 <- ((fp_0 + tn_0) / total_0) * ((fn_0 + tn_0) / total_0)
  #print(paste("Kappa for 0, p_e2_0:", p_e2_0)) # nolint
  p_e_0 <- p_e1_0 + p_e2_0
  #print(paste("Kappa for 0, p_e_0:", p_e_0)) # nolint
  k_0 <- (p_o_0 - p_e_0) / (1 - p_e_0) # Kappa statistic
  #print(paste("Kappa for 0, k_0:", k_0)) # nolint

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
  #print(paste("Kappa for 1, total_1:", total_1)) # nolint
  p_o_1 <- (tp_1 + tn_1) / total_1
  #print(paste("Kappa for 1, p_o_1:", p_o_1)) # nolint
  p_e1_1 <- ((tp_1 + fn_1) / total_1) * ((tp_1 + fp_1) / total_1)
  #print(paste("Kappa for 1, p_e1_1:", p_e1_1)) # nolint
  p_e2_1 <- ((fp_1 + tn_1) / total_1) * ((fn_1 + tn_1) / total_1)
  #print(paste("Kappa for 1, p_e2_1:", p_e2_1)) # nolint
  p_e_1 <- p_e1_1 + p_e2_1
  #print(paste("Kappa for 1, p_e_1:", p_e_1)) # nolint
  k_1 <- (p_o_1 - p_e_1) / (1 - p_e_1) # Kappa statistic
  #print(paste("Kappa for 1, k_1:", k_1)) # nolint

  #print(paste("Weight for Class 0 (weight0):", weight0)) # nolint
  #print(paste("Weight for Class 1 (weight1):", weight1)) # nolint

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

#---- 0.3 DONE *****    Functions - Store Performance Evaluation ---------------
# Function to store model results with timestamp and description
store_results <- function(combination_key, results_df, description) {
  # Get current date and time
  current_datetime <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")

  # Create a wide format dataframe with measures as columns
  results_wide <- results_df %>% # nolint
    pivot_wider(names_from = measures, values_from = values) # nolint

  # Add description and datetime columns
  results_wide <- results_wide %>% # nolint
    mutate(combination_key = combination_key, # nolint
           description = description,
           datetime = current_datetime)

  # Check if results_storage exists in the global environment
  if (!exists("results_storage", envir = .GlobalEnv)) {
    # Create it if it doesn't exist
    results_storage <<- results_wide # nolint
  } else {
    # Append to existing dataframe if it does exist
    results_storage <<- bind_rows(results_storage, results_wide) # nolint
  }

  return(results_storage) # nolint
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

# factor

factor_columns <- df_columns_info %>%
  filter(variable_type == "factor", column_name %in% names(df)) %>%
  pull(column_name)

num_factor_columns_before <- sum(sapply(df, is.factor))
print(paste("Number of factor columns before update:",
            num_factor_columns_before))

df <- df %>%
  mutate(across(all_of(factor_columns), as.factor)) %>%
  mutate(across(matches(paste0("^DETAILED-",
                               paste(factor_columns, collapse = "|"),
                               "_")), as.factor))

num_factor_columns_after <- sum(sapply(df, is.factor))
print(paste("Number of factor columns after update:",
            num_factor_columns_after))

# logical

logical_columns <- df_columns_info %>%
  filter(variable_type == "logical", column_name %in% names(df)) %>%
  pull(column_name)

num_factor_columns_before <- sum(sapply(df, is.factor))
print(paste("Number of factor columns before logical update:",
            num_factor_columns_before))

df <- df %>%
  mutate(across(all_of(logical_columns), as.factor)) %>%
  mutate(across(matches(paste0("^DETAILED-",
                               paste(logical_columns, collapse = "|"),
                               "_")), as.factor))

num_factor_columns_after <- sum(sapply(df, is.factor))
print(paste("Number of factor columns after logical update:",
            num_factor_columns_after))

#factor level column

factor_level_columns <- df_columns_info %>%
  filter(variable_type == "factor_levels", column_name %in% names(df)) %>%
  pull(column_name)

num_factor_columns_before <- sum(sapply(df, is.factor))
print(paste("Number of factor columns before factor levels update:",
            num_factor_columns_before))

df <- df %>%
  mutate(across(all_of(factor_level_columns), as.factor)) %>%
  mutate(across(matches(paste0("^DETAILED-",
                               paste(factor_level_columns, collapse = "|"),
                               "_")), as.factor))

num_factor_columns_after <- sum(sapply(df, is.factor))
print(paste("Number of factor columns after factor_levels update:",
            num_factor_columns_after))

# Integer - update columns to integer based on variable_type in df_columns_info
integer_columns <- df_columns_info %>%
  filter(variable_type == "integer") %>%
  pull(column_name)

num_integer_columns_before <- sum(sapply(df, is.integer))
print(paste("Number of integer columns before integer update:",
            num_integer_columns_before))

df <- df %>%
  mutate(across(all_of(integer_columns), as.integer)) %>%
  mutate(across(matches(paste0("^DETAILED-",
                               paste(integer_columns, collapse = "|"),
                               "_")), as.integer))

num_integer_columns_after <- sum(sapply(df, is.integer))
print(paste("Number of integer columns after integer update:",
            num_integer_columns_after))

df <- df %>%
  mutate(across(names(df)[columns_with_invalid], ~ na_if(.x, "Invalid Number")))

df_preprocessed <- df

# Export df_preprocessed as a CSV file
write.csv(df_preprocessed, file = "df_preprocessed.csv", row.names = FALSE)

# Export df_preprocessed as an R data file
save(df_preprocessed, file = "df_preprocessed.RData")
save(df_columns_info, file = "df_columns_info.RData")

################################################################################
#---- 2 DONE ******* Split - Project Step 2 ------------- df_train, df_test ----
################################################################################

# Optional - Load the preprocessed dataset
# load("df_preprocessed.RData") # nolint

set.seed(123)

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
#---- 3 DONE ******* Balance - Project Step 3 -------------- df_balanced# ------
################################################################################

# Not using SMOTE because we have a large number of categorical variables.

# Load the training and testing datasets
# load("df_train.RData") # nolint
# load("df_test.RData") # nolint

# Verify the dimensions of the loaded datasets
# print(paste("Loaded training dataset - dim:", dim(df_train)[1], ",", dim(df_train)[2])) # nolint
# print(paste("Loaded testing dataset - dim:", dim(df_test)[1], ",", dim(df_test)[2])) # nolint

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 3.1 DONE *****    Balance - Method 1 - Down Sample ----- df_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_train.RData")

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

#load("df_train.RData")

# Upsampling
df_balanced2 <- upSample(x = df_train[, -which(names(df_train) %in% "Class")],
                         y = df_train$Class)

print(paste("training balanced 2 dataset - dim:", dim(df_balanced2)[1],
            ",", dim(df_balanced2)[2]))

print(paste("training balanced 2 dataset - class distribution:",
            table(df_balanced2$Class)[1], ",",
            table(df_balanced2$Class)[2]))

save(df_balanced2, file = "df_balanced2.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 3.3 DONE *****    Balance - Method 3 - No Balance ------ df_balanced3 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Upsampling
df_balanced3 <- df_train

print(paste("training balanced 3 dataset - dim:", dim(df_balanced3)[1],
            ",", dim(df_balanced3)[2]))

save(df_balanced3, file = "df_balanced3.RData")

################################################################################
#---- 4 PROG ******* Select - Project Step 4 ------------------ df_s#b# --------
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
#load("df_balanced1.RData") # nolint
#load("df_balanced2.RData") # nolint

# Verify the dimensions of the loaded datasets
#print(paste("Loaded df_balanced1 - dim:",
#            dim(df_balanced1)[1], ",", dim(df_balanced1)[2]))
#print(paste("Loaded df_balanced2 - dim:",
#            dim(df_balanced2)[1], ",", dim(df_balanced2)[2]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 4-1 DONE *****    Select 1 - Missing Removal -------------- df_s1b# ------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-1-1 DONE ***       Select 1 - balanced 1 ------------------ df_s1b1 ----
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

df_s1b1 <- df_processing_filt_rows %>%
  select(Class, starts_with("DETAILED-"))

save(df_s1b1, file = "df_s1b1.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 4-1-2 DONE ***       Select 1 - balanced 2 ------------------ df_s1b2 ----
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

df_s1b2 <- df_processing_filt_rows %>%
  select(Class, starts_with("DETAILED-"))

save(df_s1b2, file = "df_s1b2.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 4-2 DONE *****    Select 2 - Fisher and Corr -------------- df_s2b# ------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-2-1 DONE ***       Select 2 - balanced 1 ------------------ df_s2b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_s2b1 <- df_balanced1

#---- 4-2-1-1-DONE *          Factor and Logical Variables ---------------------

df_s2b1_1factors <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("factor")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b1_2logical <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("logical")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b1_3levels <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("factor_levels")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b1_allfact <- cbind(df_s2b1_1factors,
                         df_s2b1_2logical %>% select(-Class),
                         df_s2b1_3levels %>% select(-Class))

##### Will use Fisher test over Chi-square to handle sparse data.

# Some columns have too many levels to be used in Fisher test.
# SCHL - LDSTP too small - 2e9
# ANC1P - LDSTP too small - 1e9

fisher_not_possible <- c("SCHL", "ANC1P", "DETAILED-SCHL_",
                         "DETAILED-ANC1P_",
                         "RACNH", "DETAILED-RACNH_", "Class")

s2b1_fisher_results <- list()

for (col in names(df_s2b1_allfact)) {
  print(paste(Sys.time(), "- Processing column:", col))
  if (any(startsWith(col, fisher_not_possible))) {
    print(paste("Skipping column:", col))
    next
  }
  tryCatch({
    table_data <- table(df_s2b1_allfact[[col]],
                        df_s2b1_allfact$Class)
    fisher_test <- fisher.test(table_data, workspace = 1e9)
    s2b1_fisher_results[[col]] <-
      list(column = col, p_value = fisher_test$p.value)
    print(paste(Sys.time(),
                "- Fisher test column:", col, "p-value:", fisher_test$p.value))
  }, error = function(e) {
    message(paste("Error processing column:", col, "-", e$message))
    s2b1_fisher_results[[col]] <- list(column = col, p_value = NA)
  })
}

# Convert results to a data frame for easier interpretation
df_s2b1_fisher_results <-
  do.call(rbind, lapply(s2b1_fisher_results, as.data.frame))

names(df_s2b1_fisher_results) <- c("Column", "P_value")

df_s2b1_fisher_results$neg_log10_P_value <-
  -log10(df_s2b1_fisher_results$P_value)

# Create a bar plot for Fisher scores
df_s2b1_fisher_plt <- df_s2b1_fisher_results %>%
  mutate(P_value = as.numeric(as.character(P_value))) %>%
  arrange(P_value)

plt_s2b1_fisher <-
  ggplot(df_s2b1_fisher_plt,
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

plt_s2b1_fisher

ggsave("plt_s2b1_fisher.png", plot = plt_s2b1_fisher,
       width = 10, height = 12, dpi = 300)

# Identify the columns with P-values less than 0.05
select_cols <- df_s2b1_fisher_results %>%
  filter(P_value < 0.05) %>%
  arrange(desc(neg_log10_P_value)) %>%
  pull(Column)

# Create a new dataframe with those columns
df_s2b1_allfact <- df_s2b1_allfact %>%
  select(Class, all_of(select_cols))

#---- 4-2-1-2 DONE *          Integer Variables --------------------------------

df_s2b1_4integers <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

in_select1_cor_threshold <- 0.05

repeat {
  df_numeric <- df_s2b1_4integers %>%
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
  df_s2b1_4integers <- df_numeric %>%
    select(-all_of(highly_correlated))

  print(paste("Removed variable:", highly_correlated))
}

#---- 4-2-1-3 DONE *          Outliers -----------------------------------------
# Create boxplots for each numeric variable in the dataset

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(df_s2b1_4integers, function(col) {
  ggplot(df_s2b1_4integers, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

plt_list <- lapply(names(df_s2b1_4integers)
                   [names(df_s2b1_4integers) != "Class"],
                   function(col_name) {
                     ggplot(df_s2b1_4integers,
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

plt_s2b1_corr <- grid.arrange(grobs = plt_list, ncol = 7)

ggsave("plt_s2b1_corr.png",
       plot = plt_s2b1_corr, width = 10, height = 16, dpi = 300)

# Removing class to avoid duplication.  Class is included in both dfs.
df_s2b1 <- df_s2b1_allfact %>% select(-Class) %>% # nolint
  bind_cols(df_s2b1_4integers)

#---- 4-2-1-4 DONE *          Final --------------------------------------------

save(df_s2b1, file = "df_s2b1.RData")

#---- 4-2-2 DONE ***       Select 2 - balanced 2 ------------------ df_s2b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_balanced2.RData") # nolint
#load("df_columns_info.RData") # nolint

df_s2b2 <- df_balanced2

#---- 4-2-2-1 DONE *          Factor and Logical Variables ---------------------

df_s2b2_1factors <- df_balanced2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("factor")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b2_2logical <- df_balanced2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("logical")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b2_3levels <- df_balanced2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("factor_levels")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b2_allfact <- cbind(
  df_s2b2_1factors,
  df_s2b2_2logical %>% select(-Class),
  df_s2b2_3levels %>% select(-Class)
)

##### Will use Fisher test over Chi-square to handle sparse data.

fisher_not_possible <- c("Class", "RACNH", "DETAILED-RACNH_")

s2b2_fisher_results <- list()

# Simulation-based Fisher test
for (col in names(df_s2b2_allfact)) {
  print(paste(Sys.time(), "- Processing column:", col))
  if (any(startsWith(col, fisher_not_possible))) {
    print(paste("Skipping column:", col))
    next
  }
  tryCatch({
    table_data <- table(df_s2b2_allfact[[col]],
                        df_s2b2_allfact$Class)
    fisher_test <- fisher.test(table_data, workspace = 1e9,
                               simulate.p.value = TRUE, B = 2000000)
    s2b2_fisher_results[[col]] <-
      list(column = col, p_value = fisher_test$p.value)
    print(paste(Sys.time(),
                "- Fisher test column:", col, "p-value:", fisher_test$p.value))
  }, error = function(e) {
    message(paste("Error processing column:", col, "-", e$message))
    s2b2_fisher_results[[col]] <- list(column = col, p_value = NA)
  })
}

# Convert results to a data frame for easier interpretation
df_s2b2_fisher_results <-
  do.call(rbind, lapply(s2b2_fisher_results, as.data.frame))

names(df_s2b2_fisher_results) <- c("Column", "P_value")

df_s2b2_fisher_results$neg_log10_P_value <-
  -log10(df_s2b2_fisher_results$P_value)

# Create a bar plot for Fisher scores
df_s2b2_fisher_plt <- df_s2b2_fisher_results %>%
  mutate(P_value = as.numeric(as.character(P_value))) %>%
  arrange(P_value)

plt_s2b2_fisher <-
  ggplot(df_s2b2_fisher_plt,
         aes(x = reorder(substr(Column, 10, 60), -P_value),
             y = -log10(P_value))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = -log10(0.05), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(title = "Fisher Scores for Categorical and Logical Variables 
       with Missing Included",
       subtitle = "Select 2 - Balanced 2",
       caption = "Red line indicates p-value threshold of 0.01",
       x = "",
       y = "-log10(P-value)") +
  theme_minimal()

plt_s2b2_fisher

ggsave("plt_s2b2_fisher.png", plot = plt_s2b2_fisher,
       width = 10, height = 12, dpi = 300)

# Identify the columns with P-values less than 0.05
select_cols <- df_s2b2_fisher_results %>%
  filter(P_value < 0.05) %>%
  arrange(desc(neg_log10_P_value)) %>%
  pull(Column)

# Create a new dataframe with those columns
df_s2b2_allfact <- df_s2b2_allfact %>%
  select(Class, all_of(select_cols))

#---- 4-2-2-2 DONE *          Integer Variables --------------------------------

df_s2b2_4integers <- df_balanced2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

in_select2_cor_threshold <- 0.8

repeat {
  df_numeric <- df_s2b2_4integers %>%
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
  if (abs(most_correlated_correlation) <= in_select2_cor_threshold) {
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
  df_s2b2_4integers <- df_numeric %>%
    select(-all_of(highly_correlated))

  print(paste("Removed variable:", highly_correlated))
}

#---- 4-2-2-3 DONE *          Outliers -----------------------------------------
# Create boxplots for each numeric variable in the dataset

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(df_s2b2_4integers, function(col) {
  ggplot(df_s2b2_4integers, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

plt_list <- lapply(names(df_s2b2_4integers)
                   [names(df_s2b2_4integers) != "Class"],
                   function(col_name) {
                     ggplot(df_s2b2_4integers,
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

plt_s2b2_corr <- grid.arrange(grobs = plt_list, ncol = 7)

ggsave("plt_s2b2_corr.png",
       plot = plt_s2b2_corr, width = 10, height = 16, dpi = 300)

# Removing class to avoid duplication.  Class is included in both dfs.
df_s2b2 <- df_s2b2_allfact %>% select(-Class) %>% # nolint
  bind_cols(df_s2b2_4integers)

#---- 4-2-2-4 DONE *          Final --------------------------------------------

save(df_s2b2, file = "df_s2b2.RData")

#---- 4-2-3 DONE ***       Select 2 - balanced 3 ------------------ df_s2b3 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_balanced3.RData") # nolint
#load("df_columns_info.RData") # nolint

df_s2b3 <- df_balanced3

#---- 4-2-2-1 DONE *          Factor and Logical Variables ---------------------

df_s2b3_1factors <- df_balanced3 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("factor")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b3_2logical <- df_balanced3 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("logical")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b3_3levels <- df_balanced3 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("factor_levels")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

df_s2b3_allfact <- cbind(
  df_s2b3_1factors,
  df_s2b3_2logical %>% select(-Class),
  df_s2b3_3levels %>% select(-Class)
)

##### Will use Fisher test over Chi-square to handle sparse data.

fisher_not_possible <- c("Class", "RACNH", "DETAILED-RACNH_")

s2b3_fisher_results <- list()

# Simulation-based Fisher test
for (col in names(df_s2b3_allfact)) {
  print(paste(Sys.time(), "- Processing column:", col))
  if (any(startsWith(col, fisher_not_possible))) {
    print(paste("Skipping column:", col))
    next
  }
  tryCatch({
    table_data <- table(df_s2b3_allfact[[col]],
                        df_s2b3_allfact$Class)
    fisher_test <- fisher.test(table_data, workspace = 1e9,
                               simulate.p.value = TRUE, B = 2000000)
    s2b3_fisher_results[[col]] <-
      list(column = col, p_value = fisher_test$p.value)
    print(paste(Sys.time(),
                "- Fisher test column:", col, "p-value:", fisher_test$p.value))
  }, error = function(e) {
    message(paste("Error processing column:", col, "-", e$message))
    s2b3_fisher_results[[col]] <- list(column = col, p_value = NA)
  })
}

# Convert results to a data frame for easier interpretation
df_s2b3_fisher_results <-
  do.call(rbind, lapply(s2b3_fisher_results, as.data.frame))

names(df_s2b3_fisher_results) <- c("Column", "P_value")

df_s2b3_fisher_results$neg_log10_P_value <-
  -log10(df_s2b3_fisher_results$P_value)

# Create a bar plot for Fisher scores
df_s2b3_fisher_plt <- df_s2b3_fisher_results %>%
  mutate(P_value = as.numeric(as.character(P_value))) %>%
  arrange(P_value)

plt_s2b3_fisher <-
  ggplot(df_s2b3_fisher_plt,
         aes(x = reorder(substr(Column, 10, 60), -P_value),
             y = -log10(P_value))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = -log10(0.05), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(title = "Fisher Scores for Categorical and Logical Variables 
       with Missing Included",
       subtitle = "Select 2 - Balanced 2",
       caption = "Red line indicates p-value threshold of 0.01",
       x = "",
       y = "-log10(P-value)") +
  theme_minimal()

plt_s2b3_fisher

ggsave("plt_s2b3_fisher.png", plot = plt_s2b3_fisher,
       width = 10, height = 12, dpi = 300)

neg_log10_P_cutoff <- 100

# Identify the columns to keep
select_cols <- df_s2b3_fisher_results %>%
  filter(-log10(P_value) > neg_log10_P_cutoff) %>%
  arrange(desc(neg_log10_P_value)) %>%
  pull(Column)

# Create a new dataframe with those columns
df_s2b3_allfact <- df_s2b3_allfact %>%
  select(Class, all_of(select_cols))

#---- 4-2-2-2 DONE *          Integer Variables --------------------------------

df_s2b3_4integers <- df_balanced3 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

in_select2_cor_threshold <- 0.5

repeat {
  df_numeric <- df_s2b3_4integers %>%
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
  if (abs(most_correlated_correlation) <= in_select2_cor_threshold) {
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

  df_s2b3_4integers <- df_numeric %>%
    select(-all_of(highly_correlated))

  print(paste("Removed variable:", highly_correlated))
}

#---- 4-2-2-3 DONE *          Outliers -----------------------------------------
# Create boxplots for each numeric variable in the dataset

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(df_s2b3_4integers, function(col) {
  ggplot(df_s2b3_4integers, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

plt_list <- lapply(names(df_s2b3_4integers)
                   [names(df_s2b3_4integers) != "Class"],
                   function(col_name) {
                     ggplot(df_s2b3_4integers,
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

plt_s2b3_corr <- grid.arrange(grobs = plt_list, ncol = 7)

ggsave("plt_s2b3_corr.png",
       plot = plt_s2b3_corr, width = 10, height = 16, dpi = 300)

# Removing class to avoid duplication.  Class is included in both dfs.
df_s2b3 <- df_s2b3_allfact %>% select(-Class) %>% # nolint
  bind_cols(df_s2b3_4integers)

#---- 4-2-2-4 DONE *          Final --------------------------------------------

save(df_s2b3, file = "df_s2b3.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 4-3 DONE *****    Select 3 - Missing Added ---------------- df_s3b# ------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-3-1 DONE ***       Select 3 - balanced 1 ------------------ df_s3b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# load("df_balanced1.RData") # nolint

df_s3b1 <- df_balanced1

#---- 4-3-1-1 DONE *          Factor and Logical Variables ---------------------

df_s3b1_4fct_miss <- df_s2b1_1factors %>%
  mutate(across(where(~ any(is.na(.))),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_s3b1_5log_miss <- df_s2b1_2logical %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_s3b1_6lvl_miss <- df_s2b1_3levels %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_s3b1_allfact_miss <- cbind(df_s3b1_4fct_miss,
                              df_s3b1_5log_miss %>% select(-Class),
                              df_s3b1_6lvl_miss %>% select(-Class))

##### Will use Fisher test over Chi-square to handle sparse data.

# Some columns have too many levels to be used in Fisher test.
# SCHL - LDSTP too small - 2e9
# ANC1P - LDSTP too small - 1e9

fisher_not_possible <- c("SCHL", "ANC1P", "DETAILED-SCHL_",
                         "DETAILED-ANC1P_",
                         "RACNH", "DETAILED-RACNH_", "Class")

s3b1_fisher_results <- list()

for (col in names(df_s3b1_allfact_miss)) {
  print(paste(Sys.time(), "- Processing column:", col))
  if (any(startsWith(col, fisher_not_possible))) {
    print(paste("Skipping column:", col))
    next
  }
  tryCatch({
    table_data <- table(df_s3b1_allfact_miss[[col]],
                        df_s3b1_allfact_miss$Class)
    fisher_test <- fisher.test(table_data, workspace = 1e9)
    s3b1_fisher_results[[col]] <-
      list(column = col, p_value = fisher_test$p.value)
    print(paste(Sys.time(),
                "- Fisher test column:", col, "p-value:", fisher_test$p.value))
  }, error = function(e) {
    message(paste("Error processing column:", col, "-", e$message))
    s3b1_fisher_results[[col]] <- list(column = col, p_value = NA)
  })
}

# Convert results to a data frame for easier interpretation
df_s3b1_fisher_results <-
  do.call(rbind, lapply(s3b1_fisher_results, as.data.frame))

names(df_s3b1_fisher_results) <- c("Column", "P_value")

df_s3b1_fisher_results$neg_log10_P_value <-
  -log10(df_s3b1_fisher_results$P_value)

# Create a bar plot for Fisher scores
df_s3b1_fisher_plt <- df_s3b1_fisher_results %>%
  mutate(P_value = as.numeric(as.character(P_value))) %>%
  arrange(P_value)

plt_s3b1_fisher <-
  ggplot(df_s3b1_fisher_plt,
         aes(x = reorder(substr(Column, 10, 60), -P_value),
             y = -log10(P_value))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = -log10(0.05), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(title = "Fisher Scores for Categorical and Logical Variables 
       with Missing Included",
       subtitle = "Select 3 - Balanced 1",
       caption = "Red line indicates p-value threshold of 0.01",
       x = "",
       y = "-log10(P-value)") +
  theme_minimal()

plt_s3b1_fisher

ggsave("plt_s3b1_fisher.png", plot = plt_s3b1_fisher,
       width = 10, height = 12, dpi = 300)

# Identify the columns with P-values less than 0.05
select_cols <- df_s3b1_fisher_results %>%
  filter(P_value < 0.05) %>%
  arrange(desc(neg_log10_P_value)) %>%
  pull(Column)

# Create a new dataframe with those columns
df_s3b1_allfact_miss <- df_s3b1_allfact_miss %>%
  select(Class, all_of(select_cols))

#---- 4-3-1-2 DONE *          Integer Variables --------------------------------

df_s3b1_4integers <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

in_select3_cor_threshold <- 0.05

repeat {
  df_numeric <- df_s3b1_4integers %>%
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
  if (abs(most_correlated_correlation) <= in_select3_cor_threshold) {
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

  df_s3b1_4integers <- df_numeric %>%
    select(-all_of(highly_correlated))

  print(paste("Removed variable:", highly_correlated))
}

#---- 4-3-1-3 DONE *          Outliers -----------------------------------------
# Create boxplots for each numeric variable in the dataset

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(df_s3b1_4integers, function(col) {
  ggplot(df_s3b1_4integers, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

plt_list <- lapply(names(df_s3b1_4integers)
                   [names(df_s3b1_4integers) != "Class"],
                   function(col_name) {
                     ggplot(df_s3b1_4integers,
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

plt_s3b1_corr <- grid.arrange(grobs = plt_list, ncol = 7)

ggsave("plt_s3b1_corr.png",
       plot = plt_s3b1_corr, width = 10, height = 16, dpi = 300)

# Removing class to avoid duplication.  Class is included in both dfs.
df_s3b1 <- df_s3b1_allfact_miss %>% select(-Class) %>% # nolint
  bind_cols(df_s3b1_4integers)

#---- 4-3-1-4 DONE *          Final --------------------------------------------

save(df_s3b1, file = "df_s3b1.RData")

#---- 4-3-2 DONE ***       Select - 3 balanced dataset 2 ---------- df_s3b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# load("df_balanced2.RData") # nolint

df_s3b2 <- df_balanced2

#---- 4-3-2-1 DONE *          Factor and Logical Variables ---------------------

df_s3b2_4fct_miss <- df_s2b2_1factors %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_s3b2_5log_miss <- df_s2b2_2logical %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_s3b2_6lvl_miss <- df_s2b2_3levels %>%
  mutate(across(everything(),
                ~ replace_na(factor(.x,
                                    levels = c(levels(.x),
                                               "Missing")),
                             "Missing")))

df_s3b2_allfact_miss <- cbind(df_s3b2_4fct_miss,
                              df_s3b2_5log_miss %>% select(-Class),
                              df_s3b2_6lvl_miss %>% select(-Class))

##### Will use Fisher test over Chi-square to handle sparse data.

fisher_not_possible <- c("Class", "RACNH", "DETAILED-RACNH_")

s3b2_fisher_results <- list()

# Simulation-based Fisher test
for (col in names(df_s3b2_allfact_miss)) {
  print(paste(Sys.time(), "- Processing column:", col))
  if (any(startsWith(col, fisher_not_possible))) {
    print(paste("Skipping column:", col))
    next
  }
  tryCatch({
    table_data <- table(df_s3b2_allfact_miss[[col]],
                        df_s3b2_allfact_miss$Class)
    fisher_test <- fisher.test(table_data, workspace = 1e9,
                               simulate.p.value = TRUE, B = 2000000)
    s3b2_fisher_results[[col]] <-
      list(column = col, p_value = fisher_test$p.value)
    print(paste(Sys.time(),
                "- Fisher test column:", col, "p-value:", fisher_test$p.value))
  }, error = function(e) {
    message(paste("Error processing column:", col, "-", e$message))
    s3b2_fisher_results[[col]] <- list(column = col, p_value = NA)
  })
}

# Convert results to a data frame for easier interpretation
df_s3b2_fisher_results <-
  do.call(rbind, lapply(s3b2_fisher_results, as.data.frame))

names(df_s3b2_fisher_results) <- c("Column", "P_value")

df_s3b2_fisher_results$neg_log10_P_value <-
  -log10(df_s3b2_fisher_results$P_value)

# Create a bar plot for Fisher scores
df_s3b2_fisher_plt <- df_s3b2_fisher_results %>%
  mutate(P_value = as.numeric(as.character(P_value))) %>%
  arrange(P_value)

plt_s3b2_fisher <-
  ggplot(df_s3b2_fisher_plt,
         aes(x = reorder(substr(Column, 10, 60), -P_value),
             y = -log10(P_value))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = -log10(0.05), color = "red", linetype = "dashed") +
  coord_flip() +
  labs(title = "Fisher Scores for Categorical and Logical Variables 
       with Missing Included",
       subtitle = "Select 3 - Balanced 2",
       caption = "Red line indicates p-value threshold of 0.01",
       x = "",
       y = "-log10(P-value)") +
  theme_minimal()

plt_s3b2_fisher

ggsave("plt_s3b2_fisher.png", plot = plt_s3b2_fisher,
       width = 10, height = 12, dpi = 300)

# Identify the columns with P-values less than 0.05
select_cols <- df_s3b2_fisher_results %>%
  filter(P_value < 0.05) %>%
  arrange(desc(neg_log10_P_value)) %>%
  pull(Column)

# Create a new dataframe with those columns
df_s3b2_allfact_miss <- df_s3b2_allfact_miss %>%
  select(Class, all_of(select_cols))

#---- 4-3-2-2 DONE *          Integer Variables --------------------------------

df_s3b2_4integers <- df_balanced2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

in_select3_cor_threshold <- 0.05

repeat {
  df_numeric <- df_s3b2_4integers %>%
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
  if (abs(most_correlated_correlation) <= in_select3_cor_threshold) {
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

  df_s3b2_4integers <- df_numeric %>%
    select(-all_of(highly_correlated))

  print(paste("Removed variable:", highly_correlated))
}

#---- 4-3-2-3 DONE *          Outliers -----------------------------------------

# Create boxplots for each numeric variable in the dataset

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(df_s3b2_4integers, function(col) {
  ggplot(df_s3b2_4integers, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

plt_list <- lapply(names(df_s3b2_4integers)
                   [names(df_s3b2_4integers) != "Class"],
                   function(col_name) {
                     ggplot(df_s3b2_4integers,
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

plt_s3b2_corr <- grid.arrange(grobs = plt_list, ncol = 7)

ggsave("plt_s3b2_corr.png",
       plot = plt_s3b2_corr, width = 10, height = 16, dpi = 300)

# Removing class to avoid duplication.  Class is included in both dfs.
df_s3b2 <- df_s3b2_allfact_miss %>% select(-Class) %>% # nolint
  bind_cols(df_s3b2_4integers)

#---- 4-3-2-4 DONE *          Final --------------------------------------------

save(df_s3b2, file = "df_s3b2.RData")

#---- 4-4 DONE *****    Select 4 - All Included ------------------- df_s4b3 ----

df_s4b3 <- df_balanced3

#---- 4-4-4-4 DONE *          Final --------------------------------------------

save(df_s4b3, file = "df_s4b3.RData")

################################################################################
#---- 5 PROG ******* Models - Project Step 5 ------------------ m#_s#b# --------
################################################################################

#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

## Models - Project Step 5
# Lecture 1 - Classification Slide:
# * Decision trees,
# * Naive Bayesian classification,
# * Support vector machines,
# * K-nearest neighbors,
# * Neural networks.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-1 DONE *****    Model 1 Logistic Regression ----------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Logistic regression is a statistical method for predicting binary classes.
#---- 5-1-1 DONE ***       Model 1 Logistic Regression ------------ m1_s1b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Logistic Regression Model

df_m1_s1b1 <- df_s1b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m1_s1b1 <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Recipe
rec_m1_s1b1 <- recipe(Class ~ ., data = df_m1_s1b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m1_s1b1 <- workflow() %>%
  add_model(spec_m1_s1b1) %>%
  add_recipe(rec_m1_s1b1)

# 4. Cross-validation
set.seed(123)
folds_m1_s1b1 <- vfold_cv(df_m1_s1b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m1_s1b1 <- grid_regular(penalty(), mixture(), levels = 5)

# 6. Tune the model
tune_results_m1_s1b1 <- tune_grid(
  wf_m1_s1b1,
  resamples = folds_m1_s1b1,
  grid = tune_grid_m1_s1b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m1_s1b1) +
  labs(title = "Tuning Results for Logistic Regression",
       x = "Penalty",
       y = "Mixture") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m1_s1b1 <- select_best(tune_results_m1_s1b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m1_s1b1 <- finalize_workflow(wf_m1_s1b1, best_parameters_m1_s1b1)

# 9. Fit the final model
final_fit_m1_s1b1 <- fit(final_wf_m1_s1b1, data = df_m1_s1b1)

# 10. Evaluate the model on the test dataset
# Evaluate the model on the test dataset
test_predications_m1_s1b1 <-
  predict(final_fit_m1_s1b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(final_fit_m1_s1b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m1_s1b1 <- test_predications_m1_s1b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m1_s1b1)

# Visualize the confusion matrix
autoplot(confusion_matrix_m1_s1b1, type = "heatmap") +
  labs(title = "Confusion Matrix for Logistic Regression",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m1_s1b1 <- calculate_all_measures(final_fit_m1_s1b1, df_test, 0.5)

results_m1_s1b1

store_results("m1s1b1", results_m1_s1b1, "Logistic Regression Model 1 - s1b1")

save(results_storage, file = "results_after_m1_s1b1.RData")

#---- 5-1-2 DONE ***       Model 1 Logistic Regression ------------ m1_s1b2 ----

#load("df_s1b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Logistic Regression Model

df_m1_s1b2 <- df_s1b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m1_s1b2 <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Recipe
rec_m1_s1b2 <- recipe(Class ~ ., data = df_m1_s1b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m1_s1b2 <- workflow() %>%
  add_model(spec_m1_s1b2) %>%
  add_recipe(rec_m1_s1b2)

# 4. Cross-validation
set.seed(123)
folds_m1_s1b2 <- vfold_cv(df_m1_s1b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m1_s1b2 <- grid_regular(penalty(), mixture(), levels = 5)

# 6. Tune the model
tune_results_m1_s1b2 <- tune_grid(
  wf_m1_s1b2,
  resamples = folds_m1_s1b2,
  grid = tune_grid_m1_s1b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m1_s1b2) +
  labs(title = "Tuning Results for Logistic Regression",
       x = "Penalty",
       y = "Mixture") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m1_s1b2 <- select_best(tune_results_m1_s1b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m1_s1b2 <- finalize_workflow(wf_m1_s1b2, best_parameters_m1_s1b2)

# 9. Fit the final model
final_fit_m1_s1b2 <- fit(final_wf_m1_s1b2, data = df_m1_s1b2)

# 10. Evaluate the model on the test dataset
# Evaluate the model on the test dataset
test_predications_m1_s1b2 <-
  predict(final_fit_m1_s1b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(final_fit_m1_s1b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m1_s1b2 <- test_predications_m1_s1b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m1_s1b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m1_s1b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Logistic Regression",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m1_s1b2 <- calculate_all_measures(final_fit_m1_s1b2, df_test, 0.5)

results_m1_s1b2

store_results("m1s1b2", results_m1_s1b2, "Logistic Regression Model 1 - s1b2")

save(results_storage, file = "results_after_m2_s1b1.RData")

#---- 5-1-3 DONE ***       Model 1 Logistic Regression ------------ m1_s2b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Logistic Regression Model

df_m1_s2b1 <- df_s2b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m1_s2b1 <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Recipe
rec_m1_s2b1 <- recipe(Class ~ ., data = df_m1_s2b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m1_s2b1 <- workflow() %>%
  add_model(spec_m1_s2b1) %>%
  add_recipe(rec_m1_s2b1)

# 4. Cross-validation
set.seed(123)
folds_m1_s2b1 <- vfold_cv(df_m1_s2b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m1_s2b1 <- grid_regular(penalty(), mixture(), levels = 5)

# 6. Tune the model
tune_results_m1_s2b1 <- tune_grid(
  wf_m1_s2b1,
  resamples = folds_m1_s2b1,
  grid = tune_grid_m1_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m1_s2b1) +
  labs(title = "Tuning Results for Logistic Regression",
       x = "Penalty",
       y = "Mixture") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m1_s2b1 <- select_best(tune_results_m1_s2b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m1_s2b1 <- finalize_workflow(wf_m1_s2b1, best_parameters_m1_s2b1)

# 9. Fit the final model
final_fit_m1_s2b1 <- fit(final_wf_m1_s2b1, data = df_m1_s2b1)

# 10. Evaluate the model on the test dataset
# Evaluate the model on the test dataset
test_predications_m1_s2b1 <-
  predict(final_fit_m1_s2b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(final_fit_m1_s2b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m1_s2b1 <- test_predications_m1_s2b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m1_s2b1)

results_m1_s2b1 <- calculate_all_measures(final_fit_m1_s2b1, df_test, 0.5)

results_m1_s2b1

store_results("m1s2b1", results_m1_s2b1, "Logistic Regression Model 1 - s2b1")

save(results_storage, file = "results_after_m1_s2b1.RData")

#---- 5-1-4 DONE ***       Model 1 Logistic Regression ------------ m1_s2b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Logistic Regression Model

# Starting with itegers only.
df_m1_s2b2 <- df_s2b2 %>%
  select(-where(~is.factor(.) && length(levels(.)) > 20))
#%>%
#  select(Class, matches(paste0("^DETAILED-(",
#                               paste(df_columns_info %>%
#                                       filter(variable_type %in%
#                                                c("integer")) %>%
#                                       pull(column_name),
#                                     collapse = "|"), ")_")))

df_m1_s2b2 %>% str()

# 1. Model Specification
spec_m1_s2b2 <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Recipe
rec_m1_s2b2 <- recipe(Class ~ ., data = df_m1_s2b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m1_s2b2 <- workflow() %>%
  add_model(spec_m1_s2b2) %>%
  add_recipe(rec_m1_s2b2)

# 4. Cross-validation
set.seed(123)
folds_m1_s2b2 <- vfold_cv(df_m1_s2b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
#tune_grid_m1_s2b2 <- grid_regular(
#  penalty(range = c(-10, -2)),
#  mixture(range = c(0.01, 0.05)),
#  levels = 5
#)
tune_grid_m1_s2b2 <- grid_regular(
  penalty(),
  mixture(),
  levels = 5
)

#tune_grid_m1_s2b2 |> view()

# 6. Tune the model
tune_results_m1_s2b2 <- tune_grid(
  wf_m1_s2b2,
  resamples = folds_m1_s2b2,
  grid = tune_grid_m1_s2b2,
  metrics = metric_set(roc_auc, bal_accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m1_s2b2) +
  labs(title = "Tuning Results for Logistic Regression",
       x = "Penalty",
       y = "Mixture") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m1_s2b2 <- select_best(tune_results_m1_s2b2,
            metric = "roc_auc")

best_parameters_m1_s2b2

# Try different thresholds to achieve target TPR and TNR
thresholds <- seq(0.2, 0.8, by = 0.01)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(final_fit_m1_s2b2, df_m1_s2b2, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    # Heavily penalize being below the targets
    diff_from_target = ifelse(tpr_1 < 0.81, 0.81 - tpr_1, 0) +
      ifelse(tpr_0 < 0.79, 0.79 - tpr_0, 0)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <- threshold_df[which.min(threshold_df$diff_from_target),
 "threshold"]

best_threshold

# Print results for the best threshold
best_row <- threshold_df[threshold_df$threshold == best_threshold, ]
cat("Best threshold:", best_threshold, 
    "\nTPR_1 (Sensitivity):", best_row$TPR_1, 
    "\nTPR_0 (Specificity):", best_row$TPR_0)

# 8. Finalize the workflow
final_wf_m1_s2b2 <- finalize_workflow(wf_m1_s2b2, best_parameters_m1_s2b2)

# 9. Fit the final model
final_fit_m1_s2b2 <- fit(final_wf_m1_s2b2, data = df_m1_s2b2)

# 10. Evaluate the model on the test dataset
# Evaluate the model on the test dataset
# Get probability predictions
test_predictions_prob_m1_s2b2 <- 
  predict(final_fit_m1_s2b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(final_fit_m1_s2b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m1_s2b2 <- test_predications_m1_s2b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m1_s2b2)

results_m1_s2b2 <- calculate_all_measures(final_fit_m1_s2b2, df_test, threshold = .5)
                                           #best_threshold)

results_m1_s2b2

store_results("m1s2b2", results_m1_s2b2, "Logistic Regression Model 1 - s2b2")

save(results_storage, file = "results_after_m1_s2b2.RData")

#---- 5-1-5 DONE ***       Model 1 Logistic Regression ------------ m1_s3b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Logistic Regression Model

df_m1_s3b1 <- df_s3b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m1_s3b1 <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Recipe
rec_m1_s3b1 <- recipe(Class ~ ., data = df_m1_s3b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m1_s3b1 <- workflow() %>%
  add_model(spec_m1_s3b1) %>%
  add_recipe(rec_m1_s3b1)

# 4. Cross-validation
set.seed(123)
folds_m1_s3b1 <- vfold_cv(df_m1_s3b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m1_s3b1 <- grid_regular(penalty(), mixture(), levels = 5)

# 6. Tune the model
tune_results_m1_s3b1 <- tune_grid(
  wf_m1_s3b1,
  resamples = folds_m1_s3b1,
  grid = tune_grid_m1_s3b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m1_s3b1) +
  labs(title = "Tuning Results for Logistic Regression",
       x = "Penalty",
       y = "Mixture") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m1_s3b1 <- select_best(tune_results_m1_s3b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m1_s3b1 <- finalize_workflow(wf_m1_s3b1, best_parameters_m1_s3b1)

# 9. Fit the final model
final_fit_m1_s3b1 <- fit(final_wf_m1_s3b1, data = df_m1_s3b1)

# 10. Evaluate the model on the test dataset
# Evaluate the model on the test dataset
test_predications_m1_s3b1 <-
  predict(final_fit_m1_s3b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(final_fit_m1_s3b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m1_s3b1 <- test_predications_m1_s3b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m1_s3b1)

# Visualize the confusion matrix
autoplot(confusion_matrix_m1_s3b1, type = "heatmap") +
  labs(title = "Confusion Matrix for Logistic Regression",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m1_s3b1 <- calculate_all_measures(final_fit_m1_s3b1, df_test, 0.5)

results_m1_s3b1

store_results("m1s3b1", results_m1_s3b1, "Logistic Regression Model 1 - s3b1")

save(results_storage, file = "results_after_m1_s3b1.RData")

#---- 5-1-6 DONE ***       Model 1 Logistic Regression ------------ m1_s3b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Logistic Regression Model

df_m1_s3b2 <- df_s3b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m1_s3b2 <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 2. Recipe
rec_m1_s3b2 <- recipe(Class ~ ., data = df_m1_s3b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m1_s3b2 <- workflow() %>%
  add_model(spec_m1_s3b2) %>%
  add_recipe(rec_m1_s3b2)

# 4. Cross-validation
set.seed(123)
folds_m1_s3b2 <- vfold_cv(df_m1_s3b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m1_s3b2 <- grid_regular(penalty(), mixture(), levels = 5)

# 6. Tune the model
tune_results_m1_s3b2 <- tune_grid(
  wf_m1_s3b2,
  resamples = folds_m1_s3b2,
  grid = tune_grid_m1_s3b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m1_s3b2) +
  labs(title = "Tuning Results for Logistic Regression",
       x = "Penalty",
       y = "Mixture") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m1_s3b2 <- select_best(tune_results_m1_s3b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m1_s3b2 <- finalize_workflow(wf_m1_s3b2, best_parameters_m1_s3b2)

# 9. Fit the final model
final_fit_m1_s3b2 <- fit(final_wf_m1_s3b2, data = df_m1_s3b2)

# 10. Evaluate the model on the test dataset
# Evaluate the model on the test dataset
test_predications_m1_s3b2 <-
  predict(final_fit_m1_s3b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(final_fit_m1_s3b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m1_s3b2 <- test_predications_m1_s3b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m1_s3b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m1_s3b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Logistic Regression",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m1_s3b2 <- calculate_all_measures(final_fit_m1_s3b2, df_test, 0.5)

results_m1_s3b2

store_results("m1s3b2", results_m1_s3b2, "Logistic Regression Model 1 - s3b2")

save(results_storage, file = "results_after_m1_s3b2.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-2 DONE *****    Model 2 K-Nearest Neighbors ----------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 5-2-1 DONE ***       Model 2 KNN - s1b1 --------------------- m2-s1b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m2_s1b1 <- df_s1b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m2_s1b1 <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# 2. Recipe
rec_m2_s1b1 <- recipe(Class ~ ., data = df_m2_s1b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Resolve conflict between kknn::contr.dummy and caret::contr.dummy
conflicted::conflicts_prefer(kknn::contr.dummy)

# 3. Workflow
wf_m2_s1b1 <- workflow() %>%
  add_model(spec_m2_s1b1) %>%
  add_recipe(rec_m2_s1b1)

# 4. Cross-validation
set.seed(123)
folds_m2_s1b1 <- vfold_cv(df_m2_s1b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m2_s1b1 <- grid_regular(
  neighbors(range = c(5, 50)),
  weight_func(values = c("rectangular", "triangular", "gaussian", "rank")),
  levels = c(10, 4)
)

# 6. Tune the model
tune_results_m2_s1b1 <- tune_grid(
  wf_m2_s1b1,
  resamples = folds_m2_s1b1,
  grid = grid_m2_s1b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m2_s1b1) +
  labs(title = "Tuning Results for K-Nearest Neighbors",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m2_s1b1 <- select_best(tune_results_m2_s1b1, metric = "roc_auc")

print(best_params_m2_s1b1)

# 8. Finalize the workflow
final_wf_m2_s1b1 <- finalize_workflow(wf_m2_s1b1, best_params_m2_s1b1)

# 9. Fit the final model
fit_m2_s1b1 <- fit(final_wf_m2_s1b1, data = df_m2_s1b1)

# Try different thresholds to achieve the target TPR and TNR
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m2_s1b1, df_m2_s1b1, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target = abs(tpr_1 - 0.81) + abs(tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.min(threshold_df$diff_from_target), "threshold"]

# 10. Evaluate the model on the test dataset
results_m2_s1b1 <- calculate_all_measures(fit_m2_s1b1, df_test, best_threshold)
results_m2_s1b1
store_results("m2s1b1", results_m2_s1b1, "KNN Model - s1b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m2_s1b1.RData")

#---- 5-2-2 DONE ***       Model 2 KNN - s1b2 --------------------- m2-s1b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m2_s1b2 <- df_s1b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m2_s1b2 <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# 2. Recipe
rec_m2_s1b2 <- recipe(Class ~ ., data = df_m2_s1b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Resolve conflict between kknn::contr.dummy and caret::contr.dummy
conflicted::conflicts_prefer(kknn::contr.dummy)

# 3. Workflow
wf_m2_s1b2 <- workflow() %>%
  add_model(spec_m2_s1b2) %>%
  add_recipe(rec_m2_s1b2)

# 4. Cross-validation
set.seed(123)
folds_m2_s1b2 <- vfold_cv(df_m2_s1b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m2_s1b2 <- grid_regular(
  neighbors(range = c(5, 50)),
  weight_func(values = c("rectangular", "triangular", "gaussian", "rank")),
  levels = c(10, 4)
)

# 6. Tune the model
tune_results_m2_s1b2 <- tune_grid(
  wf_m2_s1b2,
  resamples = folds_m2_s1b2,
  grid = grid_m2_s1b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m2_s1b2) +
  labs(title = "Tuning Results for K-Nearest Neighbors",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m2_s1b2 <- select_best(tune_results_m2_s1b2, metric = "roc_auc")

print(best_params_m2_s1b2)

# 8. Finalize the workflow
final_wf_m2_s1b2 <- finalize_workflow(wf_m2_s1b2, best_params_m2_s1b2)

# 9. Fit the final model
fit_m2_s1b2 <- fit(final_wf_m2_s1b2, data = df_m2_s1b2)

# Try different thresholds to achieve the target TPR and TNR
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m2_s1b2, df_m2_s1b2, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target = abs(tpr_1 - 0.81) + abs(tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.min(threshold_df$diff_from_target), "threshold"]

# 10. Evaluate the model on the test dataset
results_m2_s1b2 <- calculate_all_measures(fit_m2_s1b2, df_test, best_threshold)
results_m2_s1b2
store_results("m2s1b2", results_m2_s1b2, "KNN Model - s1b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m2_s1b2.RData")

#---- 5-2-3 DONE ***       Model 2 KNN - s2b1 --------------------- m2-s2b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m2_s2b1 <- df_s2b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m2_s2b1 <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# 2. Recipe
rec_m2_s2b1 <- recipe(Class ~ ., data = df_m2_s2b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Resolve conflict between kknn::contr.dummy and caret::contr.dummy
conflicted::conflicts_prefer(kknn::contr.dummy)

# 3. Workflow
wf_m2_s2b1 <- workflow() %>%
  add_model(spec_m2_s2b1) %>%
  add_recipe(rec_m2_s2b1)

# 4. Cross-validation
set.seed(123)
folds_m2_s2b1 <- vfold_cv(df_m2_s2b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m2_s2b1 <- grid_regular(
  neighbors(range = c(5, 50)),
  weight_func(values = c("rectangular", "triangular", "gaussian", "rank")),
  levels = c(10, 4)
)

# 6. Tune the model
tune_results_m2_s2b1 <- tune_grid(
  wf_m2_s2b1,
  resamples = folds_m2_s2b1,
  grid = grid_m2_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m2_s2b1) +
  labs(title = "Tuning Results for K-Nearest Neighbors",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m2_s2b1 <- select_best(tune_results_m2_s2b1, metric = "roc_auc")

print(best_params_m2_s2b1)

# 8. Finalize the workflow
final_wf_m2_s2b1 <- finalize_workflow(wf_m2_s2b1, best_params_m2_s2b1)

# 9. Fit the final model
fit_m2_s2b1 <- fit(final_wf_m2_s2b1, data = df_m2_s2b1)

# Try different thresholds to achieve the target TPR and TNR
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m2_s2b1, df_m2_s2b1, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target = abs(tpr_1 - 0.81) + abs(tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.min(threshold_df$diff_from_target), "threshold"]

# 10. Evaluate the model on the test dataset
results_m2_s2b1 <- calculate_all_measures(fit_m2_s2b1, df_test, best_threshold)
results_m2_s2b1
store_results("m2s2b1", results_m2_s2b1, "KNN Model - s2b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m2_s2b1.RData")

#---- 5-2-4 DONE ***       Model 2 KNN - s2b2 --------------------- m2-s2b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m2_s2b2 <- df_s2b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m2_s2b2 <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# 2. Recipe
rec_m2_s2b2 <- recipe(Class ~ ., data = df_m2_s2b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Resolve conflict between kknn::contr.dummy and caret::contr.dummy
conflicted::conflicts_prefer(kknn::contr.dummy)

# 3. Workflow
wf_m2_s2b2 <- workflow() %>%
  add_model(spec_m2_s2b2) %>%
  add_recipe(rec_m2_s2b2)

# 4. Cross-validation
set.seed(123)
folds_m2_s2b2 <- vfold_cv(df_m2_s2b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m2_s2b2 <- grid_regular(
  neighbors(range = c(5, 50)),
  weight_func(values = c("rectangular", "triangular", "gaussian", "rank")),
  levels = c(10, 4)
)

# 6. Tune the model
tune_results_m2_s2b2 <- tune_grid(
  wf_m2_s2b2,
  resamples = folds_m2_s2b2,
  grid = grid_m2_s2b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m2_s2b2) +
  labs(title = "Tuning Results for K-Nearest Neighbors",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m2_s2b2 <- select_best(tune_results_m2_s2b2, metric = "roc_auc")

print(best_params_m2_s2b2)

# 8. Finalize the workflow
final_wf_m2_s2b2 <- finalize_workflow(wf_m2_s2b2, best_params_m2_s2b2)

# 9. Fit the final model
fit_m2_s2b2 <- fit(final_wf_m2_s2b2, data = df_m2_s2b2)

# Try different thresholds to achieve the target TPR and TNR
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m2_s2b2, df_m2_s2b2, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target = abs(tpr_1 - 0.81) + abs(tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.min(threshold_df$diff_from_target), "threshold"]

# 10. Evaluate the model on the test dataset
results_m2_s2b2 <- calculate_all_measures(fit_m2_s2b2, df_test, best_threshold)
results_m2_s2b2
store_results("m2s2b2", results_m2_s2b2, "KNN Model - s2b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m2_s2b2.RData")

#---- 5-2-5 DONE ***       Model 2 KNN - s3b1 --------------------- m2-s3b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m2_s3b1 <- df_s3b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m2_s3b1 <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# 2. Recipe
rec_m2_s3b1 <- recipe(Class ~ ., data = df_m2_s3b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Resolve conflict between kknn::contr.dummy and caret::contr.dummy
conflicted::conflicts_prefer(kknn::contr.dummy)

# 3. Workflow
wf_m2_s3b1 <- workflow() %>%
  add_model(spec_m2_s3b1) %>%
  add_recipe(rec_m2_s3b1)

# 4. Cross-validation
set.seed(123)
folds_m2_s3b1 <- vfold_cv(df_m2_s3b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m2_s3b1 <- grid_regular(
  neighbors(range = c(5, 50)),
  weight_func(values = c("rectangular", "triangular", "gaussian", "rank")),
  levels = c(10, 4)
)

# 6. Tune the model
tune_results_m2_s3b1 <- tune_grid(
  wf_m2_s3b1,
  resamples = folds_m2_s3b1,
  grid = grid_m2_s3b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m2_s3b1) +
  labs(title = "Tuning Results for K-Nearest Neighbors",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m2_s3b1 <- select_best(tune_results_m2_s3b1, metric = "roc_auc")

print(best_params_m2_s3b1)

# 8. Finalize the workflow
final_wf_m2_s3b1 <- finalize_workflow(wf_m2_s3b1, best_params_m2_s3b1)

# 9. Fit the final model
fit_m2_s3b1 <- fit(final_wf_m2_s3b1, data = df_m2_s3b1)

# Try different thresholds to achieve the target TPR and TNR
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m2_s3b1, df_m2_s3b1, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target = abs(tpr_1 - 0.81) + abs(tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.min(threshold_df$diff_from_target), "threshold"]

# 10. Evaluate the model on the test dataset
results_m2_s3b1 <- calculate_all_measures(fit_m2_s3b1, df_test, best_threshold)
results_m2_s3b1
store_results("m2s3b1", results_m2_s3b1, "KNN Model - s3b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m2_s3b1.RData")

#---- 5-2-6 DONE ***       Model 2 KNN - s3b2 --------------------- m2-s3b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m2_s3b2 <- df_s3b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m2_s3b2 <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# 2. Recipe
rec_m2_s3b2 <- recipe(Class ~ ., data = df_m2_s3b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "unknown") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# Resolve conflict between kknn::contr.dummy and caret::contr.dummy
conflicted::conflicts_prefer(kknn::contr.dummy)

# 3. Workflow
wf_m2_s3b2 <- workflow() %>%
  add_model(spec_m2_s3b2) %>%
  add_recipe(rec_m2_s3b2)

# 4. Cross-validation
set.seed(123)
folds_m2_s3b2 <- vfold_cv(df_m2_s3b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m2_s3b2 <- grid_regular(
  neighbors(range = c(5, 50)),
  weight_func(values = c("rectangular", "triangular", "gaussian", "rank")),
  levels = c(10, 4)
)

# 6. Tune the model
tune_results_m2_s3b2 <- tune_grid(
  wf_m2_s3b2,
  resamples = folds_m2_s3b2,
  grid = grid_m2_s3b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m2_s3b2) +
  labs(title = "Tuning Results for K-Nearest Neighbors",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m2_s3b2 <- select_best(tune_results_m2_s3b2, metric = "roc_auc")

print(best_params_m2_s3b2)

# 8. Finalize the workflow
final_wf_m2_s3b2 <- finalize_workflow(wf_m2_s3b2, best_params_m2_s3b2)

# 9. Fit the final model
fit_m2_s3b2 <- fit(final_wf_m2_s3b2, data = df_m2_s3b2)

# Try different thresholds to achieve the target TPR and TNR
thresholds <- seq(0.3, 0.7, by = 0.05)
threshold_results <- list()

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m2_s3b2, df_m2_s3b2, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target = abs(tpr_1 - 0.81) + abs(tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.min(threshold_df$diff_from_target), "threshold"]

# 10. Evaluate the model on the test dataset
results_m2_s3b2 <- calculate_all_measures(fit_m2_s3b2, df_test, best_threshold)
results_m2_s3b2
store_results("m2s3b2", results_m2_s3b2, "KNN Model - s3b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m2_s3b2.RData")

#---- 5-3 DONE *****    Model 3 Decision Tree ----------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 5-3-1 DONE ***      Model 3 Decision Tree ------------------- m3-s1b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m3_s1b1 <- df_s1b1 %>%
  select(Class,
         matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor",
                                           "logical",
                                           "factor-level")) %>%
                                pull(column_name),
                              collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m3_s1b1 <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 2. Recipe
rec_m3_s1b1 <- recipe(Class ~ ., data = df_m3_s1b1)

# 3. Workflow
wf_m3_s1b1 <- workflow() %>%
  add_model(spec_m3_s1b1) %>%
  add_recipe(rec_m3_s1b1)

# 4. Cross-validation
set.seed(123)
folds_m3_s1b1 <- vfold_cv(df_m3_s1b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m3_s1b1 <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# 6. Tune the model
tune_results_m3_s1b1 <- tune_grid(
  wf_m3_s1b1,
  resamples = folds_m3_s1b1,
  grid = grid_m3_s1b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m3_s1b1) +
  labs(title = "Tuning Results for Decision Tree",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m3_s1b1 <- select_best(tune_results_m3_s1b1, metric = "roc_auc")

print(best_params_m3_s1b1)

# 8. Finalize the workflow
final_wf_m3_s1b1 <- finalize_workflow(wf_m3_s1b1, best_params_m3_s1b1)

# 9. Fit the final model
fit_m3_s1b1 <- fit(final_wf_m3_s1b1, data = df_m3_s1b1)

# 10. Evaluate the model on the test dataset
results_m3_s1b1 <- calculate_all_measures(fit_m3_s1b1, df_test, 0.5)
results_m3_s1b1
store_results("m3s1b1", results_m3_s1b1, "Decision Tree Model - s1b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m3_s1b1.RData")

#---- 5-3-2 DONE ***      Model 3 Decision Tree ------------------- m3-s1b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m3_s1b2 <- df_s1b2 %>%
  select(Class,
         matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor",
                                           "logical",
                                           "factor-level")) %>%
                                pull(column_name),
                              collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m3_s1b2 <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 2. Recipe
rec_m3_s1b2 <- recipe(Class ~ ., data = df_m3_s1b2)

# 3. Workflow
wf_m3_s1b2 <- workflow() %>%
  add_model(spec_m3_s1b2) %>%
  add_recipe(rec_m3_s1b2)

# 4. Cross-validation
set.seed(123)
folds_m3_s1b2 <- vfold_cv(df_m3_s1b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m3_s1b2 <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# 6. Tune the model
tune_results_m3_s1b2 <- tune_grid(
  wf_m3_s1b2,
  resamples = folds_m3_s1b2,
  grid = grid_m3_s1b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m3_s1b2) +
  labs(title = "Tuning Results for Decision Tree",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m3_s1b2 <- select_best(tune_results_m3_s1b2, metric = "roc_auc")

print(best_params_m3_s1b2)

# 8. Finalize the workflow
final_wf_m3_s1b2 <- finalize_workflow(wf_m3_s1b2, best_params_m3_s1b2)

# 9. Fit the final model
fit_m3_s1b2 <- fit(final_wf_m3_s1b2, data = df_m3_s1b2)

# 10. Evaluate the model on the test dataset
results_m3_s1b2 <- calculate_all_measures(fit_m3_s1b2, df_test, 0.5)
results_m3_s1b2
store_results("m3s1b2", results_m3_s1b2, "Decision Tree Model - s1b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m3_s1b2.RData")

#---- 5-3-3 DONE ***      Model 3 Decision Tree ------------------- m3-s2b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m3_s2b1 <- df_s2b1 %>%
  select(Class,
         matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor",
                                           "logical",
                                           "factor-level")) %>%
                                pull(column_name),
                              collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m3_s2b1 <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 2. Recipe
rec_m3_s2b1 <- recipe(Class ~ ., data = df_m3_s2b1)

# 3. Workflow
wf_m3_s2b1 <- workflow() %>%
  add_model(spec_m3_s2b1) %>%
  add_recipe(rec_m3_s2b1)

# 4. Cross-validation
set.seed(123)
folds_m3_s2b1 <- vfold_cv(df_m3_s2b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m3_s2b1 <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# 6. Tune the model
tune_results_m3_s2b1 <- tune_grid(
  wf_m3_s2b1,
  resamples = folds_m3_s2b1,
  grid = grid_m3_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m3_s2b1) +
  labs(title = "Tuning Results for Decision Tree",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m3_s2b1 <- select_best(tune_results_m3_s2b1, metric = "roc_auc")

print(best_params_m3_s2b1)

# 8. Finalize the workflow
final_wf_m3_s2b1 <- finalize_workflow(wf_m3_s2b1, best_params_m3_s2b1)

# 9. Fit the final model
fit_m3_s2b1 <- fit(final_wf_m3_s2b1, data = df_m3_s2b1)

# 10. Evaluate the model on the test dataset
results_m3_s2b1 <- calculate_all_measures(fit_m3_s2b1, df_test, 0.5)
results_m3_s2b1
store_results("m3s2b1", results_m3_s2b1, "Decision Tree Model - s2b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m3_s2b1.RData")

#---- 5-3-4 DONE ***      Model 3 Decision Tree ------------------- m3-s2b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m3_s2b2 <- df_s2b2 %>%
  select(Class,
         matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor",
                                           "logical",
                                           "factor-level")) %>%
                                pull(column_name),
                              collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m3_s2b2 <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 2. Recipe
rec_m3_s2b2 <- recipe(Class ~ ., data = df_m3_s2b2)

# 3. Workflow
wf_m3_s2b2 <- workflow() %>%
  add_model(spec_m3_s2b2) %>%
  add_recipe(rec_m3_s2b2)

# 4. Cross-validation
set.seed(123)
folds_m3_s2b2 <- vfold_cv(df_m3_s2b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m3_s2b2 <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# 6. Tune the model
tune_results_m3_s2b2 <- tune_grid(
  wf_m3_s2b2,
  resamples = folds_m3_s2b2,
  grid = grid_m3_s2b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m3_s2b2) +
  labs(title = "Tuning Results for Decision Tree",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m3_s2b2 <- select_best(tune_results_m3_s2b2, metric = "roc_auc")

print(best_params_m3_s2b2)

# 8. Finalize the workflow
final_wf_m3_s2b2 <- finalize_workflow(wf_m3_s2b2, best_params_m3_s2b2)

# 9. Fit the final model
fit_m3_s2b2 <- fit(final_wf_m3_s2b2, data = df_m3_s2b2)

# 10. Evaluate the model on the test dataset
results_m3_s2b2 <- calculate_all_measures(fit_m3_s2b2, df_test, 0.5)
results_m3_s2b2
store_results("m3s2b2", results_m3_s2b2, "Decision Tree Model - s2b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m3_s2b2.RData")

#---- 5-3-5 DONE ***      Model 3 Decision Tree ------------------- m3-s3b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m3_s3b1 <- df_s3b1 %>%
  select(Class,
         matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor",
                                           "logical",
                                           "factor-level")) %>%
                                pull(column_name),
                              collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m3_s3b1 <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 2. Recipe
rec_m3_s3b1 <- recipe(Class ~ ., data = df_m3_s3b1)

# 3. Workflow
wf_m3_s3b1 <- workflow() %>%
  add_model(spec_m3_s3b1) %>%
  add_recipe(rec_m3_s3b1)

# 4. Cross-validation
set.seed(123)
folds_m3_s3b1 <- vfold_cv(df_m3_s3b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m3_s3b1 <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# 6. Tune the model
tune_results_m3_s3b1 <- tune_grid(
  wf_m3_s3b1,
  resamples = folds_m3_s3b1,
  grid = grid_m3_s3b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m3_s3b1) +
  labs(title = "Tuning Results for Decision Tree",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m3_s3b1 <- select_best(tune_results_m3_s3b1, metric = "roc_auc")

print(best_params_m3_s3b1)

# 8. Finalize the workflow
final_wf_m3_s3b1 <- finalize_workflow(wf_m3_s3b1, best_params_m3_s3b1)

# 9. Fit the final model
fit_m3_s3b1 <- fit(final_wf_m3_s3b1, data = df_m3_s3b1)

# 10. Evaluate the model on the test dataset
results_m3_s3b1 <- calculate_all_measures(fit_m3_s3b1, df_test, 0.5)
results_m3_s3b1
store_results("m3s3b1", results_m3_s3b1, "Decision Tree Model - s3b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m3_s3b1.RData")

#---- 5-3-6 DONE ***      Model 3 Decision Tree ------------------- m3-s3b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use Integers
df_m3_s3b2 <- df_s3b2 %>%
  select(Class,
         matches(paste0("^DETAILED-(",
                        paste(df_columns_info %>%
                                filter(variable_type %in%
                                         c("factor",
                                           "logical",
                                           "factor-level")) %>%
                                pull(column_name),
                              collapse = "|"), ")_"))) %>%
  select(-matches("SERIALNO"))

# 1. Model Specification
spec_m3_s3b2 <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")

# 2. Recipe
rec_m3_s3b2 <- recipe(Class ~ ., data = df_m3_s3b2)

# 3. Workflow
wf_m3_s3b2 <- workflow() %>%
  add_model(spec_m3_s3b2) %>%
  add_recipe(rec_m3_s3b2)

# 4. Cross-validation
set.seed(123)
folds_m3_s3b2 <- vfold_cv(df_m3_s3b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
grid_m3_s3b2 <- grid_regular(
  cost_complexity(),
  tree_depth(),
  min_n(),
  levels = 5
)

# 6. Tune the model
tune_results_m3_s3b2 <- tune_grid(
  wf_m3_s3b2,
  resamples = folds_m3_s3b2,
  grid = grid_m3_s3b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m3_s3b2) +
  labs(title = "Tuning Results for Decision Tree",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m3_s3b2 <- select_best(tune_results_m3_s3b2, metric = "roc_auc")

print(best_params_m3_s3b2)

# 8. Finalize the workflow
final_wf_m3_s3b2 <- finalize_workflow(wf_m3_s3b2, best_params_m3_s3b2)

# 9. Fit the final model
fit_m3_s3b2 <- fit(final_wf_m3_s3b2, data = df_m3_s3b2)

# 10. Evaluate the model on the test dataset
results_m3_s3b2 <- calculate_all_measures(fit_m3_s3b2, df_test, 0.5)
results_m3_s3b2
store_results("m3s3b2", results_m3_s3b2, "Decision Tree Model - s3b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m3_s3b2.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-4 DONE *****    Model 4 Random Forest ----------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-4-1 DONE ***      Model 4 Random Forest ------------------- m4-s1b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use categorical variables for Random Forest
df_m4_s1b1 <- df_s1b1 %>% select(-matches("SERIALNO"))

# 1. Model Specification
spec_m4_s1b1 <- rand_forest(
  mtry = tune(),       # Number of predictors to sample at each split
  trees = tune(),      # Number of trees
  min_n = tune()       # Minimum node size
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# 2. Recipe
rec_m4_s1b1 <- recipe(Class ~ ., data = df_m4_s1b1) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) 

# 3. Workflow
wf_m4_s1b1 <- workflow() %>%
  add_model(spec_m4_s1b1) %>%
  add_recipe(rec_m4_s1b1)

# 4. Cross-validation
set.seed(123)
folds_m4_s1b1 <- vfold_cv(df_m4_s1b1, v = 20, strata = Class)

# 5. Grid of hyperparameters
# For mtry, we'll try different numbers of predictors
num_predictors <- ncol(df_m4_s1b1) - 1
mtry_values <- floor(c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75) * num_predictors)
mtry_values <- unique(mtry_values[mtry_values > 0])

grid_m4_s1b1 <- grid_regular(
  mtry(range = range(mtry_values)),
  trees(range = c(100, 500)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m4_s1b1 <- tune_grid(
  wf_m4_s1b1,
  resamples = folds_m4_s1b1,
  grid = grid_m4_s1b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m4_s1b1) +
  labs(title = "Tuning Results for Random Forest",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m4_s1b1 <- select_best(tune_results_m4_s1b1, metric = "roc_auc")

print(best_params_m4_s1b1)

# 8. Finalize the workflow
final_wf_m4_s1b1 <- finalize_workflow(wf_m4_s1b1, best_params_m4_s1b1)

# 9. Fit the final model
fit_m4_s1b1 <- fit(final_wf_m4_s1b1, data = df_m4_s1b1)

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m4_s1b1, df_m4_s1b1, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target =
      ifelse(tpr_1 - 0.81 > 0,
             (tpr_1 - 0.81) / 100,
             tpr_1 - 0.81) +
      ifelse(tpr_0 - 0.79 > 0,
             (tpr_0 - 0.79) / 100,
             tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.max(threshold_df$diff_from_target), "threshold"]

cat("Best threshold:", best_threshold, "\n")

# 10. Evaluate the model on the test dataset
results_m4_s1b1 <- calculate_all_measures(fit_m4_s1b1, df_test, 0.5)
results_m4_s1b1
store_results("m4s1b1", results_m4_s1b1, "Random Forest Model - s1b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m4_s1b1.RData")

# Optional: Extract variable importance
if (inherits(fit_m4_s1b1$fit$fit$fit, "ranger")) {
  var_imp <- ranger::importance(fit_m4_s1b1$fit$fit$fit)
  var_imp_df <- data.frame(Variable = names(var_imp), Importance = var_imp)
  var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]

  # Plot top 20 variables by importance
  ggplot(head(var_imp_df, 20),
         aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest Variable Importance",
         x = "",
         y = "Importance") +
    theme_minimal()

  ggsave("m4_s1b1_var_importance.png", width = 10, height = 8, dpi = 300)

}
#---- 5-4-2 DONE ***      Model 4 Random Forest ------------------- m4-s1b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use categorical variables for Random Forest
df_m4_s1b2 <- df_s1b2 %>% select(-matches("SERIALNO"))

# 1. Model Specification
spec_m4_s1b2 <- rand_forest(
  mtry = tune(),       # Number of predictors to sample at each split
  trees = tune(),      # Number of trees
  min_n = tune()       # Minimum node size
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# 2. Recipe
rec_m4_s1b2 <- recipe(Class ~ ., data = df_m4_s1b2) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

# 3. Workflow
wf_m4_s1b2 <- workflow() %>%
  add_model(spec_m4_s1b2) %>%
  add_recipe(rec_m4_s1b2)

# 4. Cross-validation
set.seed(123)
folds_m4_s1b2 <- vfold_cv(df_m4_s1b2, v = 20, strata = Class)

# 5. Grid of hyperparameters
# For mtry, we'll try different numbers of predictors
num_predictors <- ncol(df_m4_s1b2) - 1
mtry_values <- floor(c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75) * num_predictors)
mtry_values <- unique(mtry_values[mtry_values > 0])

grid_m4_s1b2 <- grid_regular(
  mtry(range = range(mtry_values)),
  trees(range = c(100, 500)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m4_s1b2 <- tune_grid(
  wf_m4_s1b2,
  resamples = folds_m4_s1b2,
  grid = grid_m4_s1b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m4_s1b2) +
  labs(title = "Tuning Results for Random Forest",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m4_s1b2 <- select_best(tune_results_m4_s1b2, metric = "roc_auc")

print(best_params_m4_s1b2)

# 8. Finalize the workflow
final_wf_m4_s1b2 <- finalize_workflow(wf_m4_s1b2, best_params_m4_s1b2)

# 9. Fit the final model
fit_m4_s1b2 <- fit(final_wf_m4_s1b2, data = df_m4_s1b2)

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m4_s1b2, df_m4_s1b2, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target =
      ifelse(tpr_1 - 0.81 > 0,
             (tpr_1 - 0.81) / 100,
             tpr_1 - 0.81) +
      ifelse(tpr_0 - 0.79 > 0,
             (tpr_0 - 0.79) / 100,
             tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.max(threshold_df$diff_from_target), "threshold"]

cat("Best threshold:", best_threshold, "\n")

# 10. Evaluate the model on the test dataset
results_m4_s1b2 <- calculate_all_measures(fit_m4_s1b2, df_test, 0.5)
results_m4_s1b2
store_results("m4s1b2", results_m4_s1b2, "Random Forest Model - s1b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m4_s1b2.RData")

# Optional: Extract variable importance
if (inherits(fit_m4_s1b2$fit$fit$fit, "ranger")) {
  var_imp <- ranger::importance(fit_m4_s1b2$fit$fit$fit)
  var_imp_df <- data.frame(Variable = names(var_imp), Importance = var_imp)
  var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]

  # Plot top 20 variables by importance
  ggplot(head(var_imp_df, 20),
         aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest Variable Importance",
         x = "",
         y = "Importance") +
    theme_minimal()

  ggsave("m4_s1b2_var_importance.png", width = 10, height = 8, dpi = 300)

}

#---- 5-4-3 DONE ***      Model 4 Random Forest ------------------- m4-s2b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use categorical variables for Random Forest
df_m4_s2b1 <- df_s2b1 %>% select(-matches("SERIALNO"))

# 1. Model Specification
spec_m4_s2b1 <- rand_forest(
  mtry = tune(),       # Number of predictors to sample at each split
  trees = tune(),      # Number of trees
  min_n = tune()       # Minimum node size
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# 2. Recipe
rec_m4_s2b1 <- recipe(Class ~ ., data = df_m4_s2b1) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) 

# 3. Workflow
wf_m4_s2b1 <- workflow() %>%
  add_model(spec_m4_s2b1) %>%
  add_recipe(rec_m4_s2b1)

# 4. Cross-validation
set.seed(123)
folds_m4_s2b1 <- vfold_cv(df_m4_s2b1, v = 20, strata = Class)

# 5. Grid of hyperparameters
# For mtry, we'll try different numbers of predictors
num_predictors <- ncol(df_m4_s2b1) - 1
mtry_values <- floor(c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75) * num_predictors)
mtry_values <- unique(mtry_values[mtry_values > 0])

grid_m4_s2b1 <- grid_regular(
  mtry(range = range(mtry_values)),
  trees(range = c(100, 500)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m4_s2b1 <- tune_grid(
  wf_m4_s2b1,
  resamples = folds_m4_s2b1,
  grid = grid_m4_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m4_s2b1) +
  labs(title = "Tuning Results for Random Forest",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m4_s2b1 <- select_best(tune_results_m4_s2b1, metric = "roc_auc")

print(best_params_m4_s2b1)

# 8. Finalize the workflow
final_wf_m4_s2b1 <- finalize_workflow(wf_m4_s2b1, best_params_m4_s2b1)

# 9. Fit the final model
fit_m4_s2b1 <- fit(final_wf_m4_s2b1, data = df_m4_s2b1)

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m4_s2b1, df_m4_s2b1, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target =
      ifelse(tpr_1 - 0.81 > 0,
             (tpr_1 - 0.81) / 100,
             tpr_1 - 0.81) +
      ifelse(tpr_0 - 0.79 > 0,
             (tpr_0 - 0.79) / 100,
             tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.max(threshold_df$diff_from_target), "threshold"]

cat("Best threshold:", best_threshold, "\n")

# 10. Evaluate the model on the test dataset
results_m4_s2b1 <- calculate_all_measures(fit_m4_s2b1, df_test, 0.5)
results_m4_s2b1
store_results("m4s2b1", results_m4_s2b1, "Random Forest Model - s2b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m4_s2b1.RData")

# Optional: Extract variable importance
if (inherits(fit_m4_s2b1$fit$fit$fit, "ranger")) {
  var_imp <- ranger::importance(fit_m4_s2b1$fit$fit$fit)
  var_imp_df <- data.frame(Variable = names(var_imp), Importance = var_imp)
  var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]

  # Plot top 20 variables by importance
  ggplot(head(var_imp_df, 20),
         aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest Variable Importance",
         x = "",
         y = "Importance") +
    theme_minimal()

  ggsave("m4_s2b1_var_importance.png", width = 10, height = 8, dpi = 300)

}
#---- 5-4-4 DONE ***      Model 4 Random Forest ------------------- m4-s2b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use categorical variables for Random Forest
df_m4_s2b2 <- df_s2b2 %>% select(-matches("SERIALNO"))

# 1. Model Specification
spec_m4_s2b2 <- rand_forest(
  mtry = tune(),       # Number of predictors to sample at each split
  trees = tune(),      # Number of trees
  min_n = tune()       # Minimum node size
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# 2. Recipe
rec_m4_s2b2 <- recipe(Class ~ ., data = df_m4_s2b2) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) 

# 3. Workflow
wf_m4_s2b2 <- workflow() %>%
  add_model(spec_m4_s2b2) %>%
  add_recipe(rec_m4_s2b2)

# 4. Cross-validation
set.seed(123)
folds_m4_s2b2 <- vfold_cv(df_m4_s2b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
# For mtry, we'll try different numbers of predictors
num_predictors <- ncol(df_m4_s2b2) - 1
mtry_values <- floor(c(0.05, 0.1, 0.25, 0.5, 0.75) * num_predictors)
mtry_values <- unique(mtry_values[mtry_values > 0])

grid_m4_s2b2 <- grid_regular(
  mtry(range = range(mtry_values)),
  trees(range = c(100, 500)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m4_s2b2 <- tune_grid(
  wf_m4_s2b2,
  resamples = folds_m4_s2b2,
  grid = grid_m4_s2b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m4_s2b2) +
  labs(title = "Tuning Results for Random Forest",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m4_s2b2 <- select_best(tune_results_m4_s2b2, metric = "roc_auc")

print(best_params_m4_s2b2)

# 8. Finalize the workflow
final_wf_m4_s2b2 <- finalize_workflow(wf_m4_s2b2, best_params_m4_s2b2)

# 9. Fit the final model
fit_m4_s2b2 <- fit(final_wf_m4_s2b2, data = df_m4_s2b2)

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m4_s2b2, df_m4_s2b2, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target =
      ifelse(tpr_1 - 0.81 > 0,
             (tpr_1 - 0.81) / 100,
             tpr_1 - 0.81) +
      ifelse(tpr_0 - 0.79 > 0,
             (tpr_0 - 0.79) / 100,
             tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.max(threshold_df$diff_from_target), "threshold"]

cat("Best threshold:", best_threshold, "\n")

# 10. Evaluate the model on the test dataset
results_m4_s2b2 <- calculate_all_measures(fit_m4_s2b2, df_test, 0.5)
results_m4_s2b2
store_results("m4s2b2", results_m4_s2b2, "Random Forest Model - s2b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m4_s2b2.RData")

# Optional: Extract variable importance
if (inherits(fit_m4_s2b2$fit$fit$fit, "ranger")) {
  var_imp <- ranger::importance(fit_m4_s2b2$fit$fit$fit)
  var_imp_df <- data.frame(Variable = names(var_imp), Importance = var_imp)
  var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]

  # Plot top 20 variables by importance
  ggplot(head(var_imp_df, 20),
         aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest Variable Importance",
         x = "",
         y = "Importance") +
    theme_minimal()

  ggsave("m4_s2b2_var_importance.png", width = 10, height = 8, dpi = 300)

}
#---- 5-4-5 DONE ***      Model 4 Random Forest ------------------- m4-s3b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use categorical variables for Random Forest
df_m4_s3b1 <- df_s3b1 %>% select(-matches("SERIALNO"))

# 1. Model Specification
spec_m4_s3b1 <- rand_forest(
  mtry = tune(),       # Number of predictors to sample at each split
  trees = tune(),      # Number of trees
  min_n = tune()       # Minimum node size
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# 2. Recipe
rec_m4_s3b1 <- recipe(Class ~ ., data = df_m4_s3b1) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) 

# 3. Workflow
wf_m4_s3b1 <- workflow() %>%
  add_model(spec_m4_s3b1) %>%
  add_recipe(rec_m4_s3b1)

# 4. Cross-validation
set.seed(123)
folds_m4_s3b1 <- vfold_cv(df_m4_s3b1, v = 20, strata = Class)

# 5. Grid of hyperparameters
# For mtry, we'll try different numbers of predictors
num_predictors <- ncol(df_m4_s3b1) - 1
mtry_values <- floor(c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75) * num_predictors)
mtry_values <- unique(mtry_values[mtry_values > 0])

grid_m4_s3b1 <- grid_regular(
  mtry(range = range(mtry_values)),
  trees(range = c(100, 500)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m4_s3b1 <- tune_grid(
  wf_m4_s3b1,
  resamples = folds_m4_s3b1,
  grid = grid_m4_s3b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m4_s3b1) +
  labs(title = "Tuning Results for Random Forest",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m4_s3b1 <- select_best(tune_results_m4_s3b1, metric = "roc_auc")

print(best_params_m4_s3b1)

# 8. Finalize the workflow
final_wf_m4_s3b1 <- finalize_workflow(wf_m4_s3b1, best_params_m4_s3b1)

# 9. Fit the final model
fit_m4_s3b1 <- fit(final_wf_m4_s3b1, data = df_m4_s3b1)

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m4_s3b1, df_m4_s3b1, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target =
      ifelse(tpr_1 - 0.81 > 0,
             (tpr_1 - 0.81) / 100,
             tpr_1 - 0.81) +
      ifelse(tpr_0 - 0.79 > 0,
             (tpr_0 - 0.79) / 100,
             tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.max(threshold_df$diff_from_target), "threshold"]

cat("Best threshold:", best_threshold, "\n")

# 10. Evaluate the model on the test dataset
results_m4_s3b1 <- calculate_all_measures(fit_m4_s3b1, df_test, 0.5)
results_m4_s3b1
store_results("m4s3b1", results_m4_s3b1, "Random Forest Model - s3b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m4_s3b1.RData")

# Optional: Extract variable importance
if (inherits(fit_m4_s3b1$fit$fit$fit, "ranger")) {
  var_imp <- ranger::importance(fit_m4_s3b1$fit$fit$fit)
  var_imp_df <- data.frame(Variable = names(var_imp), Importance = var_imp)
  var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]

  # Plot top 20 variables by importance
  ggplot(head(var_imp_df, 20),
         aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest Variable Importance",
         x = "",
         y = "Importance") +
    theme_minimal()

  ggsave("m4_s3b1_var_importance.png", width = 10, height = 8, dpi = 300)

}

#---- 5-4-6 DONE ***      Model 4 Random Forest ------------------- m4-s3b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Use categorical variables for Random Forest
df_m4_s3b2 <- df_s3b2 %>% select(-matches("SERIALNO"))

# 1. Model Specification
spec_m4_s3b2 <- rand_forest(
  mtry = tune(),       # Number of predictors to sample at each split
  trees = tune(),      # Number of trees
  min_n = tune()       # Minimum node size
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# 2. Recipe
rec_m4_s3b2 <- recipe(Class ~ ., data = df_m4_s3b2) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) 

# 3. Workflow
wf_m4_s3b2 <- workflow() %>%
  add_model(spec_m4_s3b2) %>%
  add_recipe(rec_m4_s3b2)

# 4. Cross-validation
set.seed(123)
folds_m4_s3b2 <- vfold_cv(df_m4_s3b2, v = 20, strata = Class)

# 5. Grid of hyperparameters
# For mtry, we'll try different numbers of predictors
num_predictors <- ncol(df_m4_s3b2) - 1
mtry_values <- floor(c(0.025, 0.05, 0.1, 0.25, 0.5, 0.75) * num_predictors)
mtry_values <- unique(mtry_values[mtry_values > 0])

grid_m4_s3b2 <- grid_regular(
  mtry(range = range(mtry_values)),
  trees(range = c(100, 500)),
  min_n(range = c(2, 10)),
  levels = 5
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m4_s3b2 <- tune_grid(
  wf_m4_s3b2,
  resamples = folds_m4_s3b2,
  grid = grid_m4_s3b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m4_s3b2) +
  labs(title = "Tuning Results for Random Forest",
       x = "Tuned Parameter") +
  theme_minimal()

# 7. Select the best parameters
best_params_m4_s3b2 <- select_best(tune_results_m4_s3b2, metric = "roc_auc")

print(best_params_m4_s3b2)

# 8. Finalize the workflow
final_wf_m4_s3b2 <- finalize_workflow(wf_m4_s3b2, best_params_m4_s3b2)

# 9. Fit the final model
fit_m4_s3b2 <- fit(final_wf_m4_s3b2, data = df_m4_s3b2)

for (thresh in thresholds) {
  results <- calculate_all_measures(fit_m4_s3b2, df_m4_s3b2, thresh)
  tpr_1 <- results$values[results$measures == "TPR_1"]
  tpr_0 <- results$values[results$measures == "TPR_0"]

  threshold_results[[as.character(thresh)]] <- data.frame(
    threshold = thresh,
    TPR_1 = tpr_1,
    TPR_0 = tpr_0,
    diff_from_target =
      ifelse(tpr_1 - 0.81 > 0,
             (tpr_1 - 0.81) / 100,
             tpr_1 - 0.81) +
      ifelse(tpr_0 - 0.79 > 0,
             (tpr_0 - 0.79) / 100,
             tpr_0 - 0.79)
  )
}

threshold_df <- do.call(rbind, threshold_results)
best_threshold <-
  threshold_df[which.max(threshold_df$diff_from_target), "threshold"]

cat("Best threshold:", best_threshold, "\n")

# 10. Evaluate the model on the test dataset
results_m4_s3b2 <- calculate_all_measures(fit_m4_s3b2, df_test, 0.5)
results_m4_s3b2
store_results("m4s3b2", results_m4_s3b2, "Random Forest Model - s3b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m4_s3b2.RData")

# Optional: Extract variable importance
if (inherits(fit_m4_s3b2$fit$fit$fit, "ranger")) {
  var_imp <- ranger::importance(fit_m4_s3b2$fit$fit$fit)
  var_imp_df <- data.frame(Variable = names(var_imp), Importance = var_imp)
  var_imp_df <- var_imp_df[order(var_imp_df$Importance, decreasing = TRUE), ]

  # Plot top 20 variables by importance
  ggplot(head(var_imp_df, 20),
         aes(x = reorder(Variable, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Random Forest Variable Importance",
         x = "",
         y = "Importance") +
    theme_minimal()

  ggsave("m4_s3b2_var_importance.png", width = 10, height = 8, dpi = 300)

}

#---- 5-5 DONE *****    Model 5 Support Vect Machine ---------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 5-5-1 DONE ***      Model 5 Support Vector Machine ---------- m5-s1b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Support Vector Machine Model

df_m5_s1b1 <- df_s1b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m5_s1b1 <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 2. Recipe
rec_m5_s1b1 <- recipe(Class ~ ., data = df_m5_s1b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m5_s1b1 <- workflow() %>%
  add_model(spec_m5_s1b1) %>%
  add_recipe(rec_m5_s1b1)

# 4. Cross-validation
set.seed(123)
folds_m5_s1b1 <- vfold_cv(df_m5_s1b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m5_s1b1 <- grid_regular(
  cost(),
  rbf_sigma(),
  levels = 5
)

# 6. Tune the model
tune_results_m5_s1b1 <- tune_grid(
  wf_m5_s1b1,
  resamples = folds_m5_s1b1,
  grid = tune_grid_m5_s1b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m5_s1b1) +
  labs(title = "Tuning Results for Support Vector Machine",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m5_s1b1 <- select_best(tune_results_m5_s1b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m5_s1b1 <- finalize_workflow(wf_m5_s1b1, best_parameters_m5_s1b1)

# 9. Fit the final model
fit_m5_s1b1 <- fit(final_wf_m5_s1b1, data = df_m5_s1b1)

# 10. Evaluate the model on the test dataset
test_predications_m5_s1b1 <-
  predict(fit_m5_s1b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m5_s1b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m5_s1b1 <- test_predications_m5_s1b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m5_s1b1)

# Visualize the confusion matrix
autoplot(confusion_matrix_m5_s1b1, type = "heatmap") +
  labs(title = "Confusion Matrix for Support Vector Machine",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m5_s1b1 <- calculate_all_measures(fit_m5_s1b1, df_test, 0.5)

results_m5_s1b1

store_results("m5s1b1", results_m5_s1b1, "Support Vector Machine Model - s1b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m5_s1b1.RData")

#---- 5-5-2 DONE ***      Model 5 Support Vector Machine ---------- m5-s1b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Support Vector Machine Model

df_m5_s1b2 <- df_s1b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m5_s1b2 <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 2. Recipe
rec_m5_s1b2 <- recipe(Class ~ ., data = df_m5_s1b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m5_s1b2 <- workflow() %>%
  add_model(spec_m5_s1b2) %>%
  add_recipe(rec_m5_s1b2)

# 4. Cross-validation
set.seed(123)
folds_m5_s1b2 <- vfold_cv(df_m5_s1b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m5_s1b2 <- grid_regular(
  cost(),
  rbf_sigma(),
  levels = 5
)

# 6. Tune the model
tune_results_m5_s1b2 <- tune_grid(
  wf_m5_s1b2,
  resamples = folds_m5_s1b2,
  grid = tune_grid_m5_s1b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m5_s1b2) +
  labs(title = "Tuning Results for Support Vector Machine",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m5_s1b2 <- select_best(tune_results_m5_s1b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m5_s1b2 <- finalize_workflow(wf_m5_s1b2, best_parameters_m5_s1b2)

# 9. Fit the final model
fit_m5_s1b2 <- fit(final_wf_m5_s1b2, data = df_m5_s1b2)

# 10. Evaluate the model on the test dataset
test_predications_m5_s1b2 <-
  predict(fit_m5_s1b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m5_s1b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m5_s1b2 <- test_predications_m5_s1b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m5_s1b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m5_s1b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Support Vector Machine",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m5_s1b2 <- calculate_all_measures(fit_m5_s1b2, df_test, 0.5)

results_m5_s1b2

store_results("m5s1b2", results_m5_s1b2, "Support Vector Machine Model - s1b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m5_s1b2.RData")

#---- 5-5-3 DONE ***      Model 5 Support Vector Machine ---------- m5-s2b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Support Vector Machine Model

df_m5_s2b1 <- df_s2b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m5_s2b1 <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 2. Recipe
rec_m5_s2b1 <- recipe(Class ~ ., data = df_m5_s2b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m5_s2b1 <- workflow() %>%
  add_model(spec_m5_s2b1) %>%
  add_recipe(rec_m5_s2b1)

# 4. Cross-validation
set.seed(123)
folds_m5_s2b1 <- vfold_cv(df_m5_s2b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m5_s2b1 <- grid_regular(
  cost(),
  rbf_sigma(),
  levels = 5
)

# 6. Tune the model
tune_results_m5_s2b1 <- tune_grid(
  wf_m5_s2b1,
  resamples = folds_m5_s2b1,
  grid = tune_grid_m5_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m5_s2b1) +
  labs(title = "Tuning Results for Support Vector Machine",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m5_s2b1 <- select_best(tune_results_m5_s2b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m5_s2b1 <- finalize_workflow(wf_m5_s2b1, best_parameters_m5_s2b1)

# 9. Fit the final model
fit_m5_s2b1 <- fit(final_wf_m5_s2b1, data = df_m5_s2b1)

# 10. Evaluate the model on the test dataset
test_predications_m5_s2b1 <-
  predict(fit_m5_s2b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m5_s2b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m5_s2b1 <- test_predications_m5_s2b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m5_s2b1)

# Visualize the confusion matrix
autoplot(confusion_matrix_m5_s2b1, type = "heatmap") +
  labs(title = "Confusion Matrix for Support Vector Machine",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m5_s2b1 <- calculate_all_measures(fit_m5_s2b1, df_test, 0.5)

results_m5_s2b1

store_results("m5s2b1", results_m5_s2b1, "Support Vector Machine Model - s2b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m5_s2b1.RData")

#---- 5-5-4 DONE ***      Model 5 Support Vector Machine ---------- m5-s2b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Support Vector Machine Model

df_m5_s2b2 <- df_s2b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m5_s2b2 <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 2. Recipe
rec_m5_s2b2 <- recipe(Class ~ ., data = df_m5_s2b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m5_s2b2 <- workflow() %>%
  add_model(spec_m5_s2b2) %>%
  add_recipe(rec_m5_s2b2)

# 4. Cross-validation
set.seed(123)
folds_m5_s2b2 <- vfold_cv(df_m5_s2b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m5_s2b2 <- grid_regular(
  cost(),
  rbf_sigma(),
  levels = 5
)

# 6. Tune the model
tune_results_m5_s2b2 <- tune_grid(
  wf_m5_s2b2,
  resamples = folds_m5_s2b2,
  grid = tune_grid_m5_s2b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m5_s2b2) +
  labs(title = "Tuning Results for Support Vector Machine",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m5_s2b2 <- select_best(tune_results_m5_s2b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m5_s2b2 <- finalize_workflow(wf_m5_s2b2, best_parameters_m5_s2b2)

# 9. Fit the final model
fit_m5_s2b2 <- fit(final_wf_m5_s2b2, data = df_m5_s2b2)

# 10. Evaluate the model on the test dataset
test_predications_m5_s2b2 <-
  predict(fit_m5_s2b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m5_s2b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m5_s2b2 <- test_predications_m5_s2b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m5_s2b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m5_s2b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Support Vector Machine",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m5_s2b2 <- calculate_all_measures(fit_m5_s2b2, df_test, 0.5)

results_m5_s2b2

store_results("m5s2b2", results_m5_s2b2, "Support Vector Machine Model - s2b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m5_s2b2.RData")

#---- 5-5-5 DONE ***      Model 5 Support Vector Machine ---------- m5-s3b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Support Vector Machine Model

df_m5_s3b1 <- df_s3b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m5_s3b1 <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 2. Recipe
rec_m5_s3b1 <- recipe(Class ~ ., data = df_m5_s3b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m5_s3b1 <- workflow() %>%
  add_model(spec_m5_s3b1) %>%
  add_recipe(rec_m5_s3b1)

# 4. Cross-validation
set.seed(123)
folds_m5_s3b1 <- vfold_cv(df_m5_s3b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m5_s3b1 <- grid_regular(
  cost(),
  rbf_sigma(),
  levels = 5
)

# 6. Tune the model
tune_results_m5_s3b1 <- tune_grid(
  wf_m5_s3b1,
  resamples = folds_m5_s3b1,
  grid = tune_grid_m5_s3b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m5_s3b1) +
  labs(title = "Tuning Results for Support Vector Machine",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m5_s3b1 <- select_best(tune_results_m5_s3b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m5_s3b1 <- finalize_workflow(wf_m5_s3b1, best_parameters_m5_s3b1)

# 9. Fit the final model
fit_m5_s3b1 <- fit(final_wf_m5_s3b1, data = df_m5_s3b1)

# 10. Evaluate the model on the test dataset
test_predications_m5_s3b1 <-
  predict(fit_m5_s3b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m5_s3b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m5_s3b1 <- test_predications_m5_s3b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m5_s3b1)

# Visualize the confusion matrix
autoplot(confusion_matrix_m5_s3b1, type = "heatmap") +
  labs(title = "Confusion Matrix for Support Vector Machine",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m5_s3b1 <- calculate_all_measures(fit_m5_s3b1, df_test, 0.5)

results_m5_s3b1

store_results("m5s3b1", results_m5_s3b1, "Support Vector Machine Model - s3b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m5_s3b1.RData")

#---- 5-5-6 DONE ***      Model 5 Support Vector Machine ---------- m5-s3b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Support Vector Machine Model

df_m5_s3b2 <- df_s3b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m5_s3b2 <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# 2. Recipe
rec_m5_s3b2 <- recipe(Class ~ ., data = df_m5_s3b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m5_s3b2 <- workflow() %>%
  add_model(spec_m5_s3b2) %>%
  add_recipe(rec_m5_s3b2)

# 4. Cross-validation
set.seed(123)
folds_m5_s3b2 <- vfold_cv(df_m5_s3b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m5_s3b2 <- grid_regular(
  cost(),
  rbf_sigma(),
  levels = 5
)

# 6. Tune the model
tune_results_m5_s3b2 <- tune_grid(
  wf_m5_s3b2,
  resamples = folds_m5_s3b2,
  grid = tune_grid_m5_s3b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m5_s3b2) +
  labs(title = "Tuning Results for Support Vector Machine",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m5_s3b2 <- select_best(tune_results_m5_s3b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m5_s3b2 <- finalize_workflow(wf_m5_s3b2, best_parameters_m5_s3b2)

# 9. Fit the final model
fit_m5_s3b2 <- fit(final_wf_m5_s3b2, data = df_m5_s3b2)

# 10. Evaluate the model on the test dataset
test_predications_m5_s3b2 <-
  predict(fit_m5_s3b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m5_s3b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m5_s3b2 <- test_predications_m5_s3b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m5_s3b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m5_s3b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Support Vector Machine",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m5_s3b2 <- calculate_all_measures(fit_m5_s3b2, df_test, 0.5)

results_m5_s3b2

store_results("m5s3b2", results_m5_s3b2, "Support Vector Machine Model - s3b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m5_s3b2.RData")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-6 PEND *****    Model 6 Gradient Boosting ------------------------------

#---- 5-6-1 DONE ***      Model 6 Gradient Boosting --------------- m6-s1b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s1b1 <- df_s1b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m6_s1b1 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s1b1 <- recipe(Class ~ ., data = df_m6_s1b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m6_s1b1 <- workflow() %>%
  add_model(spec_m6_s1b1) %>%
  add_recipe(rec_m6_s1b1)

# 4. Cross-validation
set.seed(123)
folds_m6_s1b1 <- vfold_cv(df_m6_s1b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m6_s1b1 <- grid_regular(
  trees(range = c(2000, 6000)),
  tree_depth(range = c(2, 4)),
  learn_rate(range = c(-5, -1), trans = log10_trans()),
  min_n(range = c(2, 10)),
  levels = 3
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m6_s1b1 <- tune_grid(
  wf_m6_s1b1,
  resamples = folds_m6_s1b1,
  grid = tune_grid_m6_s1b1,
  metrics = metric_set(roc_auc, bal_accuracy, sens, yardstick::specificity)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m6_s1b1) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m6_s1b1 <- select_best(tune_results_m6_s1b1,
                                       metric = "bal_accuracy")

print(best_parameters_m6_s1b1)

# 8. Finalize the workflow
final_wf_m6_s1b1 <- finalize_workflow(wf_m6_s1b1, best_parameters_m6_s1b1)

# 9. Fit the final model
fit_m6_s1b1 <- fit(final_wf_m6_s1b1, data = df_m6_s1b1)

# 10. Evaluate the model on the test dataset
test_predications_m6_s1b1 <-
  predict(fit_m6_s1b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s1b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s1b1 <- test_predications_m6_s1b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s1b1)

results_m6_s1b1 <- calculate_all_measures(fit_m6_s1b1, df_test, 0.5)

results_m6_s1b1

store_results("m6s1b1", results_m6_s1b1, "Gradient Boosting Model - s1b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s1b1.RData")

#---- 5-6-2 DONE ***      Model 6 Gradient Boosting --------------- m6-s1b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s1b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s1b2 <- df_s1b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m6_s1b2 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s1b2 <- recipe(Class ~ ., data = df_m6_s1b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m6_s1b2 <- workflow() %>%
  add_model(spec_m6_s1b2) %>%
  add_recipe(rec_m6_s1b2)

# 4. Cross-validation
set.seed(123)
folds_m6_s1b2 <- vfold_cv(df_m6_s1b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m6_s1b2 <- grid_regular(
  trees(range = c(100, 500)),
  tree_depth(range = c(3, 9)),
  learn_rate(range = c(-5, -1), trans = log10_trans()),
  min_n(range = c(2, 10)),
  levels = 5
)

# 6. Tune the model
tune_results_m6_s1b2 <- tune_grid(
  wf_m6_s1b2,
  resamples = folds_m6_s1b2,
  grid = tune_grid_m6_s1b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m6_s1b2) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m6_s1b2 <- select_best(tune_results_m6_s1b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m6_s1b2 <- finalize_workflow(wf_m6_s1b2, best_parameters_m6_s1b2)

# 9. Fit the final model
fit_m6_s1b2 <- fit(final_wf_m6_s1b2, data = df_m6_s1b2)

# 10. Evaluate the model on the test dataset
test_predications_m6_s1b2 <-
  predict(fit_m6_s1b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s1b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s1b2 <- test_predications_m6_s1b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s1b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m6_s1b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Gradient Boosting",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m6_s1b2 <- calculate_all_measures(fit_m6_s1b2, df_test, 0.5)

results_m6_s1b2

store_results("m6s1b2", results_m6_s1b2, "Gradient Boosting Model - s1b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s1b2.RData")

#---- 5-6-3 DONE ***      Model 6 Gradient Boosting --------------- m6-s2b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s2b1 <- df_s2b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_"))) %>% select(-matches("SERIALNO"))

# 1. Model Specification
spec_m6_s2b1 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s2b1 <- recipe(Class ~ ., data = df_m6_s2b1) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m6_s2b1 <- workflow() %>%
  add_model(spec_m6_s2b1) %>%
  add_recipe(rec_m6_s2b1)

# 4. Cross-validation
set.seed(123)
folds_m6_s2b1 <- vfold_cv(df_m6_s2b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m6_s2b1 <- grid_regular(
  trees(range = c(100, 500)),
  tree_depth(range = c(3, 9)),
  learn_rate(range = c(-5, -1), trans = log10_trans()),
  min_n(range = c(2, 10)),
  levels = 5)

#tune_grid_m6_s2b1 <- grid_latin_hypercube(
#  trees(range = c(100, 500)),
#  tree_depth(range = c(3, 9)),
#  learn_rate(range = c(-5, -1), trans = log10_trans()),
#  min_n(range = c(2, 10)),
#  levels = 5
#)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m6_s2b1 <- tune_grid(
  wf_m6_s2b1,
  resamples = folds_m6_s2b1,
  grid = tune_grid_m6_s2b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m6_s2b1) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m6_s2b1 <- select_best(tune_results_m6_s2b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m6_s2b1 <- finalize_workflow(wf_m6_s2b1, best_parameters_m6_s2b1)

# 9. Fit the final model
fit_m6_s2b1 <- fit(final_wf_m6_s2b1, data = df_m6_s2b1)

# 10. Evaluate the model on the test dataset
test_predications_m6_s2b1 <-
  predict(fit_m6_s2b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s2b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s2b1 <- test_predications_m6_s2b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s2b1)

results_m6_s2b1 <- calculate_all_measures(fit_m6_s2b1, df_test, 0.5)

results_m6_s2b1

store_results("m6s2b1", results_m6_s2b1, "Gradient Boosting Model - s2b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s2b1.RData")

#---- 5-6-4 DONE ***      Model 6 Gradient Boosting --------------- m6-s2b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s2b2 <- df_s2b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m6_s2b2 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s2b2 <- recipe(Class ~ ., data = df_m6_s2b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m6_s2b2 <- workflow() %>%
  add_model(spec_m6_s2b2) %>%
  add_recipe(rec_m6_s2b2)

# 4. Cross-validation
set.seed(123)
folds_m6_s2b2 <- vfold_cv(df_m6_s2b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m6_s2b2 <- grid_regular(
  trees(range = c(100, 500)),
  tree_depth(range = c(3, 9)),
  learn_rate(range = c(-5, -1), trans = log10_trans()),
  min_n(range = c(2, 10)),
  levels = 5
)

# 6. Tune the model
tune_results_m6_s2b2 <- tune_grid(
  wf_m6_s2b2,
  resamples = folds_m6_s2b2,
  grid = tune_grid_m6_s2b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m6_s2b2) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m6_s2b2 <- select_best(tune_results_m6_s2b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m6_s2b2 <- finalize_workflow(wf_m6_s2b2, best_parameters_m6_s2b2)

# 9. Fit the final model
fit_m6_s2b2 <- fit(final_wf_m6_s2b2, data = df_m6_s2b2)

# 10. Evaluate the model on the test dataset
test_predications_m6_s2b2 <-
  predict(fit_m6_s2b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s2b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s2b2 <- test_predications_m6_s2b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s2b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m6_s2b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Gradient Boosting",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m6_s2b2 <- calculate_all_measures(fit_m6_s2b2, df_test, 0.5)

results_m6_s2b2

store_results("m6s2b2", results_m6_s2b2, "Gradient Boosting Model - s2b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s2b2.RData")
#---- 5-6-5 DONE ***      Model 6 Gradient Boosting --------------- m6-s3b1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b1.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s3b1 <- df_s3b1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m6_s3b1 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s3b1 <- recipe(Class ~ ., data = df_m6_s3b1) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m6_s3b1 <- workflow() %>%
  add_model(spec_m6_s3b1) %>%
  add_recipe(rec_m6_s3b1)

# 4. Cross-validation
set.seed(123)
folds_m6_s3b1 <- vfold_cv(df_m6_s3b1, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m6_s3b1 <- grid_regular(
  trees(range = c(100, 500)),
  tree_depth(range = c(3, 9)),
  learn_rate(range = c(-5, -1), trans = log10_trans()),
  min_n(range = c(2, 10)),
  levels = 5
)

# 6. Tune the model
tune_results_m6_s3b1 <- tune_grid(
  wf_m6_s3b1,
  resamples = folds_m6_s3b1,
  grid = tune_grid_m6_s3b1,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m6_s3b1) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m6_s3b1 <- select_best(tune_results_m6_s3b1, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m6_s3b1 <- finalize_workflow(wf_m6_s3b1, best_parameters_m6_s3b1)

# 9. Fit the final model
fit_m6_s3b1 <- fit(final_wf_m6_s3b1, data = df_m6_s3b1)

# 10. Evaluate the model on the test dataset
test_predications_m6_s3b1 <-
  predict(fit_m6_s3b1, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s3b1, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s3b1 <- test_predications_m6_s3b1 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s3b1)

# Visualize the confusion matrix
autoplot(confusion_matrix_m6_s3b1, type = "heatmap") +
  labs(title = "Confusion Matrix for Gradient Boosting",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m6_s3b1 <- calculate_all_measures(fit_m6_s3b1, df_test, 0.5)

results_m6_s3b1

store_results("m6s3b1", results_m6_s3b1, "Gradient Boosting Model - s3b1")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s3b1.RData")

#---- 5-6-6 DONE ***      Model 6 Gradient Boosting --------------- m6-s3b2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s3b2.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s3b2 <- df_s3b2 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

# 1. Model Specification
spec_m6_s3b2 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s3b2 <- recipe(Class ~ ., data = df_m6_s3b2) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes())

# 3. Workflow
wf_m6_s3b2 <- workflow() %>%
  add_model(spec_m6_s3b2) %>%
  add_recipe(rec_m6_s3b2)

# 4. Cross-validation
set.seed(123)
folds_m6_s3b2 <- vfold_cv(df_m6_s3b2, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m6_s3b2 <- grid_regular(
  trees(range = c(100, 500)),
  tree_depth(range = c(3, 9)),
  learn_rate(range = c(-5, -1), trans = log10_trans()),
  min_n(range = c(2, 10)),
  levels = 5
)

# 6. Tune the model
tune_results_m6_s3b2 <- tune_grid(
  wf_m6_s3b2,
  resamples = folds_m6_s3b2,
  grid = tune_grid_m6_s3b2,
  metrics = metric_set(roc_auc, accuracy, sens, spec)
)

# Show the tuning results
autoplot(tune_results_m6_s3b2) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m6_s3b2 <- select_best(tune_results_m6_s3b2, metric = "roc_auc")

# 8. Finalize the workflow
final_wf_m6_s3b2 <- finalize_workflow(wf_m6_s3b2, best_parameters_m6_s3b2)

# 9. Fit the final model
fit_m6_s3b2 <- fit(final_wf_m6_s3b2, data = df_m6_s3b2)

# 10. Evaluate the model on the test dataset
test_predications_m6_s3b2 <-
  predict(fit_m6_s3b2, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s3b2, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s3b2 <- test_predications_m6_s3b2 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s3b2)

# Visualize the confusion matrix
autoplot(confusion_matrix_m6_s3b2, type = "heatmap") +
  labs(title = "Confusion Matrix for Gradient Boosting",
       x = "Predicted Class",
       y = "Actual Class") +
  theme_minimal()

results_m6_s3b2 <- calculate_all_measures(fit_m6_s3b2, df_test, 0.5)

results_m6_s3b2

store_results("m6s3b2", results_m6_s3b2, "Gradient Boosting Model - s3b2")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s3b2.RData")

#---- 5-6-7 DONE ***      Model 6 Gradient Boosting --------------- m6-s4b3 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s4b3.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s4b3 <- df_s4b3 %>% select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))
#"DETAILED-PUBCOV_Public health coverage recode",
#"DETAILED-WRK_Worked last week") #,
#"DETAILED-INDP_Industry recode for 2023 and later based on 2022 IND codes")
#"DETAILED-ESR_Employment status recode",
#"DETAILED-MSP_Married, spouse present/spouse absent", # nolint
#"DETAILED-POBP_Place of birth (Recode)",
#"DETAILED-SEX_Sex",
#"DETAILED-HINS1_Insurance through a current or former employer or union",
#"DETAILED-DECADE_Decade of entry", "DETAILED-NWAB_Temporary absence from work (UNEDITED - See 'Employment Status Recode' (ESR))") # nolint

# 1. Model Specification
spec_m6_s4b3 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost", scale_pos_weight = 0.07) %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s4b3 <- recipe(Class ~ ., data = df_m6_s4b3) %>%
  # Handle missing values
  step_impute_median(all_numeric_predictors()) %>%
  step_unknown(all_nominal_predictors(), new_level = "Missing") %>%
  # Transform skewed numeric features
  #step_YeoJohnson(all_numeric_predictors()) %>%
  # Normalize numeric features
  #step_normalize(all_numeric_predictors()) %>%
  # Proper encoding for categorical variables
  step_dummy(all_nominal_predictors()) %>%
  # Remove zero-variance predictors
  step_zv(all_predictors()) %>%
  # Remove highly correlated features
  step_corr(all_numeric_predictors(), threshold = 0.9)

# 3. Workflow
wf_m6_s4b3 <- workflow() %>%
  add_model(spec_m6_s4b3) %>%
  add_recipe(rec_m6_s4b3)

# 4. Cross-validation
set.seed(123)
folds_m6_s4b3 <- vfold_cv(df_m6_s4b3, v = 5, strata = Class)

# 5. Grid of hyperparameters
tune_grid_m6_s4b3 <- grid_regular(
  trees(range = c(500, 2000)),
  tree_depth(range = c(2, 4)),
  learn_rate(range = c(-2.5, -1.5), trans = log10_trans()),
  min_n(range = c(0, 2)),
  levels = 3
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m6_s4b3 <- tune_grid(
  wf_m6_s4b3,
  resamples = folds_m6_s4b3,
  grid = tune_grid_m6_s4b3,
  metrics = metric_set(yardstick::specificity, sens, roc_auc, bal_accuracy)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Extract metrics in a readable format
tune_metrics <- collect_metrics(tune_results_m6_s4b3) %>%
  arrange(desc(mean))

# Show the tuning results
autoplot(tune_results_m6_s4b3) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# Summarize best parameters for each metric
cat("\nBest Parameters by Metric:\n")
tune_metrics %>%
  group_by(.metric) %>%
  slice_max(mean, n = 1) %>%
  ungroup() %>%
  select(trees, tree_depth, learn_rate, min_n, .metric, mean) %>%
  mutate(
    learn_rate = sprintf("%.5f", learn_rate),
    mean = sprintf("%.4f", mean)
  ) %>%
  print(width = Inf)

# 7. Select the best parameters
best_parameters_m6_s4b3 <-
  select_best(tune_results_m6_s4b3, metric = "bal_accuracy")

print(best_parameters_m6_s4b3)

# 8. Finalize the workflow
final_wf_m6_s4b3 <- finalize_workflow(wf_m6_s4b3, best_parameters_m6_s4b3)

# 9. Fit the final model
fit_m6_s4b3 <- fit(final_wf_m6_s4b3, data = df_m6_s4b3)

test_predications_m6_s4b3 <-
  predict(fit_m6_s4b3, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s4b3, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s4b3 <- test_predications_m6_s4b3 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s4b3)

results_m6_s4b3 <- calculate_all_measures(fit_m6_s4b3, df_test, 0.5)

results_m6_s4b3

store_results("m6s4b3", results_m6_s4b3, "Gradient Boosting Model - s4b3")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s4b3.RData")

#---- 5-6-8 DONE ***      Model 6 Gradient Boosting --------------- m6-s2b3 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#load("df_s2b3.RData") # nolint
#load("df_columns_info.RData") # nolint
#load("df_test.RData") # nolint

# Gradient Boosting Model

df_m6_s2b3 <- df_s2b3

# 1. Model Specification
spec_m6_s2b3 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost", scale_pos_weight = .1) %>%
  set_mode("classification")

# 2. Recipe
rec_m6_s2b3 <- recipe(Class ~ ., data = df_m6_s2b3) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) #%>%
  #step_normalize(all_predictors())

# 3. Workflow
wf_m6_s2b3 <- workflow() %>%
  add_model(spec_m6_s2b3) %>%
  add_recipe(rec_m6_s2b3)

# 4. Cross-validation
set.seed(123)
folds_m6_s2b3 <- vfold_cv(df_m6_s2b3, v = 5, strata = Class)

# 5. Grid of hyperparameters
# 5. Grid of hyperparameters

#.789, .758
#tune_grid_m6_s2b3 <- grid_regular(
#  trees(range = c(100, 500)),
#  tree_depth(range = c(2, 6)),
#  learn_rate(range = c(-4, -1.5), trans = log10_trans()),
#  min_n(range = c(2, 6)),
#  levels = 3
#)

tune_grid_m6_s2b3 <- grid_regular(
  trees(range = c(10, 200)),
  tree_depth(range = c(2, 4)),
  learn_rate(range = c(-2.5, -1), trans = log10_trans()),
  min_n(range = c(2, 6)),
  levels = 3
)

# Determine number of cores to use (leave one core free)
n_cores <- parallel::detectCores() - 1
n_cores <- max(n_cores, 1)  # Ensure at least one core

# Set the parallel plan - this activates parallel processing
# plan(multisession, workers = n_cores)  # For Windows # nolint
plan(multicore, workers = n_cores)   # For Unix/Linux/Mac

# Display information about parallel processing
cat("Using", n_cores, "cores for parallel processing\n")

# 6. Tune the model
tune_results_m6_s2b3 <- tune_grid(
  wf_m6_s2b3,
  resamples = folds_m6_s2b3,
  grid = tune_grid_m6_s2b3,
  metrics = metric_set(roc_auc, bal_accuracy, sens, yardstick::specificity)
)

# Reset the future plan to sequential
plan(sequential)
# Unregister the parallel backend
registerDoSEQ()  # Switch back to sequential processing
# Display information about stopping parallel processing
cat("Stopped parallel processing\n")

# Show the tuning results
autoplot(tune_results_m6_s2b3) +
  labs(title = "Tuning Results for Gradient Boosting",
       x = "Tuned Parameter",
       y = "Performance") +
  theme_minimal()

# 7. Select the best parameters
best_parameters_m6_s2b3 <- select_best(tune_results_m6_s2b3,
                                       metric = "bal_accuracy")

print(best_parameters_m6_s2b3)

# 8. Finalize the workflow
final_wf_m6_s2b3 <- finalize_workflow(wf_m6_s2b3, best_parameters_m6_s2b3)

# 9. Fit the final model
fit_m6_s2b3 <- fit(final_wf_m6_s2b3, data = df_m6_s2b3)

# 10. Evaluate the model on the test dataset
test_predications_m6_s2b3 <-
  predict(fit_m6_s2b3, new_data = df_test, type = "prob") %>%
  bind_cols(predict(fit_m6_s2b3, new_data = df_test, type = "class")) %>%
  bind_cols(df_test %>% select(Class))

# Generate a confusion matrix
confusion_matrix_m6_s2b3 <- test_predications_m6_s2b3 %>%
  conf_mat(truth = Class, estimate = .pred_class)

# Print the confusion matrix
print(confusion_matrix_m6_s2b3)

results_m6_s2b3 <- calculate_all_measures(fit_m6_s2b3, df_test, 0.5)

results_m6_s2b3

store_results("m6s2b3", results_m6_s2b3, "Gradient Boosting Model - s2b3")

# Save the results to an RData file
save(results_storage, file = "results_after_m6_s2b3.RData")

#---- 6 DONE *******      Final Steps ------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Export the results to a CSV file
results_storage %>%
  bind_rows() %>%
  write_csv("results_storage.csv")
