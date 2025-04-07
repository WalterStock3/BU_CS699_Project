#---- 0 ******* Project Step 0 -------------------------------------------------
## TODO:
#  - Complete a graph for Fisher Scores for Logical - 4-2-1-1
#  - Complete a grpah for Factor wihout processing Missing.
#  - There are a lot of Invalid Numbers in df - look into these.

# Project Goal (Lecture 1): Generate a model to predict the likelihood of a
# person having difficulty living independently.

# Dataset (Lecture 1): Part of the 2023 American Community Survey modified by
# the class professor for this project.

# Target Variable: The Class variable represents difficulty living independently
# and is binary (Yes(1)/No(0)).

# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(caret)
library(rsample)
library(ROSE)

################################################################################
#---- 1 ******* Preprocess - Project Step 1 --------------- df_preprocessed ----
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

df_preprocessed <- df

# Export df_preprocessed as a CSV file
write.csv(df_preprocessed, file = "df_preprocessed.csv", row.names = FALSE)

# Export df_preprocessed as an R data file
save(df_preprocessed, file = "df_preprocessed.RData")

################################################################################
#---- 2 ******* Split - Project Step 2 ------------------ df_train, df_test ----
################################################################################

# Optional - Load the preprocessed dataset
# load("df_preprocessed.RData")

set.seed(1)

split <- initial_split(df_preprocessed, prop = 0.7, strata = "Class")
df_train <- training(split)
df_test <- testing(split)

print(paste("training dataset - dim:", dim(train)[1], ",", dim(train)[2]))
print(paste("testing dataset - dim:", dim(test)[1], ",", dim(test)[2]))
print(paste("training dataset - class distribution:",
            table(train$Class)[1], ",", table(train$Class)[2]))
print(paste("testing dataset - class distribution:",
            table(test$Class)[1], ",", table(test$Class)[2]))

# Save df_train and df_test to R data files
save(df_train, file = "df_train.RData")
save(df_test, file = "df_test.RData")

################################################################################
#---- 3 ******* Balance - Project Step 3 ------- df_balanced1, df_balanced2 ----
################################################################################

# Not using SMOTE because we have a large number of categorical variables.

# Load the training and testing datasets
# load("df_train.RData")
# load("df_test.RData")

# Verify the dimensions of the loaded datasets
# print(paste("Loaded training dataset - dim:", dim(df_train)[1], ",", dim(df_train)[2]))
# print(paste("Loaded testing dataset - dim:", dim(df_test)[1], ",", dim(df_test)[2]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 3.1 *****    Balance - Method 1 - Down Sample ---------- df_balanced1 ----
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
#---- 3.2 *****    Balance - Method 2 - Up Sample ------------ df_balanced2 ----
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
#---- 4 ******* Select - Project Step 4 -------------- df_select#_balanced# ----
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
#---- 4-1 *****    Select 1 - Missing Removal -------- df_select1_balanced# ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-1-1 ***       Select 1 - balanced 1 ---------- df_select1_balanced1 ----
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

#---- 4-1-2 ***       Select 1 - balanced 2 ---------- df_select1_balanced2 ----
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
#---- 4-2 *****    Select 2 - Fisher and Corr -------- df_select2_balanced# ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-2-1 ***       Select 2 - balanced 1 ---------- df_select2_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_select2_balanced1 <- df_balanced1

#---- 4-2-1-1 *          Factor and Logical Variables --------------------------

in_row_limit <- 2

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

df_select2_balanced1_merged <- cbind(df_select2_balanced1_1factors,
                                     df_select2_balanced1_2logical,
                                     df_select2_balanced1_3levels)

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

ggplot(df_sel2_bal1_fisher_plt,
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

# Display the Fisher results ordered by -log10(P_value) from largest to smallest
fisher_results_df %>%
  arrange(desc(-log10(P_value))) %>%
  head(10) %>%
  print()

# Identify the in_row_limit columns with the highest neg_log10_P_value
top_cols <- df_sel2_bal1_fisher_results %>%
  arrange(desc(neg_log10_P_value)) %>%
  slice_head(n = in_row_limit) %>%
  pull(Column)

# Create a new dataframe with those columns
df_select2_balanced1 <- df_select2_balanced1 %>%
  select(all_of(top_cols))

#---- 4-2-1-2 *          Integer Variables -------------------------------------

df_select2_balanced1_4integers <- df_balanced1 %>%
  select(Class, matches(paste0("^DETAILED-(",
                               paste(df_columns_info %>%
                                       filter(variable_type %in%
                                                c("integer")) %>%
                                       pull(column_name),
                                     collapse = "|"), ")_")))

in_select1_cor_threshold <- 0.1

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

#---- 4-2-1-3 *          Outliers ----------------------------------------------
# Create boxplots for each numeric variable in the dataset

# Generate boxplots dynamically for all numeric columns
boxplots <- lapply(df_select2_balanced1_4integers, function(col) {
  ggplot(df_select2_balanced1_4integers, aes(x = "", y = .data[[col]])) +
    geom_boxplot() +
    theme(axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank())
})

# Arrange boxplots in a grid
#boxplots <- boxplots[order(colnames(df_select2_balanced1_4integers))]

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

grid.arrange(grobs = plt_list, ncol = 7)

#---- 4-2-1-4 *          Final -------------------------------------------------

save(df_select2_balanced1, file = "df_select2_balanced1.RData")

#---- 4-2-2 ***       Select 2 - balanced 2 ---------- df_select2_balanced2 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_select2_balanced2 <- df_balanced2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 4-3 *****    Select 3 - Missing Added ---------- df_select3_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#---- 4-3-1 ***       Select 3 - balanced 1 ---------- df_select3_balanced1 ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_select3_balanced1 <- df_balanced1

#---- 4-3-1-1 *          Factor and Logical Variables --------------------------

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

#---- 4-3-2 ***       Select - Method 3 balanced dataset 2 ---------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_select3_balanced2 <- df_balanced2

################################################################################
#---- 5 ******* Models - Project Step 5 ----------------------------------------
################################################################################

## Models - Project Step 5
# Lecture 1 - Classification Slide:
# * Decision trees,
# * Naive Bayesian classification,
# * Support vector machines,
# * K-nearest neighbors,
# * Neural networks.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-1 ***** Model 1 Logistic Regression ------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

#         Actual
#Predicted   0   1
#        0 923  24
#        1 286  63

# TPR No
tp_0 <- confusion_matrix[1, 1]
fp_0 <- confusion_matrix[1, 2]
tn_0 <- confusion_matrix[2, 2]
fn_0 <- confusion_matrix[2, 1]
tpr_0 <- tp_0 / (tp_0 + fn_0)
print(paste("Class No True Positive Rate (TPR):", tpr_0))

# TPR Yes
tp_1 <- confusion_matrix[2, 2]
fp_1 <- confusion_matrix[2, 1]
tn_1 <- confusion_matrix[1, 1]
fn_1 <- confusion_matrix[1, 2]
tpr_1 <- tp_1 / (tp_1 + fn_1)
print(paste("Class Yes True Positive Rate (TPR):", tpr_1))

calculate_measures <- function(tp_0, fp_0, tn_0, fn_0, tp_1, fp_1, tn_1, fn_1) {
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
  # Kappa statistic - starting
  total_0 <- (tp_0 + fn_0 + fp_0 + tn_0)
  p_o_0 <- (tp_0 + tn_0) / total_0
  p_e1_0 <- ((tp_0 + fn_0) / total_0) * ((tp_0 + fp_0) / total_0)
  p_e2_0 <- ((fp_0 + tn_0) / total_0) * ((fn_0 + tn_0) / total_0)
  p_e_0 <- p_e1_0 + p_e2_0
  k_0 <- (p_o_0 - p_e_0) / (1 - p_e_0) # Kappa statistic

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
  p_o_1 <- (tp_1 + tn_1) / total_1
  p_e1_1 <- ((tp_1 + fn_1) / total_1) * ((tp_1 + fp_1) / total_1)
  p_e2_1 <- ((fp_1 + tn_1) / total_1) * ((fn_1 + tn_1) / total_1)
  p_e_1 <- p_e1_1 + p_e2_1
  k_1 <- (p_o_1 - p_e_1) / (1 - p_e_1) # Kappa statistic

  measures <- c("TPR_0", "FPR_0", "TNR_0", "FNR_0",
                "Precision_0", "Recall_0", "F-measure_0", "MCC_0", "Kappa_0",
                "TPR_1", "FPR_1", "TNR_1", "FNR_1",
                "Precision_1", "Recall_1", "F-measure_1", "MCC_1", "Kappa_1")
  values <- c(tpr_0, fpr_0, tnr_0, fnr_0,
              precision_0, recall_0, f_measure_0, mcc_0, k_0,
              tpr_1, fpr_1, tnr_1, fnr_1,
              precision_1, recall_1, f_measure_1, mcc_1, k_1)
  measure_df <- data.frame(measures, values)
  return(measure_df)
}

performance_measures <- calculate_measures(tp_0, fp_0, tn_0, fn_0,
                                           tp_1, fp_1, tn_1, fn_1)
performance_measures

# Weighted Average TPR
weighted_average_tpr <- (tpr_1 + tpr_0) / 2
print(paste("Weighted Average True Positive Rate (TPR):", weighted_average_tpr))

# FPR No
fp_0 <- confusion_matrix[1, 2]
tn_0 <- confusion_matrix[2, 2]
fpr_0 <- fp_0 / (fp_0 + tn_0)
print(paste("Class No False Positive Rate (FPR):", fpr_0))

# FPR Yes
fp_1 <- confusion_matrix[2, 1]
tn_1 <- confusion_matrix[1, 1]
fpr_1 <- fp_1 / (fp_1 + tn_1)
print(paste("Class Yes False Positive Rate (FPR):", fpr_1))

# Weighted Average FPR
weighted_average_fpr <- (fpr_1 + fpr_0) / 2
print(paste("Weighted Average False Positive Rate (FPR):",
            weighted_average_fpr))

# Precision No
precision_0 <- tp_0 / (tp_0 + fp_0)
print(paste("Class No Precision:", precision_0))

# Precision Yes
precision_1 <- tp_1 / (tp_1 + fp_1)
print(paste("Class Yes Precision:", precision_1))

# Weighted Average Precision
weighted_average_precision <- (precision_1 + precision_0) / 2
print(paste("Weighted Average Precision:", weighted_average_precision))

# Recall No
recall_0 <- tp_0 / (tp_0 + fn_0)
print(paste("Class No Recall:", recall_0))

# Recall Yes
recall_1 <- tp_1 / (tp_1 + fn_1)
print(paste("Class Yes Recall:", recall_1))

# Weighted Average Recall
weighted_average_recall <- (recall_1 + recall_0) / 2
print(paste("Weighted Average Recall:", weighted_average_recall))

# F-measure No
f_measure_0 <- (2 * precision_0 * recall_0) / (precision_0 + recall_0)
print(paste("Class No F-measure:", f_measure_0))

# F-measure Yes
f_measure_1 <- (2 * precision_1 * recall_1) / (precision_1 + recall_1)
print(paste("Class Yes F-measure:", f_measure_1))

# Weighted Average F-measure
weighted_average_f_measure <- (f_measure_1 + f_measure_0) / 2
print(paste("Weighted Average F-measure:", weighted_average_f_measure))

# ROC
#roc <- roc(test$Class, test_predictions)
#roc_auc <- auc(roc)
#print(paste("ROC AUC:", roc_auc))

# MCC
#mcc <- (tp_0 * tp_1 - fp_0 * fn_1) /
#  sqrt((tp_0 + fp_0) * (tp_0 + fn_1) * (tp_1 + fn_0) * (tp_1 + fn_1))
#print(paste("Matthews Correlation Coefficient (MCC):", mcc))

# Kappa
#kappa <- (accuracy - sum(rowSums(confusion_matrix) / sum(confusion_matrix)) /
#            (1 - sum(rowSums(confusion_matrix) / sum(confusion_matrix))))
#print(paste("Kappa:", kappa))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-1-2 ***       Model 1-2 Naive Bayes ------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

model_nb <- naiveBayes(Class ~ ., data = df_balanced1_select1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-2 ***** Model 2 K-Nearest Neighbors ------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-3 ***** Model 3 Decision Tree ------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-4 ***** Model 4 Random Forest ------------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-5 ***** Model 5 Support Vect Machine -----------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#---- 5-6 ***** Model 6 Gradient Booston ---------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
