# Load necessary library
library(readr)

# Define the file path
file_path <- "/Users/walterstock3/Source/BU_CS699_Project/CS699_Provided_Artifacts/project_data.csv"

# Load the CSV file into a data frame
project_data <- read_csv(file_path)

# Display the first few rows of the data frame
head(project_data)