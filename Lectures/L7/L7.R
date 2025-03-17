library(tidyverse)
library(caret)

setwd('C:\\Courses\\C2025\\699\\Slides-Spring2025\\L7')

##########################################################################################################
## uplifting

## Original dataset was downloaded from JMP Pro (SAS) data library
## Modified for this illustration
## The dataset already has promotion column and purchase column

library(caret)

df <- read.csv('hair_care_1k.csv')
dim(df)
head(df)

df$Purchase <- factor(df$Purchase)
table(df$Promotion)
table(df$Purchase)

train.df <- df
set.seed(31)

# use random forest
# method="cv" specifies 10-fold cross-validation once
train_control <- trainControl(method="cv")
rf.grid = expand.grid(mtry = c(3, 4, 5)) 
model <- train(Purchase ~ ., data=train.df,
               trControl=train_control,
               method="rf", tuneGrid = rf.grid)
model

# calculate the uplift for new customers
uplift_df <- read.csv('hair_care_new.csv')
dim(uplift_df)
head(uplift_df)

# set promotion to yes for all customers and predict probabilities
uplift_df$Promotion <- 'Yes'
predTreatment <- predict(model, newdata=uplift_df[, -1], type="prob")
predTreatment$CID <- uplift_df$CID
predTreatment = predTreatment %>% select("CID", everything())
head(predTreatment)

# set promotion to no for all customers and predict probabilities
uplift_df$Promotion <- 'No'
predControl <- predict(model, newdata=uplift_df[, -1], type="prob")
predControl$CID <- uplift_df$CID
predControl = predControl %>% select("CID", everything())
head(predControl)

# calculate uplift
upliftResult <- data.frame(CID = predTreatment$CID,
                           probYesPromotion = predTreatment[, 3],
                           probNoPromotion = predControl[, 3],
                           uplift = predTreatment[, 3] - predControl[, 3]
)
head(upliftResult)

# select customers with uplift > 5%
send_promotion <- upliftResult[upliftResult$uplift > 0.05, ]
# sort
send_promotion <- send_promotion[order(-send_promotion$uplift), ]
send_promotion


## use KNN
model <- train(Purchase ~ ., data=train.df,
               trControl=train_control,
               tuneGrid=expand.grid(k=seq(3, 51, 2)),
               method="knn")
model

# calculate the uplift for new customers
uplift_df <- read.csv('hair_care_new.csv')
dim(uplift_df)
head(uplift_df)

# set promotion to yes for all customers and predict probabilities
uplift_df$Promotion <- 'Yes'
predTreatment <- predict(model, newdata=uplift_df[, -1], type="prob")
predTreatment$CID <- uplift_df$CID
predTreatment = predTreatment %>% select("CID", everything())
head(predTreatment)

# set promotion to no for all customers and predict probabilities
uplift_df$Promotion <- 'No'
predControl <- predict(model, newdata=uplift_df[, -1], type="prob")
predControl$CID <- uplift_df$CID
predControl = predControl %>% select("CID", everything())
head(predControl)

# calculate uplift
upliftResult <- data.frame(CID = predTreatment$CID,
                           probYesPromotion = predTreatment[, 3],
                           probNoPromotion = predControl[, 3],
                           uplift = predTreatment[, 3] - predControl[, 3]
)
head(upliftResult)

# select customers with uplift > 5%
send_promotion <- upliftResult[upliftResult$uplift > 0.05, ]
# sort
send_promotion <- send_promotion[order(-send_promotion$uplift), ]
send_promotion

#######################################################################
## association rules

library(arules)
library(arulesViz)
library(mlbench)

## small baskets 1
tdb<-read.transactions('small-baskets-1.csv', sep = ",")
summary(tdb)
colnames(tdb)
as(tdb, "matrix")
image(tdb)
inspect(tdb)
itemFrequency(tdb)

# plot the frequency of items
itemFrequencyPlot(tdb)
itemFrequencyPlot(tdb, support = 0.5)
itemFrequencyPlot(tdb, topN = 3)

# mine frequent itemsets
# default: min. support = 0.1, min. confidence = 0.8, maxlen = 10. maxtime = 5 sec
L <- apriori(tdb, parameter = list(support =0.3), target = "frequent itemsets")
summary(L)
inspect(L)
L.sorted <- sort(L, by = "support")
inspect(L.sorted)

L_df <- as(L, "data.frame")
L_df[L_df$items=="{Milk}", ]

#######################################################################
## small baskets 2

tdb<-read.transactions('small-baskets-2.csv', sep = ",")
summary(tdb)
as(tdb, "matrix")
image(tdb)
inspect(tdb)
itemFrequency(tdb)

# plot the frequency of items
itemFrequencyPlot(tdb)
itemFrequencyPlot(tdb, support = 0.5)
itemFrequencyPlot(tdb, topN = 3)

# mine frequent itemsets
L <- apriori(tdb, parameter = list(support =0.3, target = "frequent itemsets"))
summary(L)
inspect(L)
inspect(sort(L, by = "support")[1:5])
L_df <- as(L, "data.frame")
L_df[L_df$items=="{milk}", ]

# mine rules
# apriori (in arules package) only generates rules with one item in RHS
rules <- apriori(tdb, parameter = list(support =0.3, 
                                       confidence = 0.2, minlen = 2),
                 target = "rules")
summary(rules)
inspect(rules)

# sorting rules by confidence
inspect(sort(rules, by = "confidence"))

# finding subsets of rules containing any beer items
beerrules <- subset(rules, items %in% "beer")
inspect(beerrules)

# write rules to a CSV file
write(rules, file = "small-baskets-2-rules.csv",
      sep = ",", quote = TRUE, row.names = FALSE)

# convert rules to data frame
rules_df <- as(rules, "data.frame")

# add cosine, chi-square, and Kulczynski measures
rules_df$chiSquared <- interestMeasure(rules, "chiSquared")
rules_df$cosine <- interestMeasure(rules, "cosine")
rules_df$kulczynski <- interestMeasure(rules, "kulczynski")
sapply(rules_df, class)
head(rules_df)
rules_df
corrMatrix <- cor(rules_df[c(5,8,9)])
corrMatrix

##########################################################################
## L7 exercise

tdb<-read.transactions('L7-ex.csv', sep = ",")
summary(tdb)
inspect(tdb)
itemFrequency(tdb)

# plot the frequency of items
itemFrequencyPlot(tdb)
itemFrequencyPlot(tdb, topN = 5)

# a visualization of the sparse matrix for the first five transactions
image(tdb[1:5])

# mine frequent itemsets
L <- apriori(tdb, parameter = list(support =0.3), target = "frequent itemsets")
summary(L)
inspect(L)
inspect(sort(L, by = "support")[1:5])
L_df <- as(L, "data.frame")
L_df[L_df$items=="{5}", ]

# mine rules
rules <- apriori(tdb, parameter = list(support =0.3, 
                                       confidence = 0.8, minlen = 2),
                 target = "rules")
summary(rules)
inspect(rules)

# sorting rules by confidence
inspect(sort(rules, by = "confidence")[1:5])

# finding subsets of rules containing any berry items
beerrules <- subset(rules, items %in% "5")
inspect(beerrules)

# write rules to a CSV file
write(rules, file = "L7-ex-rules.csv",
      sep = ",", quote = TRUE, row.names = FALSE)

# convert rules to data frame
rules_df <- as(rules, "data.frame")
# add cosine, chi-square, and Kulczynski measures
rules_df$chiSquared <- interestMeasure(rules, "chiSquared")
rules_df$cosine <- interestMeasure(rules, "cosine")
rules_df$kulczynski <- interestMeasure(rules, "kulczynski")
rules_df
corrMatrix <- cor(rules_df[c(5,8,9)])
corrMatrix

##########################################################################
# groceries
# dataset is from arules package

tdb<-read.transactions('groceries.csv', sep = ",")
tdb
summary(tdb)

L <- apriori(tdb, parameter = list(support =0.01), target = "frequent itemsets")
summary(L)
inspect(L)
L.sorted <- sort(L, by = "support")
inspect(L.sorted)

rules <- apriori(tdb, parameter = list(support =0.01, 
                                       confidence = 0.2, minlen = 2),
                 target = "rules")
summary(rules)
inspect(sort(rules, by = "confidence")[1:10])
inspect(sort(rules, decreasing = FALSE, by = "confidence")[1:5])
inspect(sort(rules, by = "support")[1:5])
inspect(sort(rules, decreasing = FALSE, by = "support")[1:5])

# convert rules to data frame
rules_df <- as(rules, "data.frame")
# add cosine, chi-square, and Kulczynski measures
rules_df$chiSquared <- interestMeasure(rules, "chiSquared")
rules_df$cosine <- interestMeasure(rules, "cosine")
rules_df$kulczynski <- interestMeasure(rules, "kulczynski")
measures <- rules_df[, -c(2,4,5,6)]
sorted_by_confidence <- measures[order(-measures$confidence), ]
sorted_by_confidence[1:20, ]

# plot scatter plot: support vs. confidence
plot(rules)


