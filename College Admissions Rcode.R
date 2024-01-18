library(readr)
library(ggplot2)
library(patchwork)
library(caret)
library(gains)
library (rpart)
library(rpart.plot)
library(pROC)

mydata <- read_csv("College_Admissions_final.csv")
View(mydata)

#creating dummy/binary variables for potential target variables
mydata$Admitted_binary <- ifelse(mydata$Admitted == 'Yes', 1,0)
mydata$Sex_binary <- ifelse(mydata$Sex == 'M',1,0)
mydata$Enrolled_binary <- ifelse(mydata$Enrolled == 'Yes', 1,0)

#setting random seed to ensure same results in future research on code
set.seed(123)

#training dataset
train.r <- sample(row.names(mydata), dim(mydata)[1]*0.8)
train.data <- mydata[train.r,] 
dim(train.data)
View(train.data)

#validation dataset
valid.rows <- setdiff(rownames(mydata), train.r)
valid.data <- mydata[valid.rows,]
dim(valid.data)
View(valid.data)

View(mydata)

#logistic regression model, excluding enrollment and college gpa because only few students got enrolled in order to even have a gpa. 
Model1 <- glm(mydata$Admitted_binary ~ mydata$Sex_binary + mydata$HSGPA + mydata$SAT_ACT + mydata$Edu_Parent2 + mydata$Edu_Parent1 , family = binomial, data = train.data)
summary(Model1)

#prediction for validation dataset
pHat1 <- predict(Model1, valid.data, type = "response")
pHat1

#ROC 
roc_curve <- roc(mydata$Admitted_binary, pHat1)
plot(roc_curve, col = "blue", lwd = 2, legacy.axes = TRUE, grid = TRUE, main = "ROC Curve for Model1",
     xlab = "False Positive Rate", ylab = "True Positive Rate")
auc_value <- round(auc(roc_curve), 3)
legend("bottomright", legend = paste("AUC =", auc_value), col = "black", lty = 1, cex = 0.8)
abline(a = 0, b = 1, col = "red", lty = 2)

#decile-wise lift chart
df <- data.frame(Predicted_Probabilities = pHat1, Actual_Target = mydata$Admitted_binary)
df <- df[order(-df$Predicted_Probabilities), ]
total_obs <- nrow(df)
decile_size <- total_obs / 10
df$Decile <- ceiling(seq_along(df$Predicted_Probabilities) / decile_size)
df$Cumulative_Positives <- ave(df$Actual_Target, df$Decile, FUN = cumsum)
df$Decile_Lift <- df$Cumulative_Positives / (decile_size * df$Decile)
ggplot(df, aes(x = Decile, y = Decile_Lift)) +
  geom_line() +
  geom_point() +
  labs(x = "Deciles", y = "Decile-wise Lift", title = "Decile-wise Lift Chart")


#cummulative lift chart

sorted_probs <- sort(pHat1, decreasing = TRUE)
total_obs <- length(mydata$Admitted_binary)
cumulative_response <- cumsum(mydata$Admitted_binary)
cumulative_lift <- cumulative_response / total_obs
plot(seq_along(cumulative_lift) / total_obs, cumulative_lift,
     type = "l", col = "blue", lwd = 2,
     xlab = "Proportion of Cases", ylab = "Cumulative Lift",
     main = "Cumulative Lift Chart")





#seperating predicted probabilities into positive class(>=0.7) or negative class(<0.7), where positive =1 and negative =0
yHat1 <- ifelse(pHat1 >= 0.7, 1,0)
yHat1

#
sprintf("Accuracy measure for model1 = %f",100*mean(valid.data$Admitted_binary==yHat1))


#CLASSIFICATION TREE
mydata2 <- read_csv("College_Admissions_final.csv")

#convert target variable to a factor
mydata2$Admitted <- factor(mydata2$Admitted)

#confirm that Admitted is now a factor
str(mydata2)

set.seed(1)

selected_columns <- c("Admitted", "HSGPA", "SAT_ACT")

mydata2 <- mydata2[selected_columns]
#View(myData)
myIndex <- createDataPartition(mydata2$Admitted, p=0.7, list = FALSE)
trainSet <- mydata2[myIndex, ]
#View(trainSet)
validationSet <- mydata2[-myIndex, ]
#View(validationSet)

#DefaultTree
default_tree <- rpart(Admitted ~  HSGPA + SAT_ACT, data = trainSet, method = "class")
summary(default_tree)
prp(default_tree, type=1, extra=1, under = TRUE)


predictions <- predict(default_tree, newdata = validationSet, type = "class")

# Confusion matrix_defalut to evaluate performance measures
conf_matrix <- table(predictions, validationSet$Admitted)
conf_matrix

misclassification_rate <- (conf_matrix[2, 1] + conf_matrix[1, 2]) / sum(conf_matrix)
accuracy <- 1-misclassification_rate
specificity <- conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 2])
sensitivity <- conf_matrix[2, 2] / (conf_matrix[2, 1] + conf_matrix[2, 2])
precision <- conf_matrix[2, 2] / (conf_matrix[1, 2] + conf_matrix[2, 2])

# Print results_defalut
print(paste("Misclassification Rate_default:", round(misclassification_rate*100, 3), "%"))
print(paste("Accuracy_default:", round(accuracy*100,3), "%"))
print(paste("Specificity_default:", round(specificity*100,3), "%"))
print(paste("Sensitivity_default:", round(sensitivity*100,3), "%"))
print(paste("Precision_default:", round(precision*100,3), "%"))

############################################

#creating the full tree
full_tree <- rpart (Admitted ~ ., data= trainSet, method="class", cp=0, minsplit=2, minbucket=1)
prp(full_tree, type=1, extra=1, under = TRUE)
printcp(full_tree)

predictions_Full <- predict(full_tree, newdata = validationSet, type = "class")

# Confusion matrix_full to evaluate performance measures
conf_matrix_full<- table(predictions_Full, validationSet$Admitted)

misclassification_rate_full <- (conf_matrix_full[2, 1] + conf_matrix_full[1, 2]) / sum(conf_matrix_full)
accuracy_full <- 1-misclassification_rate
specificity_full <- conf_matrix_full[1, 1] / (conf_matrix_full[1, 1] + conf_matrix_full[1, 2])
sensitivity_full <- conf_matrix_full[2, 2] / (conf_matrix_full[2, 1] + conf_matrix_full[2, 2])
precision_full <- conf_matrix_full[2, 2] / (conf_matrix_full[1, 2] + conf_matrix_full[2, 2])

# Print results
print(paste("Misclassification Rate_full:", round(misclassification_rate_full*100, 3), "%"))
print(paste("Accuracy_full:", round(accuracy_full*100,3), "%"))
print(paste("Specificity_full:", round(specificity_full*100,3), "%"))
print(paste("Sensitivity_full (Recall):", round(sensitivity_full*100,3), "%"))
print(paste("Precision_full:", round(precision_full*100,3), "%"))

#creating the pruned tree
xerror_values <- full_tree$cptable[, "xerror"]
min_xerror_index <- which.min(xerror_values)
#min_xerror_index
min_cp_value <- full_tree$cptable[min_xerror_index, "CP"]
#min_cp_value

weights <- ifelse(mydata2$Admitted == "Yes", 2, 1)

# Prune the tree with weights
pruned_tree <- prune(full_tree, cp = min_cp_value, weights = weights)
prp(pruned_tree, type=1, extra=1, under = TRUE)
predictions_pruned <- predict(pruned_tree, newdata = validationSet, type = "class")

# Confusion matrix_pruned to evaluate performance measures
conf_matrix_pruned <- table(predictions_pruned, validationSet$Admitted)
conf_matrix_pruned

misclassification_rate_pruned <- (conf_matrix_pruned[2, 1] + conf_matrix_pruned[1, 2]) / sum(conf_matrix_pruned)
accuracy_pruned <- 1-misclassification_rate_pruned
specificity_pruned <- conf_matrix_pruned[1, 1] / (conf_matrix_pruned[1, 1] + conf_matrix_pruned[1, 2])
sensitivity_pruned <- conf_matrix_pruned[2, 2] / (conf_matrix_pruned[2, 1] + conf_matrix_pruned[2, 2])
precision_pruned <- conf_matrix_pruned[2, 2] / (conf_matrix_pruned[1, 2] + conf_matrix_pruned[2, 2])

# Print results
print(paste("Misclassification Rate_pruned:", round(misclassification_rate_pruned*100, 3), "%"))
print(paste("Accuracy_pruned:", round(accuracy_pruned*100,3), "%"))
print(paste("Specificity_pruned:", round(specificity_pruned*100,3), "%"))
print(paste("Sensitivity_pruned (Recall):", round(sensitivity_pruned*100,3), "%"))
print(paste("Precision_pruned:", round(precision_pruned*100,3), "%"))


#Data Visualization:

#creating stalked column chart to know how many students were admited based on parent's education level.
plot1 <- ggplot(mydata, aes(x = Edu_Parent1, fill = Admitted)) +
  geom_bar(position = "stack") +
  labs(title = "Admission Given 1st Parent's Education Level",
       x = "P1_Education Level",
       y = "Admittions") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "lightgreen"))

plot2 <- ggplot(mydata, aes(x = Edu_Parent2, fill = Admitted)) +
  geom_bar(position = "stack") +
  labs(title = "Admission Given 2nd Parent's Education Level",
       x = "P2_Education Level",
       y = "Admittions") +
  scale_fill_manual(values = c("No" = "blue", "Yes" = "lightgreen"))

plot1 + plot2


#Scaterplot for number of admits in different school.

ggplot(mydata, aes(x = College, fill = Admitted)) +
  geom_bar(position = "stack") +
  labs(title = "Admissions into Different Colleges",
       x = "College",
       y = "Number of Admissions",
       fill = "Admitted") +
  scale_fill_manual(values = c("No" = "purple", "Yes" = "darkred")) +
  theme_minimal()
