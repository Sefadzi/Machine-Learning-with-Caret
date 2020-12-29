install.packages("mlbench")
library(mlbench)
library(caret)
library(caretEnsemble)
library(e1071)
install.packages("e1071")

data("Ionosphere")
dataset = Ionosphere
dataset <- dataset[,-2]
dataset$V1 <- as.numeric(as.character(dataset$V1))
View(head(dataset))

#Example of Boosting Algorithm
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
seed <- 7
metric <- "Accuracy"

#C5.0
set.seed(seed)
fit.c50 <- train(Class~.,data=dataset,method="C5.0",metric=metric,trControl=control)
#Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Class~.,data=dataset,method="gbm",metric=metric,trControl=control,verbose=FALSE)
#Summarize results
boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
summary(boosting_results)
dotplot(boosting_results)



#BAGGING ALGORITHMS
control <- trainControl(method = "repeatedcv",number = 10, repeats = 3)
seed <- 7
metric = "Accuracy"
#Bagged CART
set.seed(seed)
fit.treebag <- train(Class~.,data=dataset,method="treebag",metric=metric,trCOntrol=control)
#Random Forest
fit.rf <- train(Class~.,data=dataset, method="rf", metric=metric, trControl=control)
#summarize results
bagging_results<-resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)


#Stacking Algorithm

# Example of Stacking algorithms
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretList(Class~., data=dataset, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)


# Example of Stacking algorithms
# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretList(Class~., data=dataset, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)


# correlation between results
modelCor(results)
splom(results)


# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)


# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)


#sTACKING using random forest
set.seed(seed)
stack.rf <- caretStack(models, method="rf",metric="Accuracy",trControl=stackControl)
print(stack.rf)


# Model Ensemble 2
library(randomForest)
library(parallel)
library(doMC)
install.packages("doMC")

library(parallel)
library(doMC)

numCores <- detectCores()

#function to compute classification error
classification_error <- function(conf_mat){
  conf_mat = as.matrix(conf_mat)
  error = 1 - sum(diag(conf_mat))/sum(conf_mat)
  return(error)
}


#Model
rf_model <- randomForest(employmentstatus ~. -worker, data = train,importance=TRUE)
rf_pred <- as.factor(predict(rf_model, val))
rf_conf_mat <- table(true = val$employmentstatus, pred = rf_pred)

#Results
print(rf_model)

cat("\n", "RF Classification Error Rate, Validation:", classification_error(rf_conf_mat),"\n")


# Load the required libraries
library(caret)
library(nnet)
library(e1071)
library(caretEnsemble)

# Prepare a Phase 1 model, by reducing the outcome to a binary `labor_force` variable

# Create a new variable for workers
train$worker <- factor(ifelse(train$employmentstatus == "Not.in.labor.force", 0, 1))
val$worker <- factor(ifelse(val$employmentstatus == "Not.in.labor.force", 0, 1))

train$worker <- as.factor(make.names(train$worker))
val$worker <- as.factor(make.names(val$worker))

# Model to predict workers 
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid", savePredictions = "final", index = createResample(train$worker, 10), summaryFunction = twoClassSummary, classProbs = TRUE, verboseIter = TRUE)

# List of algorithms to use in ensemble
alg_list <- c("rf", "glm", "gbm", "glmboost", "nnet", "treebag", "svmLinear")

multi_mod <- caretList(worker ~ . - employmentstatus, data = train, trControl = control, methodList = alg_list, metric = "ROC")

# Results
res <- resamples(multi_mod)
summary(res)




# Stack 
stackControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, savePredictions = TRUE, classProbs = TRUE, verboseIter = TRUE)

stack <- caretStack(multi_mod, method = "rf", metric = "Accuracy", trControl = stackControl)

# Predict
stack_val_preds <- data.frame(predict(stack, val, type = "prob"))
stack_test_preds <- data.frame(predict(stack, test, type = "prob"))


# Function to find threshold

# Values
thresholds <- seq(0, 1, .05)
num_thresh <- length(thresholds)

# Empty list to store results
errors <- rep(0, num_thresh)

iter <- 1

for (i in thresholds) {
  
  cat("Calculating error for threshold value-", i, "\n")
  
  threshold_value <- i
  
  val_work_pred <- ifelse(stack_val_preds > threshold_value, "Yes", "No")
  
  conf_mat <- table(true = val$worker, pred = val_work_pred)
  
  errors[iter]<- classification_error(conf_mat) 
  
  iter <- iter + 1
}



# Compute final threshold value
result <- data.table(cbind(thresholds, errors))

final_value <- result[which(result$error == min(result$errors))]

val_worker_pred <- ifelse(stack_val_preds >= final_value$thresholds, 1, 0)

# Report error rate
phase1_conf <- table(true = val$worker, pred = val_worker_pred)

cat("Classification Error for Worker Predictions:", classification_error(phase1_conf), "\n")


# Include predictions as part of model
val$worker <- as.factor(val_worker_pred)

# Model
rf <- randomForest(employmentstatus ~ ., data = train, importance = TRUE)

# Predictions
rf_val_pred <- predict(rf, val)

# Results
rf_conf_mat <- table(true = val$employmentstatus, pred = rf_val_pred)

print(rf)

cat("\n", "Random Forest Classification Error, Validation:", classification_error(rf_conf_mat), "\n")