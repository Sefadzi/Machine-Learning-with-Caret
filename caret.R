install.packages(c("PerformanceAnalytics","corrplot","car","doParallel","ggthemes","psych"))
library(ggthemes)
library(tidyverse)
library(PerformanceAnalytics)
library(ggthemes)
library(corrplot)
library(car)
library(psych)
library(caret)
library(caretEnsemble)
library(doParallel)

df = read.csv("houses1990.csv",stringsAsFactors = FALSE)

glimpse(df)

summary(df$houseValue)
View(df[,1:7])
corrplot(cor(df[,1:7]), method = "square")
chart.Correlation(df)


sapply(df, {function(x) any(is.na(x))})

#Outlier detection
View(df[,-c(8:9)])
boxplot(df[,-c(8:9)], col = "orange", main = "Features Boxplot")

boxplot(df$houseValue, col="red")

houseValue_outliers <- which(df$HouseValue > (mean(df$houseValue)+3*sqrt(var(df$houseValue))))
df[age_outliers, "houseValue"]


# Check for Multicollinearity
simple_lm <- lm(houseValue ~ .,data=df[,1:7])
vif(simple_lm)



par(mfrow = c(2,2))
hist(concrete$age)
hist(concrete$superplastic)
hist(log(concrete$age), col = "red")
hist(log(concrete$superplastic), col = "red")

concrete$age <- log(concrete$age)
concrete$superplastic <- log(concrete$superplastic)
concrete$superplastic <- ifelse(concrete$superplastic == -Inf, 0,  
                                concrete$superplastic)
head(concrete)


#shuffle data
set.seed(123)
housevalue_rand = df[sample(1:nrow(df)),]


X = housevalue_rand[, -1]
y = housevalue_rand[,1]

set.seed(123)
part.index <- createDataPartition(housevalue_rand$houseValue,p = 0.75,list = FALSE)


X_train <- X[part.index, ]
X_test <- X[-part.index, ]
y_train <- y[part.index]
y_test <- y[-part.index]

str(X_train)
str(X_test)
str(y_train)
str(y_test)


registerDoParallel(2)
getDoParWorkers()

set.seed(123)
my_control <- trainControl(method = "cv",
                           number = 5,
                           savePredictions = "final",
                           allowParallel = TRUE)


set.seed(222)
model_list <- caretList(X_train,
                        y_train,
                        trControl = my_control,
                        methodList = c("lm","svmRadial","rf","xgbTree","xgbLinear"),
                        tuneList = NULL,
                        continue_on_fail = FALSE,
                        preProcess = c("center","scale"))



# Model Accuracy
options(digits = 3)

model_results <- data.frame(
  LM = min(model_list$lm$results$RMSE),
  SVM = min(model_list$svmRadial$results$RMSE),
  RF = min(model_list$rf$results$RMSE),
  XGBT = min(model_list$xgbTree$results$RMSE),
  XGBL = min(model_list$xgbLinear$results$RMSE)
)
print(model_results)


# checking accuracy
resamples <- resamples(model_list)
dotplot(resamples, metric="RMSE")


#Correlation between models for ensemble
modelCor(resamples)

#Ensemble One
set.seed(222)
ensemble_1 <- caretEnsemble(model_list,
                            metric = "RMSE",
                            trControl = my_control)

plot(ensemble_1)


#Ensemble Two
set.seed(222)
ensemble_2 <- caretStack(model_list,
                         method="glmnet",
                         metric="RMSE",
                         trControl=my_control)

ensemble_2


# PREDICTIONS
pred_lm <- predict.train(model_list$lm, newdata = X_test)
pred_svm <- predict.train(model_list$svmRadial, newdata = X_test)
pred_rf <- predict.train(model_list$rf, newdata = X_test)
pred_xgbT <- predict.train(model_list$xgbTree, newdata = X_test)
pred_xgbL <- predict.train(model_list$xgbLinear, newdata = X_test)
predict_ens1 <- predict(ensemble_1, newdata = X_test)
predict_ens2 <- predict(ensemble_2, newdata = X_test)

#RMSE
pred_RMSE <- data.frame(ensemble_1 = RMSE(predict_ens1, y_test),
                        ensemble_2 = RMSE(predict_ens2, y_test),
                        LM = RMSE(pred_lm, y_test),
                        SVM = RMSE(pred_svm, y_test),
                        RF = RMSE(pred_rf, y_test),
                        XGBT = RMSE(pred_xgbT, y_test),
                        XGBL = RMSE(pred_xgbL, y_test))





set.seed(123)
xgbTree_model <- train(X_train,
                       y_train,
                       trControl = my_control,
                       method = "xgbLinear",
                       metric = "RMSE",
                       preProcess = c("center","scale"),
                       importance = TRUE)

plot(varImp(xgbTree_model))


#CARET ENSEMBLE
