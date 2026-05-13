library(readxl)
library(tidymodels)
library(ranger)
library(matrixStats)

set.seed(17)  
theme_set(theme_classic())
# Load experiments
training <- read_excel("MagmaTAB.xlsx", sheet="workbook")
data <- read_excel("MagmaTAB.xlsx", sheet="Askja_glass")

# Melt / MeltMin
model_select <- "MeltMin"
NumbMin   <- 2
filter    <- 0.7
scalefac  <- 2

training <- subset(training,number_phases>=NumbMin)

colnames(training) <- make.names(colnames(training))
colnames(data) <- make.names(colnames(data))


if (model_select == "MeltMin"){
  y.data_train <- training[,c("P_kbar","ol","opx","cpx","plag","amph","ox","bt","ksp","gt","qz","SiO2.n.","TiO2.n.","Al2O3.n.","FeO.n.","MgO.n.","CaO.n.","Na2O.n.","K2O.n.")]
  y.data_test <- data[,c("ol","opx","cpx","plag","amph","ox","bt","ksp","gt","qz","SiO2.n.","TiO2.n.","Al2O3.n.","FeO.n.","MgO.n.","CaO.n.","Na2O.n.","K2O.n.")]
}  else{
  y.data_train <- training[,c("P_kbar","ol","opx","cpx","plag","amph","ox","bt","ksp","gt","qz","SiO2.n.","TiO2.n.","Al2O3.n.","FeO.n.","MgO.n.","CaO.n.","Na2O.n.","K2O.n.")]
  y.data_test <- data[,c("ol","opx","cpx","plag","amph","ox","bt","ksp","gt","qz","SiO2.n.","TiO2.n.","Al2O3.n.","FeO.n.","MgO.n.","CaO.n.","Na2O.n.","K2O.n.")]
}

y_train <- y.data_train$P_kbar

model1 <- ranger(
  formula =  P_kbar ~. ,
  data = y.data_train,
  num.trees = 500,
  mtry = NULL,
  importance = "none",
  min.node.size = NULL,
  max.depth = NULL,
  replace = FALSE,
  sample.fraction = 1,
  num.random.splits = 10,
  splitrule = "extratrees",
  keep.inbag = TRUE)

#Make predictions using the first model for the test data
initial_predictions <- predict(model1, data = y.data_test)$predictions
initial_predictions_votes <- predict(model1, data = y.data_test,type='response',predict.all = TRUE)
initial_predictions_SD <- rowSds(initial_predictions_votes$predictions)

# Compute residuals for the training data using the first model
train_predictions <-  predict(model1, data = y.data_train)$predictions
train_residuals <- y_train -  predict(model1, data = y.data_train)$predictions

# Add the residuals as a new column in the training data
train_data_with_residuals <- cbind(y.data_train, Residuals = train_residuals)

#train_data_with_residuals <- train_data_with_residuals[-1]
if(filter>0){
  threshold <- as.numeric(quantile(initial_predictions_SD,probs=filter))  
}else{
  threshold=0
}


model2 <- ranger(
  formula = Residuals~.,
  data = train_data_with_residuals,
  num.trees = 500,
  mtry = NULL,
  importance = "none",
  min.node.size = NULL,
  max.depth = NULL,
  replace = FALSE,
  sample.fraction = 1,
  num.random.splits = 10,
  splitrule = "extratrees",
  keep.inbag = TRUE)


P_kbar <- initial_predictions
y.data_t <- cbind(y.data_test,P_kbar)

# Make predictions using the second model for the test data
residual_predictions <- predict(model2, data = y.data_t)$predictions
residual_predictions_votes <- predict(model2, data = y.data_t,type='response',predict.all = TRUE)

# Correct the initial predictions by adding the predicted residuals
predictions <- initial_predictions + scalefac*residual_predictions
df_predictions <- data.frame(predictions,data) 
df_predictions_filtered <- subset(df_predictions, initial_predictions_SD <= threshold)


