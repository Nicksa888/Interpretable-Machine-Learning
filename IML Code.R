#########################
#########################
#### Clear Workspace ####
#########################
#########################

rm(list = ls()) 
# clear global environment to remove all loaded data sets, functions and so on.

###################
###################
#### Libraries ####
###################
###################

library(easypackages) # enables the libraries function
suppressPackageStartupMessages(
  libraries(
    # Helper Packages
    "dplyr", # for data wrangling
    "ggplot2", # for data visualisation
    
    # Modelling Packages
    "h2o", # for interfacing with h2o
    "recipes", # for ML recipes
    "rsample", # for data splitting
    "xgboost", # for fitting GBMs
    
    # Model Interpretability Packages
    "pdp",  # for Partial Dependence Plots and ICE Curves
    "vip",  # for variable importance plots
    "iml", # for general iml related functions
    "DALEX", # for general iml related functions        
    "lime", # for local interpretable model-agnostic explanations
    "ggbeeswarm" # for beeswarm plots
    ))

###############################
###############################
#### Set Working Directory ####
###############################
###############################

setwd("C:/R Portfolio/Interpretable Machine Learning/Data")

bikes <- read.csv("bikes.csv")
str(bikes)
glimpse(bikes)
summary(bikes)

# Convert categorical variables into factors

bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$weekday <- as.factor(bikes$weekday)
bikes$weather <- as.factor(bikes$weather)

# Convert numeric variables into integers

bikes$temperature <- as.integer(bikes$temperature)
bikes$realfeel <- as.integer(bikes$realfeel)
bikes$windspeed <- as.integer(bikes$windspeed)

levels(bikes$season) <- c("Spring", "Summer", "Autumn", "Winter")

# remove column named date
bikes <- bikes %>% dplyr::select(-date)
str(bikes)

###############################
###############################
# Training and Test Data Sets #
###############################
###############################

set.seed(1234) # changing this alters the make up of the data set, which affects predictive outputs

split <- initial_split(bikes, 
                       strata = "rentals")

bikes_train <- training(split)
bikes_test <- testing(split)

# Ensure we have consistent categorical levels #

blueprint <- recipe(rentals ~ ., 
                    bikes_train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Create training and test data sets for h2o

Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk-11.0.12") # your own path of Java SE installed
h2o.init()

# Train set

train_h2o <- prep(blueprint, 
                  training = bikes_train, 
                  retain = T) %>%
  juice() %>%
  as.h2o()

# Test Set 

test_h2o <- prep(blueprint, 
                 training = bikes_train) %>%
  bake(new_data = bikes_test) %>%
  as.h2o()

# Set response and feature names 

Y <- "rentals"
X <- setdiff(names(bikes_train), Y)

##########################
##########################
# Create a Stacked Model #
##########################
##########################

# Train and validate a GLM Model

best_glm <- h2o.glm(
  x = X,
  y = Y,
  training_frame = train_h2o,
  alpha = 0.1,
  remove_collinear_columns = T, 
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123)

# Train and cross validate a Random Forest Model

best_rf <- h2o.randomForest(
  x = X,
  y = Y,
  training_frame = train_h2o,
  ntrees = 1000, 
  mtries = -1,
  max_depth = 30, 
  min_rows = 1,
  sample_rate = 0.8,
  nfolds = 10,
  score_each_iteration = TRUE,
  score_tree_interval = 0,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0.001
)
?h2o.randomForest
# Train and cross validate a GBM Model 

best_gbm <- h2o.gbm(
  x = X, 
  y = Y,
  training_frame = train_h2o,
  ntrees = 5000,
  learn_rate = 0.01,
  max_depth = 7,
  min_rows = 5,
  sample_rate = 0.8,
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train and Cross Validate a XGBoost Model

h2o.xgboost.available() # check if it is available. Not available on Windows machines

?h2o.xgboost
best_xgb <- h2o.xgboost(
  x = X, 
  y = Y,
  training_frame = train_h2o,
  ntrees = 5000,
  learn_rate = 0.05,
  max_depth = 3,
  min_rows = 3,
  sample_rate = 0.8,
  categorical_encoding = "Enum",
  nfolds = 10,
  fold_assignment = "Modulo",
  keep_cross_validation_predictions = T,
  seed = 123,
  stopping_rounds = 50,
  stopping_metric = "RMSE",
  stopping_tolerance = 0
)

# Train a stacked tree ensemble 

ensemble_tree <- h2o.stackedEnsemble(
  x = X,
  y = Y,
  training_frame = train_h2o,
  model_id = "my_tree_ensemble",
  base_models = list(best_glm, best_rf, best_gbm),
  metalearner_algorithm = "drf"
)

# Compute Predictions #

predictions <- predict(ensemble_tree, train_h2o) %>% as.vector()

# Print the highest and lowest rental totals

paste("Observation", which.max(predictions),
      "has a predicted rental of bikes", max(predictions))

paste("Observation", which.min(predictions),
      "has a predicted rental of bikes", min(predictions))

# Obtain feature values for observations with mim/max predicted rental total

high_ob <- as.data.frame(train_h2o)[which.max(predictions), ] %>%
  select(-rentals)
low_ob <- as.data.frame(train_h2o)[which.min(predictions), ] %>%
  select(-rentals)

################################
################################
# Create Model Agnostic Object #
################################
################################

# The advantage of the model agnostic approach is that it enables direct comparison of feature and observation importance across different models as it is not model specific.
# To create a model agnostic object, three steps are necessary:

# 1 A data frame with just the features 
# 2 A vector with actual responses (must be numeric, 0/1 for binary classification problems)
# 3 A custom function that takes the features from 1, apply the ML algorithym and return predicted values as a vector

# 1. Create data frame with just the features 

features <- as.data.frame(train_h2o) %>% dplyr::select(-rentals)

# 2. Create vector with actual responses

response <- as.data.frame(train_h2o) %>% pull(rentals)

# 3. Create custom predict function that returns the predicted values as vector

pred <- function(object, newdata) {
  results <- as.vector(h2o.predict(object, as.h2o(newdata)))
  return(results)
}

# Example of prediction output

pred(ensemble_tree, features) %>% head()

# iml model agnostic object

components_iml <- Predictor$new(
  model = ensemble_tree,
  data = features,
  y = response,
  predict.fun = pred
)

# DALEX model agnostic object

components_dalex <- DALEX::explain(
  model = ensemble_tree,
  data = features,
  y = response,
  predict.fun = pred
)

vip(
  ensemble_tree,
  train = as.data.frame(train_h2o),
  method = "permute",
  target = "rentals",
  metric = "RMSE",
  nsim = 5,
  sample_frac = 0.5,
  pred_wrapper = pred
)

# Custom prediction function wrapper

pdp_pred <- function(object, newdata) {
  results <- mean(as.vector(h2o.predict(object, as.h2o(newdata))))
  return(results)
}

# Compute partial dependence values

pd_values <- partial(
  ensemble_tree, 
  train = as.data.frame(train_h2o),
  pred.var = "temperature",
  pred.fun = pdp_pred,
  grid.resolution = 20
)

head(pd_values)

# Partial Dependence Plot

autoplot(pd_values, 
         rug = T, 
         train = as.data.frame(train_h2o))

# Individual Conditional Expectation


# Construct c-ICE Curves

partial(
  ensemble_tree, 
  train = as.data.frame(train_h2o),
  pred.var = "temperature",
  pred.fun = pred,
  grid.resolution = 20,
  plot = T,
  center = T,
  plot.engine = "ggplot2"
  
)

interact <- Interaction$new(components_iml)

interact$results %>%
  arrange(desc(.interaction)) %>%
  head()

plot(interact)

# Temperature has been identified as the variable with the strongest interaction. Now we can compute the h-statistic to identify which feature(s) it mostly interacts with

# Feature of Interest

feat <- "temperature"

interact_2way <- Interaction$new(components_iml, feature = feat)

interact_2way$results %>%
  arrange(desc(.interaction)) %>%
  top_n(10)

# Two way pdp using iml

interaction_pdp <- Partial$new(
  components_iml,
  c("temperature", "season"),
  ice = F,
  grid.size = 20
)
plot(interaction_pdp)

###################################################
###################################################
# Local Interpretable Model - Agnostic explanations
###################################################
###################################################

# create explainer object

components_lime <- lime(
  x = features,
  model = ensemble_tree,
  n_bins = 10
)

class(components_lime)
summary(components_lime)

# Use Lime to explain previously defined observations: high ob and low ob

lime_explanation <- lime::explain(
  x = rbind(high_ob, low_ob),
  explainer = components_lime,
  n_permutations = 5000,
  dist_fun = "gower",
  kernel_width = 0.25,
  n_features = 10,
  feature_select = "highest_weights"
)

glimpse(lime_explanation)

plot_features(lime_explanation, ncol = 1)

# Tuning

lime_explanation2 <- lime::explain(
  x = rbind(high_ob, low_ob),
  explainer = components_lime,
  n_permutations = 5000,
  dist_fun = "euclidean",
  kernel_width = 0.75,
  n_features = 10,
  feature_select = "lasso_path"
)

plot_features(lime_explanation2, ncol = 1)

# Shapley Values

shapley <- Shapley$new(components_iml, 
                       x.interest = high_ob,
                       sample.size = 500)

# Plot results

shapley$results %>%
  ggplot(aes(phi)) +
  geom_bar(aes(y = feature.value))
plot(shapley, xlab = "Rental Prediction Contribution") + theme_classic() 
?plot
# The predicted average rental bikes total is 3200 less than the predicted bike rental total of 7756. The plot reveals that a temperature of 65 degrees contributed 1500 towards this 3200 difference, so very nearly exactly half.

#######################################
#######################################
# XGBoost and Built in Shapley Values #
#######################################
#######################################

xgb_prep <- recipe(rentals ~., bikes_train) %>%
  step_integer(all_nominal()) %>%
  prep(training = bikes_train, retain = T) %>%
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "rentals")])
Y <- xgb_prep$rentals

set.seed(123)
bikes_xgb <- xgb.cv(
  X,
  label = Y,
  nrounds = 6000,
  objective = "reg:squarederror",
  nfold = 10, 
  params = list(
    eta = 0.01,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.5,
    colsample_bytree = 0.5),
  verbose = 0
)

# Minimum test CV RMSE

min(bikes_xgb$evaluation_log$test_rmse_mean)

# Hyper-parameter grid

hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5,
  gamma = c(1, 10, 100, 1000),
  lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  rmse = 0, # place to send RMSE results
  trees = 0 # place to send required number of trees
)

###############
# Grid Search #
###############

for(i in seq_len(nrow(hyper_grid))) { 
  set.seed(123)
  m <- xgb.cv(
    X,
    label = Y,
    nrounds = 4000,
    objective = "reg:linear",
    early_stopping_rounds = 50,
    nfold = 10, 
    verbose = 10,
    params = list(
      eta = hyper_grid$eta[i],
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i],
      lambda = hyper_grid$lambda[i],
      alpha = hyper_grid$alpha[i]
    )
    
  )
  
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
  
}

###########
# Results #
###########

hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()

##########################
# Optimal Parameter List #
##########################

params <- list(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5
)

#####################
# Train Final Model #
#####################

xgb.fit.final <- xgboost(
  params = params,
  X,
  label = Y,
  nrounds = 3944,
  objective = "reg:squarederror",
  verbose = 0
)
xgb.save(xgb.fit.final, "xgb.fit.final")
# Rescale features from low to high

feature_values <- X %>%
  as.data.frame() %>%
  mutate_all(scale) %>%
  gather(feature, feature_value) %>%
  pull(feature_value)

# Compute Shap Values and Shap based importance etc

shap_df <- xgb.fit.final %>%
  predict(newdata = X, 
          predcontrib = T) %>%
  as.data.frame() %>%
  dplyr::select(-BIAS) %>%
  gather(feature, shap_value) %>%
  mutate(feature_value = feature_values) %>%
  group_by(feature) %>%
  mutate(shap_importance = mean(abs(shap_value)))

# Shap Contribution Plot

p1 <- ggplot(shap_df,
            aes(shap_value,
                y = reorder(feature, shap_importance))) +
  ggbeeswarm::geom_quasirandom(groupOnX = F,
                               varwidth = T,
                               size = 0.4,
                               alpha = 0.25) +
  xlab("SHAP Values") +
  ylab(NULL)

# SHAP Importance Plot

p2 <- shap_df %>%
  dplyr::select(feature, shap_importance) %>%
  filter(row_number() == 1) %>%
  ggplot(aes(x = reorder(feature, shap_importance),
             y = shap_importance)) +
  geom_col() +
  coord_flip() +
  xlab(NULL) +
  ylab("mean(|SHAP Value|)")

# Combine Plots

# In the left plot, each dot indicates the contribution of each observation in a variable to the predicted bike rental total. The right plot indicates the average absolute Shapley value across all observations for each feature.

gridExtra::grid.arrange(p1, p2, nrow = 1)

# Shapley based dependency plot 

# Indicates variability in contribution across range of season and temperature values

shap_df %>%
  filter(feature %in% c("temperature", "season")) %>%
  ggplot(aes(x = feature_value, y = shap_value)) +
  geom_point(aes(color = shap_value)) +
  scale_colour_viridis_c(name = "Feature value\n(standarised", 
                         option = "C") +
  facet_wrap(~ feature, scales = "free") +
  scale_y_continuous("Shapley Value", labels = scales::comma) +
  xlab("Normalised Feature Value")
