library(tidyverse)
library(readr)
library(here)
library(skimr)
library(dplyr)
library(data.table)
library(tidymodels)
library(doParallel)
library(knitr)
# Data wrangling and load data

url <- "https://raw.githubusercontent.com/hannahtabea/HR-analytics/8c7abc5ef610c1f7ecc4596cf0ce6f55a2ffccf1/WA_Fn-UseC_-HR-Employee-Attrition.csv"
ibm_dat <- fread(url) %>%
  # make sure that factor levels are correctly ordered to ensure correct performance metrics!!!
  # the first level should be your level of interest (e.g., YES)
  mutate(Attrition = ifelse(Attrition == "Yes", TRUE, FALSE)) 


ibm_dat$Attrition <- factor(ibm_dat$Attrition, levels = c(TRUE, FALSE))

ibm_dat[ , `:=`(MedianCompensation = median(MonthlyIncome)),by = .(JobLevel) ]
ibm_dat[ , `:=`(CompensationRatio = (MonthlyIncome/MedianCompensation)), by =. (JobLevel)]
ibm_dat[ , `:=`(CompensationLevel = factor(fcase(
  CompensationRatio %between% list(0.75,1.25), "Average",
  CompensationRatio %between% list(0, 0.75), "Below",
  CompensationRatio %between% list(1.25,2),  "Above"),
  levels = c("Below","Average","Above"))),
  by = .(JobLevel) ][, c("EmployeeCount","StandardHours","Over18") := NULL]



set.seed(69)
ibm_split <- initial_split(ibm_dat, strata = Attrition)

# Create the training data
train <- ibm_split %>%
  training()

# Create the test data
test <- ibm_split %>%
  testing()


# Cross validation
ibm_folds <- vfold_cv(train, strata = Attrition)

# Prepare for parallel processing
all_cores <- parallel::detectCores(logical = TRUE)
registerDoParallel(cores = all_cores)


# Modelling while adressing class imbalance
ibm_rec_balance <- recipe(Attrition ~ ., data = train) %>%
  # normalize all numeric predictors
  step_normalize(all_numeric()) %>%
  # create dummy variables 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  # remove zero variance predictors
  step_nzv(all_predictors(), -all_outcomes()) %>%
  # remove highly correlated vars
  step_corr(all_numeric(), threshold = 0.75) %>%
  # deal with class imbalance
  step_smote(Attrition)

# create glm-model
ibm_log_mod <- logistic_reg() %>%
  set_engine("glm")

# create XGBoost-model

xgb_model <- 
  parsnip::boost_tree(
    mode = "classification",
    trees = 1000,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) %>%
  set_engine("xgboost")

# Create workflow

ibm_wflow_bal <- workflow() %>%
  add_recipe(ibm_rec_balance)

# Create metrics
ibm_metrics <- metric_set(roc_auc, accuracy, sensitivity, specificity, precision)

# Add glm-model and fit 

glm_rs <- ibm_wflow_bal %>%
  add_model(ibm_log_mod) %>%
  fit_resamples(
    resamples = ibm_folds,
    metrics = ibm_metrics,
    control = control_resamples(save_pred = TRUE)
  )

glm_rs

# Add XGBoost model and search for best parameters

xgb_pre_tune <- ibm_wflow_bal %>%
  add_model(xgb_model)


# Grid search for hyperparameters for XGB
xgb_params <- 
  dials::parameters(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction()
  )

xgb_grid <- 
  dials::grid_max_entropy(
    xgb_params, 
    size = 200
  )

xgb_tuned <- tune::tune_grid(
  object = xgb_pre_tune,
  resamples = ibm_folds,
  grid = xgb_grid,
  metrics = ibm_metrics,
  control = tune::control_grid(verbose = TRUE)
)


# Collect best parameters and finalize model 
xgb_best_params <- xgb_tuned %>%
  tune::select_best("precision")

xgb_model_final <- xgb_model %>% 
  finalize_model(xgb_best_params)

# Fit model
xgb_rs <- ibm_wflow_bal %>%
  add_model(xgb_model_final) %>%
  fit_resamples(
    resamples = ibm_folds,
    metrics = ibm_metrics,
    control = control_resamples(save_pred = TRUE)
  )

xgb_rs

# Collect metrics and check results
collect_metrics(glm_rs)
collect_metrics(xgb_rs)

# Confusion matrixes
glm_rs %>%
  conf_mat_resampled()

xgb_rs %>%
  conf_mat_resampled()

# Plot the ROC curve for the glm-model

glm_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  roc_curve(Attrition, .pred_TRUE) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal()


# Plot the ROC curve for the XGB-model

xgb_rs %>%
  collect_predictions() %>%
  group_by(id) %>%
  roc_curve(Attrition, .pred_TRUE) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
  coord_equal()


# Make predictions on the test set
ibm_final <- ibm_wflow_bal %>%
  add_model(ibm_log_mod) %>%
  last_fit(ibm_split, metrics = ibm_metrics)

ibm_final

# Collect metrics from test predictions


collect_metrics(ibm_final)


# Look at coefficients

ibm_final %>%
  pull(.workflow) %>%
  pluck(1) %>%
  tidy(exponentiate = TRUE) %>%
  arrange(estimate)


ibm_final %>%
  pull(.workflow) %>%
  pluck(1) %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  ggplot(aes(estimate, fct_reorder(term, estimate))) +
  geom_vline(xintercept = 0, color = "gray50", lty = 2, size = 1.2) +
  geom_errorbar(aes(
    xmin = estimate - std.error,
    xmax = estimate + std.error
  ),
  width = .2, color = "gray50", alpha = 0.7
  ) +
  geom_point(size = 2, color = "#85144B") +
  labs(y = NULL, x = "Coefficent from logistic regression")
















