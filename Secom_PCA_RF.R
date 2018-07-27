
set.seed(123)

library(yardstick)
library(tidyverse)
library(methods)
library(aider)

follow_missing_patterns <- function(x){
  
  name_old <- names(x)
  x$vec_imputed <- ifelse(is.na(x[[1]]), 1, 0)
  names(x)[2] <- paste0("missing_", name_old)
  x
}

# Benchmark ---------------------------------------------------------------

# df_x <- read_delim("http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data", " ", col_names = FALSE)
# df_y <- read_delim("http://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data", " ", col_names = c("target", "target_date"))

# saveRDS(df_all, "df_all.Rds")
df_all <- readRDS("df_all.Rds")

df_all %<>%
  mutate(
    target = as.factor(ifelse(target == -1, "Success", "Failure")),
    target_time = ifelse(str_sub(target_date, 12, 13) == "15", "eq15", "neq15"),
    target_date = str_c("month_", str_sub(target_date, 4, 5))
  ) %>% 
  select(-target_time, -target_date)

df_miss <- naniar::miss_var_summary(df_all) %>% 
  filter(percent <= 50)

df_all %<>% 
  lmap_at(
    naniar::miss_var_summary(df_all) %>% filter(percent > 0) %>% .$variable,
    follow_missing_patterns)

# split <- createDataPartition(df_all$target, 1, 0.9, list = FALSE)
# saveRDS(split, "split.Rds")
split <- readRDS("split.Rds")

df_all_train <- df_all[split, ]
df_all_test  <- df_all[-split, ]

# split_cv <- createFolds(df_all_train$target, k = 5, list = TRUE)
# saveRDS(split_cv, "split_cv.Rds")
split_cv <- readRDS("split_cv.Rds")

df_imp <- df_all_train %>% 
  select(target, one_of(df_miss$variable)) %>% 
  calculate_importance(target)

df_imp_sel <- df_imp %>% 
  filter(imp >= 0.525)

recipe <- df_all_train %>% 
  select(target, one_of(df_imp_sel$variable)) %>%
  recipe(target ~ .) %>% 
  step_meanimpute(all_numeric(), -contains("missing")) %>% 
  step_center(all_numeric(), -contains("missing"), -target) %>%
  step_scale(all_numeric(), -contains("missing"), -target) %>% 
  step_pca(all_numeric(), threshold = .70)

recipe_prep <- prep(recipe)
recipe_bake <- prep(recipe, retain = TRUE) %>% 
  juice()

ctrl <- trainControl(
  method = "cv",
  index = split_cv,
  classProbs = TRUE,
  verboseIter = TRUE,
  summaryFunction = prSummary,
  savePredictions = "final",
  search = "grid",
  sampling = "smote",
  seeds = list(
    c(1, 2, 3, 4, 5, 6), 
    c(2, 3, 4, 5, 6, 7), 
    c(3, 4, 5, 6, 7, 8), 
    c(4, 5, 6, 7, 8, 9), 
    c(5, 6, 7, 8, 9, 10), 
    c(6)
    )
)

model_rf <- train(
  target ~ .,
  data = recipe_bake,
  method = "ranger",
  trControl = ctrl,
  num.trees = c(250)
)

model <- model_rf

dinner <- df_all_test %>% 
  select(target, one_of(df_imp_sel$variable)) %>% 
  bake(recipe_prep, .)

df_prd_raw <- predict.train(
  model, 
  dinner,
  type = "raw"
  )

df_prd_prb <- predict.train(
  model, 
  dinner,
  type = "prob"
  )

supper <- bind_cols(dinner["target"], as_data_frame(df_prd_raw), df_prd_prb["Failure"]) %>% 
  rename(truth = target, prediction = value, estimate = Failure) %>% 
  mutate(prediction = as.factor(prediction))

plot_density(supper, estimate, truth, quantile_low = 0, quantile_high = 1)
plot_boxplot(supper, truth, estimate, truth, quantile_low = 0, quantile_high = 1)

confusionMatrix.train(model)
confusionMatrix(supper$prediction, supper$truth, "Failure", mode = "everything")
