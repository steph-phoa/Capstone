#################################################################
##                    INSTALLATION PACKAGES                    ##
#################################################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(tibble)) install.packages("tibble", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(quanteda)) install.packages("quanteda", repos = "http://cran.us.r-project.org")
if(!require(text2vec)) install.packages("text2vec", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(tidytext)) install.packages("tidytext", repos = "http://cran.us.r-project.org")
if(!require(quanteda.textplots)) install.packages("quanteda.textplots", repos = "http://cran.us.r-project.org")
if(!require(wordcloud)) install.packages("wordcloud", repos = "http://cran.us.r-project.org")
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(quanteda.textstats)) install.packages("quanteda.textstats", repos = "http://cran.us.r-project.org")
if(!require(sentimentr)) install.packages("sentimentr", repos = "http://cran.us.r-project.org")



library(tidyverse)
library(caret)
library(MASS)
library(tibble)
library(data.table)
library(dslabs)
library(dplyr)
library(quanteda)
library(text2vec)
library(stringr)
library(tidytext)
library(quanteda.textplots)
library(wordcloud)
library(tm)
library(randomForest)
library(Rborist)
library(e1071)
library(xgboost)
library(quanteda.textstats)
library(sentimentr)


##################################################################
##                        IMPORTING DATA                        ##
##################################################################

# Github for this Project:
# https://github.com/steph-phoa/Capstone/tree/main/Choose%20Your%20Own%20-%20NLP

dl <- tempfile()
download.file("https://raw.githubusercontent.com/steph-phoa/Capstone/main/Choose%20Your%20Own%20-%20NLP/siop_data_2019.csv", dl)

siop <- read.csv(dl, stringsAsFactors=F)

# If the download link doesnt work, please proceed to my github to download the file and put it in your working directory
# You may remove run the following function by removing the hashtag if you manually downloaded the file

# siop <- read.csv("siop_data_2019.csv", stringsAsFactors=F)

train <- siop %>% filter(Dataset=="Train")
test <- siop %>% filter(Dataset=="Test")
dev <- siop %>% filter(Dataset=="Dev")

names(siop)


##::::::::::::::::::::::::::::::::::::::
##  DOCUMENT TERM MATRICES - TRAIN SET  
##::::::::::::::::::::::::::::::::::::::


# tokenize and dtm all columns
# remove stopwords #      tokens_select(., pattern=stopwords("en"), selection="remove") %>%


t_all  <- lapply(c(2:6), function(x){
    t <- tokens(train[,x] ,
           remove_punct = T,
           remove_symbols = T,
           remove_numbers = T,
           remove_url = T ) %>%
        tokens_ngrams(., n=1:3) %>%
        dfm(.)
} )


# remove single occurrences and near zero variances
 
t_all <- sapply(t_all, function(x) {dfm_trim(x, min_docfreq=2)})
t_all_tfidf <-sapply(t_all, function(x) {dfm_tfidf(x)}) 
nzv_all <- sapply(t_all_tfidf, function(x) {nearZeroVar(x)})
t_all_final <- sapply(1:5, function(x) {t_all[[x]][,-nzv_all[[x]]]})


# combine dtms of all questions into a single dtm

dtm_all <- do.call(cbind, t_all_final) %>% dfm_compress


##::::::::::::::::::::::::::::::::::::
##  DOCUMENT TERM MATRICES - DEV SET  
##::::::::::::::::::::::::::::::::::::


t_all_dev  <- lapply(c(2:6), function(x){
    t <- tokens(dev[,x] ,
                remove_punct = T,
                remove_symbols = T,
                remove_numbers = T,
                remove_url = T ) %>%
        tokens_ngrams(., n=1:3) %>%
        dfm(.)
} )


t_all_dev <- sapply(t_all_dev, function(x) {dfm_trim(x, min_docfreq=2)})
t_all_tfidf_dev <-sapply(t_all_dev, function(x) {dfm_tfidf(x)}) 
nzv_all_dev <- sapply(t_all_tfidf_dev, function(x) {nearZeroVar(x)})
t_all_final_dev <- sapply(1:5, function(x) {t_all_dev[[x]][,-nzv_all_dev[[x]]]})


dtm_all_dev <- do.call(cbind, t_all_final_dev) %>% dfm_compress

# Only Keep overlapping features
t_keep <- convert(dtm_all,to="data.frame")
t_keep <- t_keep[,-1]

dev_keep <- convert(dtm_all_dev,to="data.frame")
dev_keep <- dev_keep[,-1]

# select overlapping features
keep_features <- intersect(colnames(t_keep), colnames(dev_keep))

# remove features that dont exist in each other
t_clean <- t_keep [ , keep_features , drop=FALSE] 
dev_clean <- dev_keep [ , keep_features ,drop=FALSE] 

# Double checking if our features are identical
identical(ncol(t_clean), ncol(dev_clean))
identical(colnames(t_clean),colnames(dev_clean))


##################################################################
##                       MACHINE LEARNING                       ##
##################################################################

##----------------------------------------------------------------
##                        Random Forests                        --
##----------------------------------------------------------------


library(Rborist)

# 4 hour run time

rf_train <- function(x) { 
  s <- paste0(x, "_Scale_score")
  
  trainx <- cbind(t_clean, score=train[,s])
  devx <- cbind(dev_clean, score=test[,s])
  
  surpressWarnings(set.seed(1, sample.kind="Rounding"))
  model <- train(score ~., data= trainx,  method="Rborist", ntree=500) 
  
  yhat <- predict(model, devx)
  
  cor(yhat, devx$score)
}

traits <- c("E","A","O","C","N")

rf_results <- sapply(traits, rf_train)

rf_results <- as.data.frame(rf_results) %>% t

colnames(rf_results) <- traits


##############################################################################################################################

##---------------------------------------------------------------
##           Regression (Ridge, Lasso, Elastic Net)            --
##---------------------------------------------------------------

library(glmnet)

# tune for each trait > fit > predict > correlate?

# t_list <- list(t_e,t_a,t_o,t_c,t_n)
# dev_list <- list(t_dev_e, t_dev_a, t_dev_o, t_dev_c, t_dev_n)


# loading model from 
# glmnet_results <- readRDS("glmnet_ALL")


# creating the function to tune and model all traits 
glmnet_function <- function(x){
  
  s <- paste0(x, "_Scale_score")
  
  trainx <- cbind(t_clean, score=train[,s])
  devx <- cbind(dev_clean, score=dev[,s])
  
  # tuning the alpha value using cross validation
  l <- sapply( seq(0.1,1,0.1) , function(y){
    
    suppressWarnings(set.seed(1,sample.kind="Rounding"))
    cv <- cv.glmnet(data.matrix(t_clean), data.matrix(trainx$score), type.measure="mse", 
                    alpha=y ,family="gaussian")
    
    cv$lambda.1se
    
  })
  
  a <- which.min(l)/10
  
  suppressWarnings(set.seed(1,sample.kind="Rounding"))
  fit <- glmnet(data.matrix(t_clean), data.matrix(trainx$score), alpha=a )
  
  yhat <- predict(fit, newx=data.matrix(dev_clean), s=fit$lambda.1se )
  
  h <- cor(yhat, devx$score) 
  print(c(a, h[which.max(h)]))
  
}

# Applying the tuning and training function
glmnet_results <- sapply(traits, glmnet_function)

glmnet_results <- as.data.frame(glmnet_results)

colnames(glmnet_results) <- c("E","A","O","C","N")
rownames(glmnet_results) <- c("alpha","correlation")

glmnet_results


##############################################################################################################################

##----------------------------------------------------------------
##             Extreme Gradient Boosting (XGBoost))             --
##----------------------------------------------------------------


library(xgboost)


#### Determining tuning variables

xgb_tune <- function(x){
  
  s <- paste0(x, "_Scale_score")
  
  trainx <- cbind(t_clean, score=train[,s])
  devx <- cbind(dev_clean, score=test[,s])
  
  t_data <- train_clean %>% data.matrix
  t_label <- trainx$score %>% data.matrix
  dev_data <- dev_clean %>% data.matrix
  dev_label <- devx$score %>% data.matrix
  
  xgb_train <- xgb.DMatrix(data=t_data,label=t_label)
  xgb_dev <- xgb.DMatrix(data=dev_data,label=dev_label)
  
  
  ### tuning params 
  
  
  # train control parameters
  xgb_trcontrol <- trainControl(
    method = "cv",
    number = 3,
    allowParallel = TRUE,
    verboseIter = FALSE,
    returnData = FALSE
  )
  
  
  # ggplot function for showing tuning graph
  tuneplot <- function(x, probs = .90) {
    ggplot(x) +
      coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) +
      theme_bw()
  }
  
  
  # tuning max depth, eta, 
  
  xgb_grid_1 <- expand.grid(
    nrounds = seq(50, 150, 10),
    eta= c(0.025, 0.05, 0.1, 0.3),
    max_depth=seq(2,6,1),
    gamma = 0 ,
    colsample_bytree = 1,
    min_child_weight=1,
    subsample= 0.5
  )
  
  suppressWarnings(set.seed(1,sample.kind="Rounding"))
  xgb_tune_1 <- train( score ~.,
                       data= data.matrix(trainx)  ,
                       trControl = xgb_trcontrol ,
                       tuneGrid = xgb_grid_1,
                       objective="reg:squarederror",
                       method = "xgbTree")

  tuneplot(xgb_tune_1)
  xgb_tune_1$bestTune

  # tuning min child weight


  xgb_grid_2 <- expand.grid(
    nrounds = seq(50, 150, 10),
    eta= xgb_tune_1$bestTune$eta,
    max_depth=seq(2,4,1),
    gamma = 0 ,
    colsample_bytree = 1,
    min_child_weight=seq(1,3,1),
    subsample= 0.5
  )


  suppressWarnings(set.seed(1,sample.kind="Rounding"))
  xgb_tune_2 <- train( score ~.,data= data.matrix(trainx),
                     trControl = xgb_trcontrol ,
                     tuneGrid = xgb_grid_2,
                     objective="reg:squarederror",
                     method = "xgbTree")

  tuneplot(xgb_tune_2)
  xgb_tune_2$bestTune

  # tuning column and row sampling


  xgb_grid_3 <- expand.grid(
    nrounds = seq(200, 1000, 50),
    eta= xgb_tune_1$bestTune$eta, 
    max_depth=xgb_tune_2$bestTune$max_depth, 
    gamma = 0 ,
    colsample_bytree = seq(0.4, 1, 0.2),
    min_child_weight=xgb_tune_2$bestTune$min_child_weight,
    subsample= c(0.35, 0.5, 0.75)
  )




  suppressWarnings(set.seed(1,sample.kind="Rounding"))
  xgb_tune_3 <- train(score ~., data= data.matrix(trainx) ,
                     trControl = xgb_trcontrol ,
                     tuneGrid = xgb_grid_3,
                     objective="reg:squarederror",
                     method = "xgbTree")

  tuneplot(xgb_tune_3, probs = .95)
  xgb_tune_3$bestTune


  # tuning gamma

  xgb_grid_4 <- expand.grid(
    nrounds = seq(100, 350, 50),
    eta= xgb_tune_1$bestTune$eta, #0.05
    max_depth=xgb_tune_2$bestTune$max_depth, #3
    gamma = c(0, 1, 3, 5, 7 , 10) , 
    colsample_bytree = xgb_tune_3$bestTune$colsample_bytree, # 0.8
    min_child_weight=xgb_tune_2$bestTune$min_child_weight, # 3
    subsample = xgb_tune_3$bestTune$subsample #0.5
  )

  suppressWarnings(set.seed(1,sample.kind="Rounding"))
  xgb_tune_4 <- train(score ~., data= data.matrix(trainx) ,
                     trControl = xgb_trcontrol ,
                     tuneGrid = xgb_grid_4,
                     objective="reg:squarederror",
                     method = "xgbTree")

  tuneplot(xgb_tune_4, probs = .95)
  xgb_tune_4$bestTune

# reducing the learning rate (eta)
  
  xgb_grid_5 <- expand.grid(
    nrounds = seq(100, 350, 50),
    eta= c(0.01, 0.015, 0.025, 0.05, 0.1), #0.05
    max_depth=xgb_tune_2$bestTune$max_depth, #3
    gamma = xgb_tune_4$bestTune$gamma , 
    colsample_bytree = xgb_tune_3$bestTune$colsample_bytree, # 0.8
    min_child_weight=xgb_tune_2$bestTune$min_child_weight, # 3
    subsample = xgb_tune_3$bestTune$subsample #0.5
  )

  suppressWarnings(set.seed(1,sample.kind="Rounding"))
  xgb_tune_5 <- train(score ~., data= data.matrix(trainx) ,
                     trControl = xgb_trcontrol ,
                     tuneGrid = xgb_grid_5,
                     objective="reg:squarederror",
                     method = "xgbTree")

  tuneplot(xgb_tune_5, probs = .95)
  xgb_tune_5$bestTune

}

xgb_bestTune <- sapply(traits, xgb_tune)

# saveRDS(xgb_bestTune, file="xgb_bestTune")
# xgb_bestTune <- readRDS(file="xgb_bestTune")

colnames(xgb_bestTune) <- traits

xgb_bestTune 


# training xgboost using tuned parameters
xgb_train <- function(x ,k){
  
  s <- paste0(x, "_Scale_score")
  
  trainx <- cbind(t_clean, score=train[,s])
  devx <- cbind(dev_clean, score=test[,s])
  
  t_data <- train_clean %>% data.matrix
  t_label <- trainx$score %>% data.matrix
  dev_data <- dev_clean %>% data.matrix
  dev_label <- devx$score %>% data.matrix
  
  xgb_train <- xgb.DMatrix(data=t_data,label=t_label)
  xgb_dev <- xgb.DMatrix(data=dev_data,label=dev_label)
   
  bt <- xgb_bestTune[,k]
  
  params <- list(
    booster="gbtree",
    objective="reg:squarederror",
    eta= bt[3],
    max_depth = bt[2],
    gamma= bt[4] ,
    colsample_bytree = bt[5],
    min_child_weight= bt[6],
    subsample= bt[7])
  
  # tuning max depth, eta, 
  
  xgb_model <- xgb.train(params = params,
                         data = xgb_train, 
                         watchlist = list(
                           train = xgb_train,
                           dev = xgb_dev ),
                         nrounds=300,
                         early_stopping_rounds = 50)
  
  
  xgb_p <- predict(xgb_model, newdata=xgb_dev)
  
  cor(xgb_p, dev_label)
  
}

set.seed(1,sample.kind="Rounding")
xgb_final <-  mapply(xgb_train, x=traits, k=seq(1,5,1))

# xgb_final <- c(0.26177067, 0.32932574, 0.19083101, 0.09324962, 0.18444673)

xgb_results <- as.data.frame(xgb_final) %>% t
colnames(xgb_results) <- traits

xgb_results

##############################################################################################################################

# Sentiment Analysis

# sentiment analysis (positive or negative)

tscore_list <- list(train$E_Scale_score, train$A_Scale_score, train$O_Scale_score, train$C_Scale_score, train$N_Scale_score)
dscore_list <-  list(dev$E_Scale_score, dev$A_Scale_score, dev$O_Scale_score, dev$C_Scale_score, dev$N_Scale_score)

  
senti <- apply(train[,2:6],2,function(x){sentiment_by(x)$ave_sentiment})

senti_dev <- apply(dev[,2:6],2,function(x){sentiment_by(x)$ave_sentiment})


senti_func <- function(i,j){

  senti <- data.frame(cbind(senti, Score=i))
  senti_dev <- data.frame(cbind(senti_dev, Score=j))
  
  senti_model <- train(Score~., data=senti, method="Rborist")
  senti_pred <- predict(senti_model, senti_dev)

  cor(senti_pred, j)

}

senti_cor <- mapply(senti_func, i=tscore_list, j=dscore_list)

sentiment_results <- data.frame(senti_cor) %>% t

colnames(sentiment_results) <- traits

sentiment_results

##############################

# Review models and combine them into an ensemble

ensemble <- rbind (sentiment=sentiment_results, randomforest=rf_results, glmnet=glmnet_results[2,], xgboost=xgb_results)

ensemble

best_models <- sapply(c("E", "A", "O", "C","N"), function(x){
  rownames(ensemble[ensemble[,x]==max(ensemble[,x]),])
  })

best_models

avg_cor_pre <- apply(ensemble, 2, max) %>% mean

avg_cor_pre

##############################

# Training the models on the test data

# Preprocessing

# Combine train and dev sets

train_dev <- rbind(train, dev)

# Tokenize Train set

train_final <- lapply(c(2:6), function(x){
  t <- tokens(train_dev[,x] ,
              remove_punct = T,
              remove_symbols = T,
              remove_numbers = T,
              remove_url = T ) %>%
    tokens_ngrams(., n=1:3) %>%
    dfm(.)
} )


train_final <- sapply(train_final, function(x) {dfm_trim(x, min_docfreq=2)})
train_final_tfidf <-sapply(train_final, function(x) {dfm_tfidf(x)}) 
nzv2 <- sapply(train_final_tfidf, function(x) {nearZeroVar(x)})
train_final <- sapply(1:5, function(x) {train_final[[x]][,-nzv2[[x]]]})

train_final <- do.call(cbind, train_final) %>% dfm_compress

# Tokenize Test set

test_dtm <- lapply(c(2:6), function(x){
  t <- tokens(test[,x] ,
              remove_punct = T,
              remove_symbols = T,
              remove_numbers = T,
              remove_url = T ) %>%
    tokens_ngrams(., n=1:3) %>%
    dfm(.)
} )


test_dtm <- sapply(test_dtm, function(x) {dfm_trim(x, min_docfreq=2)})
test_tfidf <-sapply(test_dtm, function(x) {dfm_tfidf(x)}) 
nzv_test <- sapply(test_tfidf, function(x) {nearZeroVar(x)})
test_final <- sapply(1:5, function(x) {test_dtm[[x]][,-nzv_test[[x]]]})

test_final <- do.call(cbind, test_dtm) %>% dfm_compress

# remove tokens that doesnt overlap in both sets
train_clean <- convert(train_final,to="data.frame")
train_clean <- train_clean[,-1]

test_clean <- convert(test_final,to="data.frame")

keeper <- intersect(colnames(train_clean), colnames(test_clean))
train_clean <- train_clean [ , keeper , drop=FALSE]  #remove features that dont exist in dev
test_clean <- test_clean [ , keeper ,drop=FALSE]  


# Writing functions for easier use
# rf tuning and training model 

rf_test <- function(x) { 
  s <- paste0(x, "_Scale_score")
  
  trainx <- cbind(train_clean, score=train_dev[,s])
  testx <- cbind(test_clean, score=test[,s])
  
  model <- train( score ~., data= trainx,  method="Rborist", ntree=500) 
  yhat <- predict(model, testx)
  cor(yhat, testx$score)
}

# xgb tuning and training model

xgb_tune_train <- function(x) {
  
  h <- paste0(x, "_Scale_score")
  
  set.seed(1,sample.kind="Rounding")
  tune <- xgb_tune(cbind(train_clean, score=train_dev[,h]), 
                   cbind(test_clean, score=test[,h]))
  
  trainx <- xgb.DMatrix(data=data.matrix(train_clean),
                        label=data.matrix(train_dev[,h]))
  testx <- xgb.DMatrix(data=data.matrix(test_clean),
                       label=data.matrix(test[,h]))
  
  set.seed(1,sample.kind="Rounding")
  model <- xgb.train(params = list(
    booster="gbtree",
    objective="reg:squarederror",
    eta= tune[3],
    max_depth = tune[2],
    gamma= tune[4] ,
    colsample_bytree = tune[5],
    min_child_weight= tune[6],
    subsample= tune[7]),
    data = trainx, 
    watchlist = list( 
      train = trainx,
      test = testx ),
    nrounds=500,
    early_stopping_rounds = 50)
  
  p <- predict(model, newdata=testx)
  cor(p, test[,h])
  
}

glmnet_test <- function(x){
  
  s <- paste0(x, "_Scale_score")
  
  trainx <- cbind(train_clean, score=train_dev[,s])
  testx <- cbind(test_clean, score=test[,s])
  
  # tuning the alpha value using cross validation
  l <- sapply( seq(0.1,1,0.1) , function(y){
    
    suppressWarnings(set.seed(1,sample.kind="Rounding"))
    cv <- cv.glmnet(data.matrix(train_clean), data.matrix(trainx$score), type.measure="mse", 
                    alpha=y ,family="gaussian")
    
    cv$lambda.1se
    
  })
  
  a <- which.min(l)/10
  
  suppressWarnings(set.seed(1,sample.kind="Rounding"))
  fit <- glmnet(data.matrix(train_clean), data.matrix(trainx$score), alpha=a )
  
  yhat <- predict(fit, newx=data.matrix(test_clean), s=fit$lambda.1se )
  
  h <- cor(yhat, testx$score) 
  print(c(a, h[which.max(h)]))}

# Training Extroversion with glmnet

E_final <- glmnet_test(x="E")

# Training Agreeableness with Rborist

A_final <- rf_test(x="A")

# Training Openness to Experience with Sentimentr

senti_td <- apply(train_dev[2:6],2,function(x){sentiment_by(x)$ave_sentiment})
senti_td <- data.frame(cbind(senti_td,  O=train_dev$O_Scale_score))

senti_test <- apply(test[2:6],2,function(x){sentiment_by(x)$ave_sentiment})
senti_test <- data.frame(cbind(senti_test,  O=test[,"O_Scale_score"]))

set.seed(1, sample.kind="Rounding")
senti_model <- train(O~., data=senti_td, method="Rborist", ntree=500)
senti_pred <- predict(senti_model, senti_test)

O_final <- cor(senti_pred, senti_test[,"O"])


# Training Concientiousness wtih Rborist

C_final <- rf_test(x="C")


# Training Neuroticism with xgboost

N_final <- xgb_tune_train(x="N")


# final correlations for each trait and average across all traits

final_cors <- data.frame( Extraversion = E_final[2],
                          Agreeableness = A_final,
                          Openness_to_Experience = O_final,
                          Conscientiousness = C_final,
                          Neuroticism = N_final)

final_cors

avg_cor <- rowMeans(final_cors)

avg_cor



############

# Extra Code
# Shall we determine if questions matter?

# Separating the Open Ended Questions from Train and Dev Set into 5 separate trainable sets

q1 <- t_all_final[[1]]
q2 <- t_all_final[[2]]
q3 <- t_all_final[[3]]
q4 <- t_all_final[[4]]
q5 <- t_all_final[[5]]

dev_q1 <- t_all_final_dev[[1]]
dev_q2 <- t_all_final_dev[[2]]
dev_q3 <- t_all_final_dev[[3]]
dev_q4 <- t_all_final_dev[[4]]
dev_q5 <- t_all_final_dev[[5]] 

# We ran a linear model on each question, for each personality trait to see 
# if the questions really help elicit certain personality traits
# Turns out, most of them didnt 

lm_function <- function(i,j) {

  t_keep <- convert(i,to="data.frame")
  t_keep <- t_keep[,-1]
  
  dev_keep <- convert(j,to="data.frame")
  dev_keep <- dev_keep[,-1]
  
  # select overlapping features
  keep_features <- intersect(colnames(t_keep), colnames(dev_keep))
  
  # remove features that dont exist in each other
  t_clean <- t_keep [ , keep_features , drop=FALSE] 
  dev_clean <- dev_keep [ , keep_features ,drop=FALSE] 
  
  
  sapply(c("E","A","O","C","N"), function(a){
    
    b <- paste0(a, "_Scale_score")
    
    train_lm <- cbind(t_clean , s = train[,b])
    dev_lm <- cbind(dev_clean, s = dev[,b])
    
    model <- lm( s ~ . , data=train_lm)
    yhat <- predict(model, dev_lm)
    cor(yhat, dev_lm$s)
  })
}

lm_cors <- mapply(lm_function, 
                  i=list(q1,q2,q3,q4,q5), 
                  j=list(dev_q1,dev_q2,dev_q3,dev_q4,dev_q5))

print(lm_cors)

beep()






























