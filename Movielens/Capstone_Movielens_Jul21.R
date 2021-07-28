###########################################################
########## HARVARDX CAPSTONE: MOVIELENS PROJECT ###########
#################### Stephanie Phoa #######################

###################### START HERE #########################

# First things first: 
# It is recommended to clear your workspace 
# to avoid any interference with the code 
save.image()
rm(list = ls())

################ LOAD REQUIRED PACKAGES ###################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")


library(lubridate)
library(recosystem)
library(matrixStats)
library(stringr)
library(tidyverse)
library(caret)
library(data.table)

###########################################################
#             START OF PROVIDED PROJECT CODE              #
# Create edx set, validation set (final hold-out test set)#
###########################################################

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###########################################################
#                  END OF PROVIDED CODE                   #
###########################################################



##################### DATA WRANGLING ######################


# Splitting edx into train and test sets

set.seed(1, sample.kind="Rounding")
i <- createDataPartition(edx$rating, times=1, p=0.1, list=FALSE)
train <- edx[-i,]
temp <- edx[i,]

test <- temp %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

removed <- anti_join(temp, test)
train <- rbind(train, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Creating a Movie-Genre Matrix

genres  <- train %>% select(movieId, genres) %>% group_by(movieId, genres)

genres <- separate(genres, genres, c("genre_1", "genre_2", "genre_3", "genre_4", "genre_5", "genre_6",  "genre_7",  "genre_8",  "genre_9"), "\\|")

genre_types <- sapply(2:10, function(x){
  d <- distinct(genres[,x])
})

genre_types <- unique(unlist(genre_types))[-21]

genres <- sapply(1:20, function(x){
  g<-genres[2:10]==genre_types[x]
  rowSums(g, na.rm = T)
})

colnames(genres) <- genre_types


# Determining Time factors 

time <- train  %>%   select(movieId, userId, timestamp, rating) %>% 
  mutate(rating_year=year(as_datetime(timestamp))) # pulling movie release year from title string 

release_year <- train %>% select(movieId, title) %>% 
  mutate(release_year=str_extract(train$title, "(\\d{4})")) %>% 
  select(movieId, release_year) %>% 
  distinct()

time <- left_join(time, release_year, by="movieId")



################ EXPLORATORY DATA ANALYSIS #################

# Rating Barplot

# ratings distribution
ggplot(train, aes(rating)) + 
  geom_bar() 

# user ratings distribution
train %>% group_by(userId) %>% summarize(n=n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) +
  scale_x_log10() +
  xlab("userID")

# movie ratings distribution
train %>% group_by(movieId) %>% summarize(n=n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30) +
  scale_x_log10() +
  xlab("movieID")

gc() 

# genre prevelance barplot
colSums(genres) %>% sort() %>% barplot(., las=2, cex.names=0.9)


########################section_end########################## 

###################  RMSE FUNCTION ###########################

# Root Mean Squared Error (RMSE) will be used to determine prediction accuracy
# Project Goal : RMSE < 0.86490 


RMSE <- function(trueRatings, predictedRatings){
  sqrt(mean((trueRatings - predictedRatings)^2, na.rm = TRUE))}


##################### NAIVE BASELINE ##########################

mu <- mean(train$rating)

naive_rmse <- RMSE(test$rating, mu)
naive_rmse

################### BASELINE REGRESSION ######################

# basic linear regression model

# movie effect: bi

base_bi <- train %>% group_by(movieId) %>%
  summarize(bi = mean(rating - mu)) 

base_yhat_bi <- test %>% left_join(base_bi, by="movieId") %>%
  mutate(yhat=mu+bi) %>% pull(yhat)

bi_rmse<- RMSE(test$rating, base_yhat_bi) #rmse bi only

bi_rmse

# user effect: bu

base_bu <- train %>%
  left_join(base_bi, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = mean(rating - mu- bi ))

base_yhat_bu <- test %>% left_join(base_bi, by="movieId") %>%
  mutate(yhat=mu+bi) %>% pull(yhat)
bu_rmse<- RMSE(test$rating, base_yhat_bu) #rmse bu only

bu_rmse

# Baseline regression RMSE

base_yhat <- test %>%
  left_join(base_bi, by="movieId") %>%
  left_join(base_bu, by="userId") %>%
  mutate(yhat=mu+bi+bu) %>% pull(yhat)

baseline_rmse <- RMSE(test$rating, base_yhat) #rmse bi+bu regression

baseline_rmse


#################### REGULARIZATION ##########################

# determining lambda for lowest RMSE
# Processing time ~ 10 min

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(x){
  
  base_bi <- train %>% group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(n()+x))  %>% suppressMessages()
  
  base_bu <- train %>% 
    left_join(base_bi, by="movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - mu- bi )/(n()+x))  %>% suppressMessages()
  
  base_bt <- train %>% 
    left_join(base_bi, "movieId") %>%
    left_join(base_bu, "userId") %>%
    left_join(release_year, "movieId") %>%
    mutate(bt=sum(rating - mu- bi- bu)/(n()+x), n_t = n()) %>% 
    group_by(release_year) %>%
    summarize(bt=mean(rating-mu-bi-bu))  %>% suppressMessages()
  
  base_bg <- train %>% 
    left_join(release_year, by="movieId") %>%
    left_join(base_bi, "movieId") %>%
    left_join(base_bu, "userId") %>%
    left_join(base_bt, "release_year") %>%
    mutate(bg=sum(rating - mu- bi - bu - bt  )/(n()+x), n_g = n()) %>% 
    group_by(genres) %>% 
    summarize(bg=mean(rating-mu-bi-bu-bt))  %>% suppressMessages()
  
  y_hat <- test %>% 
    left_join(release_year, by="movieId") %>%
    left_join(base_bi, by="movieId") %>%
    left_join(base_bu, by="userId") %>%
    left_join(base_bt, by="release_year") %>%
    left_join(base_bg, by="genres") %>%
    mutate(yhat=mu+bi+bu+bt+bg) %>% pull(yhat)  %>% suppressMessages()
  
  return(RMSE(test$rating, y_hat))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

lambda <- lambdas[which.min(rmses)]


########################section_end########################## 



##########  BASELINE REGRESSION W/ REGULARIZATION ############

# bi regularized

train_bi <- train %>% group_by(movieId) %>%
  summarize(bi = sum(rating - mu)/(n()+lambda), n_i = n()) 

yhat_bi <- test %>% left_join(train_bi, by="movieId") %>%
  mutate(yhat=mu+bi) %>% pull(yhat)
reg_bi<- RMSE(test$rating, yhat_bi)

# bu regularized

train_bu <- train %>% 
  left_join(train_bi, by="movieId") %>%
  group_by(userId) %>%
  summarize(bu = sum(rating - mu- bi )/(n()+lambda), n_u = n())

yhat_bu <- test %>% left_join(train_bu, by="userId") %>%
  mutate(yhat=mu+bu) %>% pull(yhat)
reg_bu <- RMSE(test$rating, yhat_bu)

# bi + bu regularized

yhat_bi_bu <- test %>% 
  left_join(train_bi, by="movieId") %>%
  left_join(train_bu, by="userId") %>%
  mutate(yhat=mu+bi+bu) %>% pull(yhat)
reg_bi_bu <- RMSE(test$rating, yhat_bi_bu)

#################### TIME & GENRE EFFECT #####################

# time effect: bt
# mu + bi + bu + bt

train_bt <- train %>% 
  left_join(train_bi, "movieId") %>%
  left_join(train_bu, "userId") %>%
  left_join(release_year, "movieId") %>%
  mutate(bt=sum(rating - mu- bi- bu)/(n()+lambda), n_t = n()) %>% # with regularization
  group_by(release_year) %>%
  summarize(bt=mean(rating-mu-bi-bu))

yhat_bt <-test %>% 
  left_join(release_year, by="movieId") %>%
  left_join(train_bi, by="movieId") %>%
  left_join(train_bu, by="userId") %>% 
  left_join(train_bt, by="release_year") %>%
  mutate(yhat=mu+bi+bu+bt)%>% pull(yhat)

reg_bi_bu_bt  <- RMSE(test$rating, yhat_bt)
reg_bi_bu_bt 


# genre effect: bg
# mu + bi + bu + bt + bg

train_bg <- train %>% 
  left_join(release_year, by="movieId") %>%
  left_join(train_bi, "movieId") %>%
  left_join(train_bu, "userId") %>%
  left_join(train_bt, "release_year") %>%
  mutate(bg=sum(rating - mu- bi - bu - bt  )/(n()+lambda), n_g = n()) %>%  # with regularization
  group_by(genres) %>% 
  summarize(bg=mean(rating-mu-bi-bu-bt))


yhat_bg <-test %>% 
  left_join(release_year, by="movieId") %>%
  left_join(train_bi, by="movieId") %>%
  left_join(train_bu, by="userId") %>% 
  left_join(train_bt, by="release_year") %>%
  left_join(train_bg, by="genres") %>%
  mutate(yhat=mu+bi+bu+bt+bg)%>% pull(yhat)

reg_bi_bu_bt_bg <- RMSE(test$rating, yhat_bg)
reg_bi_bu_bt_bg 

########################section_end########################## 




################### REGRESSION RMSE TABLE ###################


rmse_baseline <-  tibble(Model = c("Naive - Just the Average",
                                   "Movie Effect",
                                   "User Effect",
                                   "Movie+User Effect",
                                   "Movie+User+Time+Genre Effect"),
                         RMSE = c(naive_rmse, bi_rmse, bu_rmse, baseline_rmse, "NA"), 
                         Reg_RMSE = c("NA", reg_bi, reg_bu, reg_bi_bu_bt, reg_bi_bu_bt_bg ) )

rmse_baseline %>% knitr::kable()

########################section_end########################## 




################### MATRIX FACTORIZATION ####################

# !WARNING! Processing time ~ 20 mins #

# recosystem package is a wrapper of the LIBMF library, a tool for sparse matrix factorization
# uses parallel computation for multicore cpu to speed up matrix factorization big time

library(recosystem)

r <- Reco()

train_mf <- train %>% select(userId,movieId, rating) %>% as.matrix()
test_mf <- test %>% select(userId,movieId, rating) %>% as.matrix()

write.table(train_mf, file="train_mf.txt", sep=" ", row.names=F, col.names=F)
write.table(test_mf, file="test_mf.txt", sep=" ", row.names=F, col.names=F)

set.seed(1, sample.kind="Rounding")
train_mf<- data_file("train_mf.txt", index1=TRUE)
test_mf<- data_file("test_mf.txt", index1=TRUE)

# tuning parameters
tune <- r$tune(train_mf, opts=list(dim=c(10,20,30), lrate=c(0.1,0.2), costp_l1=0, costq_l1=0, nthread=1, niter=10))
tune$min

# training and prediction

set.seed(1, sample.kind="Rounding")
fit_mf <- suppressWarnings(r$train(train_mf, opts= c(tune$min,nthread=1, niter=20)))

pred_file <- tempfile()
r$predict(test_mf, out_file(pred_file))

yhat_mf<- print(scan(pred_file))

# RMSE matrix factorization
mf_rmse <- RMSE(test$rating, yhat_mf)
mf_rmse

#########################section_end###########################  





############ FINAL RMSE TESTING ##############

# final testing with validation set


library(recosystem)

r <- Reco()

train_mf <- train %>% select(userId,movieId, rating) %>% as.matrix()
valid_mf <- validation  %>% select(userId,movieId, rating) %>% as.matrix()

write.table(train_mf, file="train_mf.txt", sep=" ", row.names=F, col.names=F)
write.table(valid_mf, file="valid_mf.txt", sep=" ", row.names=F, col.names=F)

set.seed(1, sample.kind="Rounding")
train_mf<- data_file("train_mf.txt", index1=TRUE)
valid_mf<- data_file("valid_mf.txt", index1=TRUE)

# tuning parameters
tune <- r$tune(train_mf, opts=list(dim=c(10,20,30), lrate=c(0.1,0.2), costp_l1=0, costq_l1=0, nthread=1, niter=10))
tune$min

# training and prediction

set.seed(1, sample.kind="Rounding")
fit_mf <- suppressWarnings(r$train(train_mf, opts= c(tune$min,nthread=1, niter=20)))

pred_file <- tempfile()
r$predict(valid_mf, out_file(pred_file))

yhat_mf_final<- print(scan(pred_file))

# RMSE matrix factorization
mf_rmse_final <- RMSE(validation$rating, yhat_mf_final)

#########################section_end###########################  



######################### FINAL RMSE ###########################

print(mf_rmse_final)

#########################section_end############################  




# just a little tool to let you know when the code has finished running 


install.packages("beepr")
library(beepr)
beep("fanfare")












