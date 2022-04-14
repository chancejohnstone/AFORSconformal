#conformal inference classification tutorial using CogPilot data

#load necessary R packages
#make sure you install these packages as well
#e.g., install.packages("ks")
library(ks)
library(knitr)
library(MASS)
library(ranger)
library(readr)
library(dplyr)
library(reshape2)
library(pdp)
library(rgl)
library(adabag)
library(dplyr)
library(reshape2)

rotate <- function(x) t(apply(x, 2, rev))

#####DATA#####
#read in all covariate data
data_all <- read.csv("score_val_allModalities.csv")

#read in true classes
perf <- read.csv("PerfMetrics.csv")

#explore data
str(data_all)
str(perf)

#align responses with covariate data
sub <- parse_number(substr(data_all$Subject.Run,start = 7, stop = 9))
run <- parse_number(substr(data_all$Subject.Run,start = 15, stop = 17))
#keep <- data.frame(Subject = sub, Run = run, keep = 1)
keep <- data.frame(Subject = sub, Run = run)

#drop regression responses
#keep only covariates aligned with observations for which we have difficulty  responses
keep_check <- left_join(keep, perf[,-c(2,5,6,7,8)], by = c("Subject", "Run"))

#get "kept" features from CogPilot model
xtrain <- read.csv("xtrain.csv")
sub <- names(xtrain)
sub_keep <- sub[(sub %in% names(data_all))]
sub_keep

#keep selected features
data_sub <- data_all %>% select(sub_keep)

#number of observations
n <- nrow(keep_check)
#####DATA#####

#####TRAIN#####
#set random seed
set.seed(2022)
all_reps_coverage <- c()

#generate B train/test splits 
#B <- 25

#takes around 3 mins to run this once on my machine
B <- 1
for(b in 1:B){
  #select five observations for test set
  train_id <- sample(1:n, 161)
  
  #select 20 observations for test set
  #train_id <- sample(1:n, 146)
  
  #generate train and test sets
  xtrain <- data_sub[train_id,]
  ytrain <- keep_check$Difficulty[train_id]
  xtest <- data_sub[-train_id,]
  ytest <- keep_check$Difficulty[-train_id]
  
  #combine response of interest with kept features
  train <- data.frame(resp = as.factor(ytrain), xtrain)
  test <- data.frame(resp = as.factor(ytest), xtest)
  
  #fit adaboost model on all kept features
  ada <- boosting(resp~., data = train, mfinal = 35)
  class <- apply(ada$prob, FUN = which.max, MARGIN = 1)
  
  pival <- matrix(0, nrow = length(ytest), ncol = 4)
  for(i in 1:length(ytest)){
    for(j in 1:4){
      cand <- as.factor(j)
      data_aug <- rbind(train, data.frame(resp = cand, xtest[i,]))
      ada_aug <- boosting(resp~., data = data_aug, mfinal = 35)
      
      true <- cbind(1:nrow(data_aug),data_aug$resp)
      
      #generate conformity score
      conf <- 1 - ada_aug$prob[true]
      
      #generate pi for each class/test observation combination
      pival[i,j] <- 1-rank(conf, ties.method = "random")[nrow(data_aug)]/nrow(data_aug)
    }
   
  }
  
  #adaboost model output probabilities for test data
  ada_pred <- predict(ada, xtest)
  
  #vector of alpha values
  alpha_vec <- seq(.05,.95, by = .05)
  
  coverage <- soft_coverage <- rep(0, times = length(alpha_vec))
  track <- 1
  for(alpha in alpha_vec){
    #finite-sample adjustment
    alpha <- 1-ceiling((1-alpha)*nrow(data_aug))/nrow(data_aug)
    
    #get prediction set
    set_tf <- pival >= alpha
    set <- data.frame(which(pival >= alpha, arr.ind = TRUE))
    set <- set %>% arrange(row)
    
    #choose all classes with output >= alpha
    prob_set_tf <- ada_pred$prob >= alpha
    prob_set <- data.frame(which(ada_pred$prob >= alpha, arr.ind = TRUE))
    prob_set <- prob_set %>% arrange(row)
    
    total_prob <- ada_pred$prob*set_tf
    sum_prob <- apply(total_prob, FUN = sum, MARGIN = 1)
    
    true_vec <- prob_vec <- rep(0, times = length(ytest))
    for(i in 1:length(ytest)){
      true_vec[i] <- ytest[i] %in% set$col[set$row == i]
      prob_vec[i] <- ytest[i] %in% prob_set$col[set$row == i]
    }
    
    #get coverage
    coverage[track] <- sum(true_vec)/length(ytest)
    soft_coverage[track] <- sum(prob_vec)/length(ytest)
    
    track <- track + 1
  }
  
  #track coverage across all repetitions
  all_reps_coverage <- rbind(all_reps_coverage, data.frame(rep = b, method = "conf", coverage = coverage, alpha = 1-alpha_vec))
  all_reps_coverage <- rbind(all_reps_coverage, data.frame(rep = b, method = "ada", coverage = soft_coverage, alpha = 1-alpha_vec))
}


#save coverage results for all simulation reps
#saveRDS(all_reps_coverage, "all_reps_coverage_small.RDS")
#write.csv(all_reps_coverage, "all_reps_coverage_small.csv")
#####TRAIN#####

#####VISUALIZATION#####
#prediction set examples with alpha = .25
alpha <- .25
alpha <- 1-ceiling((1-alpha)*nrow(data_aug))/nrow(data_aug)

#conformity scores
pival

#adaboost output
ada_pred$prob

#generate prediction set
set_tf <- pival >= alpha
set <- data.frame(which(pival >= alpha, arr.ind = TRUE))
set <- set %>% arrange(row)

#choose classes to exceed 1-alpha
ada_hold <- ada_pred$prob
ranks <- apply(ada_hold, FUN = rank, MARGIN = 1)
save_sort <- apply(ada_hold, FUN = sort, MARGIN = 1, decreasing = FALSE)
ranks <- rotate(ranks)
save_sort <- rotate(save_sort)
cum_sum <- apply(save_sort, FUN = cumsum, MARGIN = 1)
pick <- apply(cum_sum < 1 - alpha, FUN = sum, MARGIN = 2) + 1
pick_keep <- c()
for(k in 1:nrow(test)){
  pick_keep <- rbind(pick_keep, cbind(k,which(ranks[k,] <= pick[k])))
}

#pick_keep <- data.frame(pick_keep)
prob_set_tf <- matrix(FALSE, nrow = nrow(test), ncol = 4)
prob_set_tf[pick_keep] <- TRUE

#construct labels for plots
method_labs <- c("AdaBoost", "Conformal")
names(method_labs) <- c("ada", "conf")

#visualize prediction sets compared to adaboost output probabilities
melt_set <- data.frame(melt(set_tf), method = "conf")
melt_prob_set <- data.frame(melt(prob_set_tf), method = "ada")
melt_all <- rbind(melt_set, melt_prob_set)
melt_all %>% ggplot(aes(y = Var1, x = Var2, fill = value)) + 
  geom_tile() +
  coord_fixed() +
  scale_y_continuous(name = "Test Observation") +
  scale_x_continuous(name = "Class") +
  theme(text = element_text(size = 15),
        legend.position = "bottom") +
  scale_fill_discrete(name = "prediction set") +
  facet_grid(cols = vars(method), labeller = labeller(method = method_labs))

#plot calibration plot for one repetition of train/test split
b <- 1
one_rep <- all_reps_coverage %>% filter(rep == b, method == "conf")
one_rep_ada <- all_reps_coverage %>% filter(rep == b, method == "ada")

w <- 4
h <- 4
par(pin=c(w, h))
plot(x = 1 - alpha_vec, y = one_rep$coverage, type = "l", xlab = expression(1-alpha), ylab = "empircal coverage", lwd = 2)
lines(x = 1 - alpha_vec, y = one_rep_ada$coverage, lwd = 2, col = 2, lty = 2)
abline(a = 0, b = 1, lty = 3)

#pull from completed simulation
#all_reps_coverage <- readRDS("all_reps_coverage.RDS")
all_reps_coverage <- readRDS("all_reps_coverage_small.RDS")

all_reps_coverage %>% ggplot(aes(x = alpha, y = coverage, group = alpha)) + 
  geom_boxplot(aes(fill = method)) +
  geom_abline(linetype = "dashed", slope = 1, intercept = 0, size = 1.05) +
  scale_y_continuous(name = "empircal coverage") +
  scale_x_continuous(name = "nominal coverage") +
  theme(text = element_text(size = 15),
        legend.position = "none",
        panel.background = element_blank(), 
        panel.grid.major = element_line(color = "black", linetype = "dashed"), 
        axis.line = element_line(color = "black")) +
  coord_fixed() +
  facet_grid(cols = vars(method), labeller = labeller(method = method_labs))
#####VISUALIZATION#####
