
# import data set
bank_personal_loan <- read.csv("bank_personal_loan.csv",fill = TRUE, header = FALSE)

# change col names
col.name <- bank_personal_loan[1,]
colnames(bank_personal_loan) <- col.name
bank_personal_loan <- bank_personal_loan[-1,]
rownames(bank_personal_loan) <- c(1:5000)



# Explore the data
if(!require("skimr")){
  install.packages("skimr")
}
skim(bank_personal_loan)
bpl.data <- bank_personal_loan[,-9]
bpl.data <- as.data.frame(lapply(bpl.data,as.numeric))
pairs(bpl.data)
DataExplorer::plot_histogram(bpl.data, ncol = 4)



# Split test set and detection set
set.seed(1234)
bank_personal_loan <- as.data.frame(lapply(bank_personal_loan,as.numeric))
n <- dim(bank_personal_loan)[1]
train.n <- ceiling(0.8*n)
train.index <- sample(1:n, train.n)
train.data <- na.omit(bank_personal_loan[train.index,])
test.data <- na.omit(bank_personal_loan[-train.index,])



# logistic regression linear
train.glm <- glm(as.factor(Personal.Loan) ~ ., binomial, train.data)
summary(train.glm)
pred.glm <- predict(train.glm, test.data, type = "response")
conf.mat <- table(`true Personal.Loan` = test.data$Personal.Loan, `predict Personal.Loan` = pred.glm > 0.5)
conf.mat
glm.acc <- (58 + 890) / (58 + 890 + 12 + 40)
glm.acc

# logistic regression poly
if(!require('boot')){
  install.packages('boot')
}
cv.error.poly <- rep(0,5)
for(i in 1:5){
  train.glm.poly <- glm(as.factor(Personal.Loan) ~ poly(Age+Experience+Income+
                                                          ZIP.Code+Family+CCAvg+Education+Mortgage+Securities.Account	+
                                                          CD.Account+Online+CreditCard,i), binomial, train.data)
  cv.error.poly[i] <- cv.glm(train.data,train.glm.poly,K=10)$delta[1]
}
cv.error.poly
train.glm.poly.fin <- glm(as.factor(Personal.Loan) ~ poly(Age+Experience+Income+
                                                            ZIP.Code+Family+CCAvg+Education+Mortgage+Securities.Account	+
                                                            CD.Account+Online+CreditCard,1), binomial, train.data)
pred.glm2 <- predict(train.glm.poly.fin, test.data, type = "response")
conf.mat2 <- table(`true Personal.Loan` = test.data$Personal.Loan, `predict Personal.Loan` = pred.glm2 > 0.5)
conf.mat2


# LDA
if(!require('MASS')){
  install.packages('MASS')
}
train.lda <- MASS::lda(Personal.Loan ~ ., train.data, cv=true)
summary(train.lda)
pred.lda <- predict(train.lda, test.data,drop.unused.levels = TRUE)
lda.acc <- mean(I(pred.lda$class == na.omit(test.data)$Personal.Loan))
lda.acc
table(`predict Personal.Loan`=pred.lda$class,`true Personal.Loan`=test.data$Personal.Loan)
acc <- (887+57)/(887+57+41+15)
acc


# Tree
if(!require('tree')){
  install.packages('tree')
}
train.tree <- tree(as.factor(Personal.Loan) ~ ., train.data)
summary(train.tree)
pred.tree <- predict(train.tree, test.data, type = 'class')
table(pred.tree,`true Personal.Loan`=test.data$Personal.Loan)
tree.acc <- (899 + 85)/(899+13+3+85)
tree.acc
plot(train.tree)
text(train.tree,cex=0.8)

# improvements, prune the tree
train.cv.tree <- cv.tree(train.tree,FUN = prune.misclass)
summary(train.cv.tree)
par(mfrow=c(1,2))
plot(train.cv.tree$size,train.cv.tree$dev,type="b")
plot(train.cv.tree$k, train.cv.tree$dev,type="b")
train.cv.tree
prunes.tree <- prune.misclass(train.tree,best = 7)
pred.tree.prune <- predict(prunes.tree, test.data, type = 'class')
table(pred.tree.prune,`true Personal.Loan`=test.data$Personal.Loan)
pred.tree.prune.acc <- (899+83)/(899+83+3+15)
pred.tree.prune.acc
plot(prunes.tree)
text(prunes.tree,cex=0.8)


# bag
if(!require("randomForest")){
  install.packages("randomForest")
}
set.seed(123)
train.bag <- randomForest(as.factor(Personal.Loan) ~ ., train.data, mtry=12,importance=TRUE)
summary(train.bag)
pred.bag <- predict(train.bag,test.data, type = 'class')
table(pred.bag, `true Personal.Loan`=test.data$Personal.Loan)
bag.acc <- (898 + 85) / (898 + 85+ 13 + 4)
bag.acc

# randomFroest
set.seed(123)
train.rF <- randomForest(as.factor(Personal.Loan) ~ ., train.data, mtry=3,importance=TRUE)
train.rF
pred.rF <- predict(train.rF,test.data, type = 'class')
table(pred.rF, `true Personal.Loan`=test.data$Personal.Loan)
rF.acc <- (901 + 85) / (901 + 85+1 + 13)
rF.acc




# Neural Networks
if(!require("keras")){
  install.packages("keras")
}
n <- dim(train.data)[1]
train.n.v <- ceiling(0.7*n)
train.index <- sample(1:n, train.n.v)
train.data2 <- na.omit(train.data[train.index,])
validate.data <- na.omit(train.data[-train.index,])

data_train_x <- as.matrix(train.data2[,-9])
data_train_y <- as.matrix(train.data2[,9])

data_validate_x <- as.matrix(validate.data[,-9])
data_validate_y <- as.matrix(validate.data[,9])

data_test_x <- as.matrix(test.data[,-9])
data_test_y <- as.matrix(test.data[,9])

deep.net2 <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(data_train_x))) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")
deep.net2

deep.net2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

deep.net2 %>% fit(
  data_train_x, data_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(data_validate_x, data_validate_y),
)

pred_test_prob <- deep.net2 %>% predict_proba(data_test_x)
pred_test_res <- deep.net2 %>% predict_classes(data_test_x)
table(pred_test_res, data_test_y)


# ROC curve
if(!require("pROC")){
  install.packages("pROC")
}
par(mfrow=c(2,3))
roc(test.data$Personal.Loan,as.numeric(pred.glm), plot=TRUE,main='Logistic(Linear)')
roc(test.data$Personal.Loan,as.numeric(pred.glm2), plot=TRUE,main='Logistic(poly)')
roc(test.data$Personal.Loan,as.numeric(pred.lda$class), plot=TRUE,main='LDA')
roc(test.data$Personal.Loan,as.numeric(pred.tree), plot=TRUE,main='Tree')
roc(test.data$Personal.Loan,as.numeric(pred.bag), plot=TRUE,main='Bagging')
roc(test.data$Personal.Loan,as.numeric(pred.rF), plot=TRUE,main='RandomForest')



