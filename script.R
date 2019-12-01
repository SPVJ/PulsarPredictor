#------------------------- Setup and Libraries ---------------------------

library(keras)
#keras is a machine learning library importing from PyThon, allowing it to be used on R.
library(tidyr)
#tidyr is a library used for data cleaning such as droping irrelevant columns, removing missing data
#swap columns position etc.

library(ggplot2)

install_keras()

pulsar_raw <- read.csv("pulsar_stars.csv")
#load our data into pulsar dataframe

str(pulsar_raw)
summary(pulsar_raw)

#take a look at our dataframe


#----------------------- Preprocessing Data ---------------------#

pulsar <- pulsar_raw %>% drop_na()
#drop any row with n/a value in the features

str(pulsar)
# total of 17898 rows, no change in the previous step -> no data was dropped due to na value

pulsar <- as.matrix(pulsar)
str(pulsar)
#turn df into martix format for keras

pulsar[,1:8] <- normalize(pulsar[,1:8])
pulsar[,9] <- as.numeric(pulsar[,9])
#normalizing our input data so they can be trained by model easier now that data is normalized

summary(pulsar)

set.seed(123)
#set a rng seed for r

train_test <- sample(2, nrow(pulsar),replace = T,prob = c(0.8,0.2))
# splitting data into train/test at 14319/3579 respectively

pulsar_train <- pulsar[train_test==1,1:8]
pulsar_test <- pulsar[train_test==2,1:8]
train_target <- pulsar[train_test==1,9]
test_target <- pulsar[train_test==2,9]
#retrieving data from the splitting

train_label <- to_categorical(train_target)
test_label <- to_categorical(test_target)
#one hot encoder on the target class

print(pulsar_train)
print(pulsar_test)
print(train_target)


#----------------------- Creating model ----------------------------


model <- keras_model_sequential()
#use model sequential for Artificial Neural Network

model %>% 
  layer_dropout(rate = 0.15,input_shape = c(8)) %>%
  layer_dense(units = 24,activation = "relu") %>%
  layer_dense(units = 16,activation = "relu") %>%
  layer_dense(units = 2,activation = "softmax")

summary(model)

#the model comprises of 1 input layer with 0.15 dropout rate preventing overfitting
#followed by 2 hidden layers with relu activation function
#and 1 output layer with softmax activation

model %>%
  compile(loss = "binary_crossentropy", optimizer ='adam', metrics = 'accuracy')

#add compiler to the model, binary crossentropy for binary classification
#adam optimizer and accuracy as a metrics

history <- model %>%
  fit(pulsar_train,train_label,epoch = 50, batch_size = 128, validation_split = 0.2)

plot(history)
#80% of the train data is used for fitting a model
#20% of the train data is used as a validation data to adjust the hyperparameter of the model
#loss: 0.1043 - accuracy: 0.9681 - val_loss: 0.0359 - val_accuracy: 0.9899


model %>%
  evaluate(pulsar_test,test_label)
# 0.9795 accuracy and 0.0414 loss on test data

predicted <- model %>%
  predict_classes(pulsar_test)

#let's compare our prediction and actual obs with table

table(Predicted = predicted, Actual = test_target)

#we see that actual non-pulsar are 3231 which our model guessed 3217 correct
#for pulsar, our model guessed 272 correct out of 331 total pulsars
#The model predicts non-pulsar with 0.9956 accuracy
#The model, however, only predict pulsar star with only 0.8217 accuracy