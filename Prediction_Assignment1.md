# Human Activity Recognition - Weight Lifting Dataset
Pawan Mishra  
12/9/2017  

## Overview
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
We are provided with data from sensors on the belt, forearm, arm, and dumbell of 6 young health participants who were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).
We now look at this data and focus on discriminating between different activities, i.e. try to predict "which" activity was performed at a specific point in time.

## Downloading and exploring the data
We download and read the given training and testing datasets


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
```


```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```
The training dataset contains 19622 observarions of 160 variables. Dealing with so many variables can be computationaly daunting and many also reduce interpretability of our analysis. Thus we create a new dataset (training1) with only the variables that represent the accelerometer, gyroscope and magnetometer readings of the belt, forearm, arm, and dumbell sensors along the x, y and z coordinates.
All other variables are omitted because they seem to be derived in some form from these original sensor readings and we do not want to overfit our model.

```r
# column 37-45: belt sensor; 60-68: arm sensor; 113-121: bumbell sensor; 151-159: forearm sensor
# column 2: username; column 160: output class
colnums <- c(2, 37:45, 60:68, 113:121, 151:159, 160)
training1 <- training[, colnums]
```
Thus we will try to build a predictive model on the training1 dataset which contains 19622 observations of 38 variables: 37 predictors and 1 output "classe"

## Splitting data for cross validation
We will utilize the random sampling technique for splitting the training1 dataset into our actual training and testing datasets.

```r
library("caret")
```


```r
set.seed(100)
inTrain <- createDataPartition(y=training1$classe, p=0.75, list=FALSE)
trainingSubset <- training1[inTrain,]
testingSubset <- training1[-inTrain,]
```
So now our models will be built on the trainingSubset dataset and we will use the testingSubset for cross-validation.

## Building a Predictive Model
We will be fitting 3 different classification models to the trainingSubset dataset - random forest(rf), linear discriminant analysis(lda), and boosing with regression trees(gbm). We will evaluate the accuracy of all the three models through cross validation against the testingSubset dataset.
We will also stack the three models up to come up with a stacked model and compare its accuracy with the other 3 models.
Based on the accuracy we will select the best model for predicting the classe variable.

## Random Forest

```r
# Fitting Random Forest
fitRF <- train(classe ~ ., data = trainingSubset, method = "rf")
```


We cross validate the model with testingSubset and find out the accuracy.

```r
predRF <- predict(fitRF, testingSubset)
```

```r
confusionMatrix(predRF, testingSubset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393   10    0    2    0
##          B    2  928    7    0    0
##          C    0   11  848   27    1
##          D    0    0    0  775    2
##          E    0    0    0    0  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9874          
##                  95% CI : (0.9838, 0.9903)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.984           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9779   0.9918   0.9639   0.9967
## Specificity            0.9966   0.9977   0.9904   0.9995   1.0000
## Pos Pred Value         0.9915   0.9904   0.9560   0.9974   1.0000
## Neg Pred Value         0.9994   0.9947   0.9983   0.9930   0.9993
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1892   0.1729   0.1580   0.1831
## Detection Prevalence   0.2865   0.1911   0.1809   0.1584   0.1831
## Balanced Accuracy      0.9976   0.9878   0.9911   0.9817   0.9983
```
We get an accuracy of 98.74% by this model.

## Linear Discriminant Analysis

```r
# Fitting Linear Discriminant Analysis
fitLDA <- train(classe ~ ., data = trainingSubset, method = "lda")
```
We cross validate the model with testingSubset and find out the accuracy.

```r
predLDA <- predict(fitLDA, testingSubset)
```

```r
confusionMatrix(predLDA, testingSubset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1129  171  142   77   42
##          B   59  589   73   54  128
##          C  116   95  513   94   86
##          D   88   33  108  512  137
##          E    3   61   19   67  508
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6629          
##                  95% CI : (0.6495, 0.6762)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5721          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8093   0.6207   0.6000   0.6368   0.5638
## Specificity            0.8769   0.9206   0.9034   0.9107   0.9625
## Pos Pred Value         0.7233   0.6523   0.5675   0.5831   0.7720
## Neg Pred Value         0.9204   0.9100   0.9145   0.9275   0.9074
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2302   0.1201   0.1046   0.1044   0.1036
## Detection Prevalence   0.3183   0.1841   0.1843   0.1790   0.1342
## Balanced Accuracy      0.8431   0.7706   0.7517   0.7738   0.7632
```
This model gives a significantly low accuracy of 66.29%.

## Boosting with regression trees

```r
# Fitting GBM
fitGBM <- train(classe ~ ., data = trainingSubset, method = "gbm", verbose = FALSE)
```


We cross validate the model with testingSubset and find out the accuracy.

```r
predGBM <- predict(fitGBM, testingSubset)
```

```r
confusionMatrix(predGBM, testingSubset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1356   53   36   28    6
##          B   13  836   41   14   37
##          C    9   48  761   44   22
##          D   13   10   15  712   26
##          E    4    2    2    6  810
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9125          
##                  95% CI : (0.9043, 0.9203)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8891          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9720   0.8809   0.8901   0.8856   0.8990
## Specificity            0.9649   0.9735   0.9696   0.9844   0.9965
## Pos Pred Value         0.9168   0.8884   0.8609   0.9175   0.9830
## Neg Pred Value         0.9886   0.9715   0.9766   0.9777   0.9777
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2765   0.1705   0.1552   0.1452   0.1652
## Detection Prevalence   0.3016   0.1919   0.1803   0.1582   0.1680
## Balanced Accuracy      0.9685   0.9272   0.9298   0.9350   0.9478
```

By this model we get an accuracy of 91.25% in the predictions.

## Stacking up the models using random forest method

```r
predDF <- data.frame(predRF, predLDA, predGBM, classe=testingSubset$classe)
combModFit <- train(classe ~., method="rf", data=predDF) 
combPred <- predict(combModFit,predDF)
```

```r
confusionMatrix(combPred, predDF$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393   10    0    2    0
##          B    2  928    7    0    0
##          C    0   11  848   27    1
##          D    0    0    0  775    2
##          E    0    0    0    0  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9874          
##                  95% CI : (0.9838, 0.9903)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.984           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9779   0.9918   0.9639   0.9967
## Specificity            0.9966   0.9977   0.9904   0.9995   1.0000
## Pos Pred Value         0.9915   0.9904   0.9560   0.9974   1.0000
## Neg Pred Value         0.9994   0.9947   0.9983   0.9930   0.9993
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1892   0.1729   0.1580   0.1831
## Detection Prevalence   0.2865   0.1911   0.1809   0.1584   0.1831
## Balanced Accuracy      0.9976   0.9878   0.9911   0.9817   0.9983
```
Stacking up the models gives us the same accuracy as random forest model i.e. 98.74%
Thus based on the cross validated accuracies of all the models we find the Random Forest model to be the best one.

## Predicting classe for given 20 test observations
Thus we fit our Random Forest model to the given testing data and predict the classe for the 20 test observations. Through cross validation we estimate the accuracy of these predictions to be ~98.74%

```r
# Applying model fitRF on testing dataset
FinalPredRF <- predict(fitRF, testing)
```

```r
FinalPredRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
 

<End of Report>


