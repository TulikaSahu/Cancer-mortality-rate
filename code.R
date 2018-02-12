#1. LOAD LIBRARIES
library(readr)
library(caret)
library(mlbench)
library(ggplot2)
library(modelr)
library(MASS)
library(leaps)
library(car)

#2. IMPORT DATA
myd <- read_csv("C:/Users/Tulika Sahu/Desktop/Spring 2017/Data Analytics/Final Project/1Tulika/cancer_reg.csv")

#3. CHECK WHETHER OUR DATA CONTAINS MISSING VALUES OR NOT
myd<-na.omit(myd)

#4. CREATE VARIABLES TO STORE PREDICTORS
#assigning variables
tavgAnnCount = myd$avgAnnCount
tdeathrate = myd$TARGET_deathRate
tavgdeathsyear = myd$avgDeathsPerYear
tincidencerate = myd$incidenceRate
tpop2015 = myd$popEst2015
tpoverty = myd$povertyPercent
tstudy = myd$studyPerCap
tmedianage = myd$MedianAge
tmedianagemale = myd$MedianAgeMale
tmedianagefemale = myd$MedianAgeFemale
tgeo = myd$Geography
tavghouse = myd$AvgHouseholdSize
tmarried = myd$PercentMarried
ths = myd$PctHS25_Over
temp = myd$PctEmployed16_Over
tunemp = myd$PctUnemployed16_Over
tgovcoverage = myd$PctPublicCoverage
tmarriedhouse = myd$PctMarriedHouseholds
tbirthrate = myd$BirthRate

#Geography variable factor variable

#using as.factor() for categorical data, Geography variable 
tgeo.Fac <- as.factor(tgeo)
tgeo.F <- as.integer(tgeo.Fac)
tgeo.F

#***
#Scatter plots for each independent variable against response variable
plot(tavgAnnCount, tdeathrate)
plot(tavgdeathsyear, tdeathrate)
plot(tincidencerate, tdeathrate)
plot(tpop2015, tdeathrate)
plot(tpoverty, tdeathrate)
plot(tstudy, tdeathrate)
plot(tmedianage, tdeathrate)
plot(tmedianagemale, tdeathrate)
plot(tmedianagefemale, tdeathrate)
plot(tavghouse, tdeathrate)
plot(tmarried, tdeathrate)
plot(ths, tdeathrate)
plot(temp, tdeathrate)
plot(tunemp, tdeathrate)
plot(tgovcoverage, tdeathrate)
plot(tmarriedhouse, tdeathrate)
plot(tbirthrate, tdeathrate)
plot(tgeo.F, tdeathrate)

#***-Transformation-****

tavgAnnCount = as.integer(log(tavgAnnCount))
tavgdeathsyear = as.integer(log(tavgdeathsyear))
tincidencerate = as.integer(sqrt(tincidencerate))
tpop2015 = as.integer(sqrt(log(tpop2015)))
tpoverty = as.integer(sqrt(tpoverty))
tstudy = complete.cases(log(tstudy))
tavghouse = tavghouse*tavghouse
tmarried = tmarried*tmarried
tunemp = sqrt(tunemp)
tgovcoverage = tgovcoverage*tgovcoverage
tmarriedhouse = tmarriedhouse*tmarriedhouse*tmarriedhouse


#After transformation Scatter plots for each independent variable against response variable 
plot(tavgAnnCount, tdeathrate)
plot(tavgdeathsyear, tdeathrate)
plot(tincidencerate, tdeathrate)
plot(tpop2015, tdeathrate)
plot(tpoverty, tdeathrate)
plot(tstudy, tdeathrate)
plot(tmedianage, tdeathrate)
plot(tmedianagemale, tdeathrate)
plot(tmedianagefemale, tdeathrate)
plot(tavghouse, tdeathrate)
plot(tmarried, tdeathrate)
plot(ths, tdeathrate)
plot(temp, tdeathrate)
plot(tunemp, tdeathrate)
plot(tgovcoverage, tdeathrate)
plot(tmarriedhouse, tdeathrate)
plot(tbirthrate, tdeathrate)

#################################-----------------------------------------------------------------------------------------------------
#5. CREATE FOLDS FOR N-FOLD CROSS VALIDATION

#Randomly Shuffling the data 
myd<-myd[sample(nrow(myd)),]

set.seed(3)
folds <- crossv_kfold(myd, k = 5)

#6. CONVERT RESAMPLE OBJECT TO REGULAR DATA FRAME
train1=as.data.frame(folds$train[[1]])
train2=as.data.frame(folds$train[[2]])
train3=as.data.frame(folds$train[[3]])
train4=as.data.frame(folds$train[[4]])
train5=as.data.frame(folds$train[[5]])
test1=as.data.frame(folds$test[[1]])
test2=as.data.frame(folds$test[[2]])
test3=as.data.frame(folds$test[[3]])
test4=as.data.frame(folds$test[[4]])
test5=as.data.frame(folds$test[[5]])

# declare and define list of training data sets
trainlist=list(train1,train2,train3,train4,train5)

# declare and define list of test data sets
testlist= list(test1,test2,test3,test4,test5)

#################################-----------------------------------------------------------------------------------------------------
##Model fitting

#base model with no significant variable
base = lm(tdeathrate~tincidencerate, data = myd)
Full = lm(tdeathrate~tavgAnnCount + tavgdeathsyear + tincidencerate + tpop2015 + tpoverty + tstudy + tmedianage 
          + tmedianagemale + tmedianagefemale + tgeo.F + tavghouse + tmarried + ths + temp + tunemp + tgovcoverage 
          + tmarriedhouse + tbirthrate, data=folds$train[[1]], na.action=na.exclude)

# correlation among transformed variables
allcorr=cor(cbind(deathrate, avgAnnCount, avgdeathsyear, incidencerate, pop2015, poverty, study, medianage, medianagemale, medianagefemale, geo.f, avghouse, married, hs, emp, unemp, govcoverage, marriedhouse, birthrate))
write.csv(allcorr, file = "allcorrelation_transformed.csv",row.names=FALSE)

## checking multicolinearity for independent variables.
vif(Full)
 
#defining full model with all variables
vif(lm(tdeathrate~tavgAnnCount + tincidencerate + tpop2015 + tpoverty + tstudy + tmedianage 
       + tmedianagemale + tmedianagefemale + tgeo.F + tavghouse + tmarried + ths + temp + tunemp + tgovcoverage 
       + tmarriedhouse + tbirthrate, data=folds$train[[1]]))

 
Full.new = lm(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tstudy + tmedianage 
              + tmedianagefemale + tgeo.F + tavghouse + ths + temp + tunemp + tgovcoverage 
              + tmarriedhouse + tbirthrate, data=folds$train[[1]])
summary(Full.new)

#####################-------------------------------------------------------########################
##########---------------------------Model Selection-----------------------------------------------
#####################-------------------------------------------------------########################

#STEPWISE SELECTION
Model.stepF = step(base, scope=list(upper=Full.new, lower=~1), direction = "both", trace=F)
Model.stepF
summary(Model.stepF)

######--- BACKWARD ELIMINATION


dropterm(Full.new, test = "F")

Mod1 = lm(tdeathrate ~ tavgAnnCount + tincidencerate + tpoverty +  
            tmedianage + tmedianagefemale + tgeo.F + tavghouse + ths + 
            temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=folds$train[[1]], na.action=na.exclude)
dropterm(Mod1, test = "F")
summary(Mod1)

Mod2 = lm(tdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
            tmedianage + tmedianagefemale + tgeo.F + ths + 
            temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, 
          data = folds$train[[1]], na.action = na.exclude)
dropterm(Mod2, test = "F")
summary(Mod2)

Mod3 = lm(tdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
            tmedianagefemale + tgeo.F + ths + temp + tunemp + 
            tgovcoverage + tmarriedhouse + tbirthrate, data = folds$train[[1]], 
          na.action = na.exclude)
dropterm(Mod3, test = "F")
summary(Mod3)

Mod4 = lm(tdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
            tmedianagefemale + tgeo.F + ths + temp + tgovcoverage + 
            tmarriedhouse + tbirthrate, data = folds$train[[1]], na.action = na.exclude)
dropterm(Mod4, test = "F")
summary(Mod4)


## Model selection by Best Subsets Algorithm
#library(leaps)
 

Regsubsets.out <- regsubsets(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tstudy + tmedianage 
                             + tmedianagefemale + tgeo.F + tavghouse + ths + temp + tunemp + tgovcoverage 
                             + tmarriedhouse + tbirthrate, data=folds$train[[1]], nbest = 1, nvmax = NULL, 
                             force.in = NULL, force.out = NULL, 
                             method = "exhaustive")
Regsubsets.out             

Summary.Out <- summary(Regsubsets.out)
as.data.frame(Summary.Out$outmat)

####
## Adjusted R2 
plot(Regsubsets.out, scale = "adjr2", main = "Adjusted R^2")
#### 

## Mallows Cp
plot(Regsubsets.out, scale = "Cp", main = "Mallow's Cp")

#library(car)
layout(matrix(1:1, ncol = 1))


## Adjusted R2
res.legend <-subsets(regsubsets.out, statistic="adjr2", legend = FALSE, min.size = 10, main = "Adjusted R^2")
abline(a = 0, b = 0, col="red")
## Mallow Cp
res.legend <-subsets(regsubsets.out, statistic="cp", legend = FALSE, min.size = 5, main = "Mallow Cp")
abline(a = 1, b = 1, lty = 2, col="red")


res.legend
which.max(Summary.Out$adjr2)
Summary.Out$which[11,]


B.subset.model <- lm(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                     + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=folds$train[[1]])

summary(B.subset.model)

 
###--------------------------Model evaluation method: N-fold cross validation 

#Randomly Shuffling the data 
myd<-myd[sample(nrow(myd)),]

#Create 5 equally size folds
folds <-cut(seq(1,nrow(myd)),breaks=11,labels=FALSE)

#Performing 5 fold cross validation
for(i in 1:5)
{
  #Segementingy data by fold using the which() function
  
  #testIndexes<-which(folds==i,arr.ind=TRUE)
  #testData<-myd[testIndexes, ]
  #trainData<-myd[-testIndexes, ]
  ##trainData
  #y = testData[,1]
  #y
  
  Model.stepF.1 = lm(tdeathrate ~ tincidencerate + tgovcoverage + ths + 
                     tpoverty + tgeo.F + tmedianagefemale + tbirthrate + tavgAnnCount + 
                     tunemp + tmarriedhouse + temp, data = trainlist[[i]])
 
  #Prediction by model got by STEPWISE SELECTION
  y_1.1 = predict.glm(Model.stepF.1, testlist[[i]])
  
  
  ##RMSE for model got by STEPWISE SELECTION
  #sqrt(mean((trainData$tdeathrate-testData$tdeathrate)^2, na.rm = TRUE))
  rmse_1.1 = sqrt(mean((y-y_1.1)^2, na.rm = TRUE))
  
  
  #  Validation for Backward elimination model
  Mod4.1 = lm(tdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
              tmedianagefemale + tgeo.F + ths + temp + tgovcoverage + 
              tmarriedhouse + tbirthrate, data = trainlist[[i]], na.action = na.exclude)
   
  #Prediction by model got by Backward elimination model  
  y_2.1 = predict.glm(Mod4.1, testlist[[i]])
  
  
  ##RMSE for model got by STEPWISE SELECTION
  #sqrt(mean((trainData$tdeathrate-testData$tdeathrate)^2, na.rm = TRUE))
  rmse_2.1 = sqrt(mean((y-y_2.1)^2, na.rm = TRUE))
  
  
  #  Validation for Best Subset model
  B.subset.model.1 = lm(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                       + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=trainlist[[i]])
  
  #Prediction by model got by BEST SUBSET
  y_3.1 = predict.glm(B.subset.model.1, testlist[[i]])
   
  
  ##RMSE for model got by BEST SUBSET
  #sqrt(mean((trainData$tdeathrate-testData$tdeathrate)^2, na.rm = TRUE))
   
  rmse_3.1 = sqrt(mean((y-y_3.1)^2, na.rm = TRUE))
   
  
  #Prediction on Full model 
  y_4 = predict.glm(Full.new, testlist[[i]])
  y_4
  
  ##RMSE on Full model 
  #sqrt(mean((trainData$tdeathrate-testData$tdeathrate)^2, na.rm = TRUE))
  rmse_4 = sqrt(mean((y-y_4)^2, na.rm = TRUE))
  rmse_4
  Full.new.1 = lm(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tstudy + tmedianage 
                + tmedianagefemale + tgeo.F + tavghouse + ths + temp + tunemp + tgovcoverage 
                + tmarriedhouse + tbirthrate, data=trainlist[[i]])
   

  y_4.1 = predict.glm(Full.new.1, testlist[[i]])
  
  
  rmse_4.1 = sqrt(mean((y-y_4.1)^2, na.rm = TRUE))
 
}
#############Calculating final RMSE for all the three models:


rmse_1 = mean(rmse_1.1)
rmse_1 

rmse_2 = mean(rmse_2.1)
rmse_2

rmse_3 = mean(rmse_3.1)
rmse_3

rmse_4 = mean(rmse_4.1)
rmse_4
##########

####                                                                                     ####     
####----------------RESIDUAL ANALYSIS FOR STEPWISE MODEL SELECTION ----------------------####
####                                                                                     ####   

### Residual analysis for STEPWISE MODEL
plot( fitted(Model.stepF), rstandard(Model.stepF), xlab = "Predicted", ylab = "Residual", 
      main="Predicted vs residuals plot for Stepwise selection model")
abline(a=0, b=0, col="red")

#applying log transformation to y-variable
Ltdeathrate = log(tdeathrate)
LModel.stepF = lm(Ltdeathrate ~ tincidencerate + tgovcoverage + ths + 
                    tpoverty + tgeo.F + tmedianagefemale + tbirthrate + tavgAnnCount + 
                    tunemp + tmarriedhouse + temp, data = myd)
summary(LModel.stepF)
plot( fitted(LModel.stepF), rstandard(LModel.stepF), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Stepwise selection model Log transform")
abline(a=0, b=0, col="red")

Stdeathrate = sqrt(tdeathrate)
SModel.stepF = lm(Stdeathrate ~ tincidencerate + tgovcoverage + ths + 
                    tpoverty + tgeo.F + tmedianagefemale + tbirthrate + tavgAnnCount + 
                    tunemp + tmarriedhouse + temp, data = myd)
summary(SModel.stepF)
plot( fitted(SModel.stepF), rstandard(SModel.stepF), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Stepwise selection model Sqrt transform")
abline(a=0, b=0, col="red")


Itdeathrate = 1/(tdeathrate)
IModel.stepF = lm(Itdeathrate ~ tincidencerate + tgovcoverage + ths + 
                    tpoverty + tgeo.F + tmedianagefemale + tbirthrate + tavgAnnCount + 
                    tunemp + tmarriedhouse + temp, data = myd)

plot( fitted(IModel.stepF), rstandard(IModel.stepF), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Stepwise selection model Inverse-y")
abline(a=0, b=0, col="red")
summary(IModel.stepF)

#No improvement in model observed(during residual analysis) after transformation of response variable,
#so we with the untransformed model

#Plot residuals vs each x-variable:
plot(tincidencerate, rstandard(Model.stepF), xlab = "tincidencerate", ylab = "Residual", main="tincidencerate vs residuals plot")
abline(a=0, b=0, col="red")
plot(tgovcoverage, rstandard(Model.stepF), xlab = "tgovcoverage", ylab = "Residual", main="tgovcoverage vs residuals plot")
abline(a=0, b=0, col="red")
plot(ths, rstandard(Model.stepF), xlab = "ths", ylab = "Residual", main="ths vs residuals plot")
abline(a=0, b=0, col="red")
plot(tmarriedhouse, rstandard(Model.stepF), xlab = "tpoverty", ylab = "Residual", main="poverty vs residuals plot")
abline(a=0, b=0, col="red")
plot(tgeo.F, rstandard(Model.stepF), xlab = "tgeo.F", ylab = "Residual", main="geo.F vs residuals plot")
abline(a=0, b=0, col="red")
plot(tmedianagefemale, rstandard(Model.stepF), xlab = "tmedianagefemale", ylab = "Residual", main="medianagefemale vs residuals plot")
abline(a=0, b=0, col="red")
plot(tbirthrate, rstandard(Model.stepF), xlab = "tbirthrate", ylab = "Residual", main="birthrate vs residuals plot")
abline(a=0, b=0, col="red")
plot(tavgAnnCount, rstandard(Model.stepF), xlab = "tavgAnnCount", ylab = "Residual", main="avgAnnCount vs residuals plot")
abline(a=0, b=0, col="red")
plot(tunemp, rstandard(Model.stepF), xlab = "tunemp", ylab = "Residual", main="unemp vs residuals plot")
abline(a=0, b=0, col="red") 
plot(tmarriedhouse, rstandard(Model.stepF), xlab = "tmarriedhouse", ylab = "Residual", main="marriedhouse vs residuals plot")
abline(a=0, b=0, col="red")
plot(temp, rstandard(Model.stepF), xlab = "temp", ylab = "Residual", main="emp vs residuals plot")
abline(a=0, b=0, col="red")

##QQ-plot
qqnorm(rstandard(Model.stepF))
qqline(rstandard(Model.stepF), col = 2)

#Shapiro-Wilk normality test
sp1 = lm(tdeathrate ~ tincidencerate + tgovcoverage + ths + 
           tpoverty + tgeo.F + tmedianagefemale + tbirthrate + tavgAnnCount + 
           tunemp + tmarriedhouse + temp, data = myd)
shapiro.test(residuals(sp1))
##--------Checking Influencial data points
inf = influence.measures(Model.stepF)
summary(inf)
## checking multicolinearity for independent variables.
vif(Model.stepF)

####                                                                                    ####     
####----------------RESIDUAL ANALYSIS FOR BACKWARD SELECTION MODEL----------------------####
####                                                                                    ####   

### Residual analysis for BACKWARD ELIMINATION
plot( fitted(Mod4), rstandard(Mod4), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Backward elimination model")
abline(a=0, b=0, col="red")

#applying log transformation to y-variable
Ltdeathrate = log(tdeathrate)
LMod4 = lm(Ltdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
             tmedianagefemale + tgeo.F + ths + temp + tgovcoverage + 
             tmarriedhouse + tbirthrate, data = myd, na.action = na.exclude)
summary(LMod4)

plot( fitted(LMod4), rstandard(LMod4), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Backward elimination model Log transform")
abline(a=0, b=0, col="red")

#applying sqrt transformation to y-variable
Stdeathrate = sqrt(tdeathrate)
SMod4 = lm(Stdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
             tmedianagefemale + tgeo.F + ths + temp + tgovcoverage + 
             tmarriedhouse + tbirthrate, data = myd, na.action = na.exclude)
summary(SMod4)

plot( fitted(SMod4), rstandard(SMod4), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Backward elimination model Sqrt")
abline(a=0, b=0, col="red")

#applying 1/ y-variable
Itdeathrate = 1/(tdeathrate)

IMod4 = lm(Itdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
             tmedianagefemale + tgeo.F + ths + temp + tgovcoverage + 
             tmarriedhouse + tbirthrate, data = myd, na.action = na.exclude)
summary(IMod4)

plot( fitted(IMod4), rstandard(IMod4), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Backward elimination model Inverse")
abline(a=0, b=0, col="red")
#No improvement in model observed(during residual analysis) after transformation of response variable,
#so we with the untransformed model
 
 
####Plot residuals vs each x-variable: ***************----------Tulika-----------
plot(tincidencerate, rstandard(Mod4), xlab = "tincidencerate", ylab = "Residual", main="incidencerate vs residuals plot")
abline(a=0, b=0, col="red")

plot(tgovcoverage, rstandard(Mod4), xlab = "tgovcoverage", ylab = "Residual", main="govcoverage vs residuals plot")
abline(a=0, b=0, col="red")
plot(ths, rstandard(Mod4), xlab = "ths", ylab = "Residual", main="hs vs residuals plot")
abline(a=0, b=0, col="red")
plot(tmarriedhouse, rstandard(Mod4), xlab = "tpoverty", ylab = "Residual", main="poverty vs residuals plot")
abline(a=0, b=0, col="red")
plot(tgeo.F, rstandard(Mod4), xlab = "tgeo.F", ylab = "Residual", main="geo.F vs residuals plot")
abline(a=0, b=0, col="red")
plot(tmedianagefemale, rstandard(Mod4), xlab = "tmedianagefemale", ylab = "Residual", main="medianagefemale vs residuals plot")
abline(a=0, b=0, col="red")
plot(tbirthrate, rstandard(Mod4), xlab = "tbirthrate", ylab = "Residual", main="birthrate vs residuals plot")
abline(a=0, b=0, col="red")
plot(tavgAnnCount, rstandard(Mod4), xlab = "tavgAnnCount", ylab = "Residual", main="avgAnnCount vs residuals plot")
abline(a=0, b=0, col="red")
plot(tmarriedhouse, rstandard(Mod4), xlab = "tmarriedhouse", ylab = "Residual", main="marriedhouse vs residuals plot")
abline(a=0, b=0, col="red")
plot(temp, rstandard(Mod4), xlab = "temp", ylab = "Residual", main="emp vs residuals plot")
abline(a=0, b=0, col="red")

####------Normality test
##QQ-plot
qqnorm(rstandard(Mod4))
qqline(rstandard(Mod4), col = 2)

#Shapiro-Wilk normality test
sp.backward = lm(tdeathrate ~ tavgAnnCount + tincidencerate + tpoverty + 
                   tmedianagefemale + tgeo.F + ths + temp + tgovcoverage + 
                   tmarriedhouse + tbirthrate, data = myd, na.action = na.exclude)
shapiro.test(residuals(sp.backward))

##--------Checking Influencial data points
inf = influence.measures(Mod4)
summary(inf)
## checking multicolinearity for independent variables.
vif(Mod4)

####                                                                             ####     
####----------------RESIDUAL ANALYSIS FOR BEST SUBSET MODEL----------------------####
####                                                                             ####  


B.subset.model <- lm(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                     + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)
plot( fitted(B.subset.model), rstandard(B.subset.model), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for best subset model")
abline(a=0, b=0, col="red")

#applying log transformation to y-variable
Ltdeathrate = log(tdeathrate)
LB.subset.model <- lm(Ltdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                      + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)


summary(LB.subset.model)

plot( fitted(LB.subset.model), rstandard(LB.subset.model), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Best Subset model log transformation")
abline(a=0, b=0, col="red")

#applying sqrt transformation to y-variable
Stdeathrate = sqrt(tdeathrate)
SB.subset.model <- lm(Stdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                      + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)

summary(SB.subset.model)
#try
SB.subset.model1 <- lm(Stdeathrate~tavgAnnCount + tincidencerate + tmedianagefemale
                       + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)
summary(SB.subset.model1)
SB.subset.model2 <- lm(Stdeathrate~tavgAnnCount + tincidencerate + tmedianagefemale
                       + tgeo.F + ths + temp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)

summary(SB.subset.model2)
plot( fitted(SB.subset.model2), rstandard(SB.subset.model2), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Best Subset model Sqrt")
abline(a=0, b=0, col="red")
#try

plot( fitted(SB.subset.model), rstandard(SB.subset.model), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Best Subset model Sqrt")
abline(a=0, b=0, col="red")

#applying 1/ y-variable
Itdeathrate = 1/(tdeathrate)
IB.subset.model <-  lm(Itdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                       + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)
summary(IB.subset.model)

plot( fitted(IB.subset.model), rstandard(IB.subset.model), xlab = "Predicted", ylab = "Residual", main="Predicted vs residuals plot for Best Subset model Inverse")
abline(a=0, b=0, col="red")

#No improvement in model observed(during residual analysis) after transformation of response variable,
#so we with the untransformed model

####Plot residuals vs each x-variable:

B.subset.model <- lm(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                     + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)

plot(tavgAnnCount, rstandard(B.subset.model), xlab = "tavgAnnCount", ylab = "Residual", main="avgAnnCount vs residuals plot")
abline(a=0, b=0, col="red")
plot(tincidencerate, rstandard(B.subset.model), xlab = "tincidencerate", ylab = "Residual", main="tincidencerate vs residuals plot")
abline(a=0, b=0, col="red")
plot(tpoverty, rstandard(B.subset.model), xlab = "tpoverty", ylab = "Residual", main="poverty vs residuals plot")
abline(a=0, b=0, col="red")
plot(tmedianagefemale, rstandard(B.subset.model), xlab = "tmedianagefemale", ylab = "Residual", main="medianagefemale vs residuals plot")
abline(a=0, b=0, col="red")
plot(tgeo.F, rstandard(B.subset.model), xlab = "tgeo.F", ylab = "Residual", main="geo.f vs residuals plot")
abline(a=0, b=0, col="red")
plot(ths, rstandard(B.subset.model), xlab = "ths", ylab = "Residual", main="hs vs residuals plot")
abline(a=0, b=0, col="red")
plot(temp, rstandard(B.subset.model), xlab = "temp", ylab = "Residual", main="emp vs residuals plot")
abline(a=0, b=0, col="red") 
plot(tunemp, rstandard(B.subset.model), xlab = "tunemp", ylab = "Residual", main="unemp vs residuals plot")
abline(a=0, b=0, col="red")
plot(tgovcoverage, rstandard(B.subset.model), xlab = "tgovcoverage", ylab = "Residual", main="tgovcoverage vs residuals plot")
abline(a=0, b=0, col="red")
plot(tmarriedhouse, rstandard(B.subset.model), xlab = "tmarriedhouse", ylab = "Residual", main="marriedhouse vs residuals plot")
abline(a=0, b=0, col="red")
plot(tbirthrate, rstandard(B.subset.model), xlab = "tbirthrate", ylab = "Residual", main="birthrate vs residuals plot")
abline(a=0, b=0, col="red")

####------Normality test  
##QQ-plot
qqnorm(rstandard(B.subset.model))
qqline(rstandard(B.subset.model), col = 2)

#Shapiro-Wilk normality test
sp.Bsubset = lm(tdeathrate~tavgAnnCount + tincidencerate + tpoverty + tmedianagefemale
                + tgeo.F + ths + temp + tunemp + tgovcoverage + tmarriedhouse + tbirthrate, data=myd)
shapiro.test(residuals(sp.Bsubset))

##--------Checking Influencial data points
inf = influence.measures(B.subset.model)
summary(inf)

## checking multicolinearity for independent variables.
vif(B.subset.model)

####
# correlation among variables in final model
corr.final=cor(cbind(tdeathrate, tavgAnnCount, tincidencerate, tpoverty, tmedianagefemale, 
                     tgeo.F, ths, temp, tgovcoverage, tmarriedhouse, tbirthrate))
write.csv(corr.final, file = "correlation_final_variables.csv",row.names=FALSE)
