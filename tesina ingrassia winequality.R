#REPORT STATISTICAL LEARNING CARLA PERRONE

#WHITE WINE QUALITY DATASET


# LIBRARIES:

library(ISLR)

library(labstatR)

library(EnvStats)

library(corrplot)

library(MASS)

library(caret)

library(ROCR)

# install.packages("tree")

library(tree)

# install.packages("randomForest")

library(randomForest)

# install.packages("gbm")

library(gbm)

library(ggplot2)

library(gamlss)

library(gridExtra)

library(tidyverse) # data manipulation and visualization

library(kernlab) # SVM methodology

library(e1071) # SVM methodology

library(ISLR) # contains example data set "Khan"

library(RColorBrewer) # customized coloring of plots

library(mlbench)

library(caret) # for easy machine learning workflow

# install.packages("neuralnet")

library(neuralnet)

library(visdat) # visdat package: at-a-glance ggplot object of what is inside a dataframe

library(caTools) # for data partition into training and test set

# install.packages("ggthemes")

library(ggthemes) # for additional plotting themes

# install.packages("DataExplorer")

library(DataExplorer)


# ------------------ #

#### PREPARE DATA ####

# ------------------ #

data_train = read.csv("winequality_white_train.csv")

str(data_train)

# Analysis of missing values

sum(!complete.cases(data_train)) # number of missing values
plot_missing(data_train) # DataExplorer package: return and plot frequency of missing values for each feature.
vis_dat(data_train) # visdat package: at-a-glance ggplot object of what is inside a dataframe
vis_miss(data_train) # visdat package: at-a-glance ggplot of the missingness inside a dataframe
data_train = data_train[,-1] # remove the first column with the index of the wine

attach(data_train)


# ----------------------------- #

#### 1) DESCRIPTIVE ANALYSIS ####

# ----------------------------- #


#### A) Univariate Analysis ####


# FIXED ACIDITY:

summary(fixed.acidity)

max(fixed.acidity)-min(fixed.acidity) #range

frequencyFixedAcidity <- table(fixed.acidity) # how many time each value is present

frequencyFixedAcidity

length(frequencyFixedAcidity) # number of values

frequencyFixedAcidity/length(frequencyFixedAcidity) #relative frequences

names(which(frequencyFixedAcidity==max(frequencyFixedAcidity))) # mode

sd(fixed.acidity) #standard deviation

labstatR::cv(fixed.acidity) # coefficient of variation. Note that the formula to calculate it is one line below

cv = sd(fixed.acidity)/mean(fixed.acidity)*100

cv

hist(fixed.acidity, freq = F, breaks = 88, main = "Fixed Acidity")

lines(density(fixed.acidity), lwd = 2, col = "blue")

skewness(fixed.acidity)

kurtosis(fixed.acidity)

boxplot(fixed.acidity, main="Boxplot fixed acidity")

# VOLATILE ACIDITY:

summary(volatile.acidity)

max(volatile.acidity)-min(volatile.acidity) #range

frequencyVolatileAcidity <- table(volatile.acidity) 

frequencyVolatileAcidity

length(frequencyVolatileAcidity) 

frequencyVolatileAcidity/length(frequencyVolatileAcidity) #relative frequences

names(which(frequencyVolatileAcidity==max(frequencyVolatileAcidity))) # mode

sd(volatile.acidity)

labstatR::cv(volatile.acidity) 

cv = sd(volatile.acidity)/mean(volatile.acidity)*100

cv

hist(volatile.acidity, freq = F, breaks = 88, main= "Volatile Acidity")

lines(density(volatile.acidity), lwd = 2, col = "blue")

skewness(volatile.acidity)

kurtosis(volatile.acidity)

boxplot(volatile.acidity, main="Boxplot volatile acidity")

# CITRIC ACID:

summary(citric.acid)

max(citric.acid)-min(citric.acid) #range

frequencyCitricAcid <- table(citric.acid) 

frequencyCitricAcid

length(frequencyCitricAcid) 

frequencyCitricAcid/length(frequencyCitricAcid) #relative frequences

names(which(frequencyCitricAcid==max(frequencyCitricAcid))) # mode

sd(citric.acid)

labstatR::cv(citric.acid) 

cv = sd(citric.acid)/mean(citric.acid)*100

cv

hist(citric.acid, freq = F, breaks = 88, main="Citric Acid")

lines(density(citric.acid), lwd = 2, col = "blue")

skewness(citric.acid)

kurtosis(citric.acid)

boxplot(citric.acid, main="Boxplot citric acid")

# RESIDUAL SUGAR:

summary(residual.sugar)

max(residual.sugar)-min(residual.sugar) #range

frequencyResidualSugar <- table(residual.sugar) 

frequencyResidualSugar

length(frequencyResidualSugar) 

names(frequencyResidualSugar)[frequencyResidualSugar == max(frequencyResidualSugar)] 

sd(residual.sugar)

labstatR::cv(residual.sugar) 

cv = sd(residual.sugar)/mean(residual.sugar)*100

cv

hist(residual.sugar, freq = F, breaks = 78, main= "Residual Sugar")

lines(density(residual.sugar), lwd = 2, col = "blue")

skewness(residual.sugar)

kurtosis(residual.sugar)

kurtosis(residual.sugar, excess=FALSE)


boxplot(residual.sugar, main="Boxplot residual sugar")


# CHLORIDES:

summary(chlorides)

max(chlorides)-min(chlorides) #range

frequencyChlorides <- table(chlorides) 

frequencyChlorides

length(frequencyChlorides) 

frequencyChlorides/length(frequencyChlorides) #relative frequences

names(which(frequencyChlorides==max(frequencyChlorides))) # mode

sd(chlorides)

labstatR::cv(chlorides) 

cv = sd(chlorides)/mean(chlorides)*100

cv

hist(chlorides, freq = F, breaks = 78, main="Chlorides")

lines(density(chlorides), lwd = 2, col = "blue")

skewness(chlorides)

kurtosis(chlorides)

boxplot(chlorides, main="Boxplot chlorides")

# FREE SULFUR DIOXIDE:

summary(free.sulfur.dioxide)

max(free.sulfur.dioxide)-min(free.sulfur.dioxide) #range

frequencyFreeSulfurDioxide <- table(free.sulfur.dioxide) 

frequencyFreeSulfurDioxide

length(frequencyFreeSulfurDioxide) 

frequencyFreeSulfurDioxide/length(frequencyFreeSulfurDioxide) #relative frequences

names(which(frequencyFreeSulfurDioxide==max(frequencyFreeSulfurDioxide))) # mode

sd(free.sulfur.dioxide)

labstatR::cv(free.sulfur.dioxide) 

cv = sd(free.sulfur.dioxide)/mean(free.sulfur.dioxide)*100

cv

hist(free.sulfur.dioxide, freq = F, breaks = 52, main="Free Sulfur Dioxide")

lines(density(free.sulfur.dioxide), lwd = 2, col = "blue")

skewness(free.sulfur.dioxide)

kurtosis(free.sulfur.dioxide)

boxplot(free.sulfur.dioxide, main="Boxplot free sulfur dioxide")

# TOTAL SULFUR DIOXIDE:

summary(total.sulfur.dioxide)

max(total.sulfur.dioxide)-min(total.sulfur.dioxide) #range

frequencyTotalSulfurDioxide <- table(total.sulfur.dioxide) 

frequencyTotalSulfurDioxide

length(frequencyTotalSulfurDioxide) 

frequencyTotalSulfurDioxide/length(frequencyTotalSulfurDioxide) #relative frequences

names(which(frequencyTotalSulfurDioxide==max(frequencyTotalSulfurDioxide))) # mode

sd(total.sulfur.dioxide)

labstatR::cv(total.sulfur.dioxide) 

cv = sd(total.sulfur.dioxide)/mean(total.sulfur.dioxide)*100

cv

hist(total.sulfur.dioxide, freq = F, breaks = 135, main="Total Sulfur Dioxide")

lines(density(total.sulfur.dioxide), lwd = 2, col = "blue")

skewness(total.sulfur.dioxide)

kurtosis(total.sulfur.dioxide)

boxplot(total.sulfur.dioxide, main="Boxplot total sulfur dioxide")

# DENSITY:

summary(density)

max(density)-min(density) #range

frequencyDensity <- table(density) 

frequencyDensity

length(frequencyDensity) 

frequencyDensity/length(frequencyDensity) #relative frequences

names(which(frequencyDensity==max(frequencyDensity))) # mode

sd(density)

labstatR::cv(density) 

cv = sd(density)/mean(density)*100

cv

hist(density, freq = F, breaks =352, main="Density")

lines(density(density), lwd = 2, col = "blue")

skewness(density)

kurtosis(density)

boxplot(density, main="Boxplot density")

# pH:

summary(pH)

max(pH)-min(pH) #range

frequencypH <- table(pH) 

frequencypH

length(frequencypH) 

frequencypH/length(frequencypH) #relative frequences

names(which(frequencypH==max(frequencypH))) # mode

sd(pH)

labstatR::cv(pH) 

cv = sd(pH)/mean(pH)*100

cv

hist(pH, freq = F, breaks = 83, main="pH")

lines(density(pH), lwd = 2, col = "blue")

skewness(pH)

kurtosis(pH)

boxplot(pH, main="Boxplot pH")

# SULPHATES:

summary(sulphates)

max(sulphates)-min(sulphates) #range

frequencySulphates <- table(sulphates) 

frequencySulphates

length(frequencySulphates) 

frequencySulphates/length(frequencySulphates) #relative frequences

names(which(frequencySulphates==max(frequencySulphates))) # mode

sd(sulphates)

labstatR::cv(sulphates) 

cv = sd(sulphates)/mean(sulphates)*100

cv

hist(sulphates, freq = F, breaks = 90, main="Sulphates")

lines(density(sulphates), lwd = 2, col = "blue")

skewness(sulphates)

kurtosis(sulphates)

boxplot(sulphates, main="Boxplot sulphates")

# ALCOHOL:

summary(alcohol)

max(alcohol)-min(alcohol) #range

frequencyAlcohol <- table(alcohol) 

frequencyAlcohol

length(frequencyAlcohol) 

frequencyAlcohol/length(frequencyAlcohol) #relative frequences

names(which(frequencyAlcohol==max(frequencyAlcohol))) # mode

sd(alcohol)

labstatR::cv(alcohol) 

cv = sd(alcohol)/mean(alcohol)*100

cv

hist(alcohol, freq = F, breaks = 58, main="Alcohol")

lines(density(alcohol), lwd = 2, col = "blue")

skewness(alcohol)

kurtosis(alcohol)

boxplot(alcohol, main="Boxplot alcohol")

# QUALITY:

summary(quality)

frequencyQuality <- table(quality) 

frequencyQuality

length(frequencyQuality) 

names(frequencyQuality)[frequencyQuality == max(frequencyQuality)] 

sd(quality)

labstatR::cv(quality) 

cv = sd(quality)/mean(quality)*100

cv

skewness(quality)

kurtosis(quality)

kurtosis(quality, excess=FALSE)

table(quality)

round(table(quality)/length(quality)*100,2)

barplot(table(quality), col = "blue", main = "Barplot of Quality", las = 1)



#### B) Bivariate Analysis ####

#install.packages("ggcorrplot")
library(ggcorrplot)

#correlation<-cor(data_train)
#correlation
#dev.new(width=5, height=4)
#corrplot(correlation, method = "number")

classquality <- ifelse(quality < 6, 0, 1) # classquality = 0 if quality = 4 or 5. classquality = 1 if quality = 6 or 7 or 8 
#classquality <- ifelse(quality < 6, "bad", "good")

data_train1 = data.frame(data_train, classquality) # create a dataframe adding to the previous dataset the variable "classquality"
str(data_train1)

correlation1<-cor(data_train1)
ggcorrplot(correlation1, hc.order = TRUE, type = "lower",lab = TRUE)
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(correlation1,
         method="shade", # visualisation method
         shade.col=NA, # colour of shade line
         tl.col="black", # colour of text label
         tl.srt=45, # text label rotation
         col=col(200), # colour of glyphs
         addCoef.col="black", # colour of coefficients
         order="AOE", # ordering method
         number.cex=0.7
)

xyplot( total.sulfur.dioxide ~ seq(1, length(density)) , group=quality, data=data_train1, 
       auto.key=list(space="right"), 
       jitter.x=TRUE, jitter.y=TRUE) 

xyplot( sulphates ~ seq(1, length(sulphates)) , group=classquality, data=data_train1, 
        auto.key=list(space="right"), 
        jitter.x=TRUE, jitter.y=TRUE,xlab="index")



boxplot(fixed.acidity ~ classquality, data = data_train1, col = "lightgreen", ylab = "fixed acidity")
boxplot(volatile.acidity ~ classquality, data = data_train1, col = "lightgreen", ylab = "volatile acidity")
boxplot(citric.acid ~ classquality, data = data_train1, col = "lightgreen", ylab = "citric acid")
boxplot(residual.sugar ~ classquality, data = data_train1, col = "lightgreen", ylab = "residual sugar")
boxplot(chlorides ~ classquality, data = data_train1, col = "lightgreen", ylab = "chlorides")
boxplot(free.sulfur.dioxide ~ classquality, data = data_train1, col = "lightgreen", ylab = "free sulfur dioxide")

boxplot(total.sulfur.dioxide ~ classquality, data = data_train1, col = "lightgreen", ylab = "total sulfur dioxide")
boxplot(density ~ classquality, data = data_train1, col = "lightgreen", ylab = "density")
boxplot(pH ~ classquality, data = data_train1, col = "lightgreen", ylab = "pH")
boxplot(sulphates ~ classquality, data = data_train1, col = "lightgreen", ylab = "sulphates")
boxplot(alcohol ~ classquality, data = data_train1, col = "lightgreen", ylab = "alcohol")


'#
grid.arrange(ggplot(data_train1, aes(fixed.acidity)) + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(volatile.acidity)) + geom_histogram(binwidth = 0.2, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(citric.acid)) + geom_histogram(binwidth = 0.2, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(residual.sugar)) + geom_histogram(binwidth = 8, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(chlorides)) + geom_histogram(binwidth = 0.05, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(free.sulfur.dioxide)) + geom_histogram(binwidth = 20, position ="fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ncol = 2, nrow = 3, top = "Conditional distributions on values of classquality")

grid.arrange(ggplot(data_train1, aes(total.sulfur.dioxide)) + geom_histogram(binwidth = 60, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(density)) + geom_histogram(binwidth = 0.001, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(pH)) + geom_histogram(binwidth = 0.05, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(sulphates)) + geom_histogram(binwidth = 0.03, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(alcohol)) + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ncol = 2, nrow = 3, top = "Conditional distributions on values of classquality")
#'

cor(fixed.acidity,classquality) #except [9.5,10.5] 
ggplot(data_train1, aes(fixed.acidity)) + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5) 
ggplot(data_train1, aes(x =fixed.acidity, fill = factor(classquality)))+geom_histogram(binwidth = 1)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(fixed.acidity)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$fixed.acidity, 1)) #bins intervals

cor(volatile.acidity,classquality) #lower than 0.5
ggplot(data_train1, aes(volatile.acidity)) + geom_histogram(binwidth = 0.2, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x = volatile.acidity, fill = factor(classquality)))+geom_histogram(binwidth = 0.2)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(volatile.acidity)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$volatile.acidity, 0.2)) #bins intervals

cor(citric.acid,classquality) #between 0.1 and 0.5; greater than 0.9
ggplot(data_train1, aes(citric.acid)) + geom_histogram(binwidth = 0.2, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x =citric.acid, fill = factor(classquality)))+geom_histogram(binwidth = 0.2)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(citric.acid)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$citric.acid, 0.2)) #bins intervals

cor(residual.sugar,classquality) #lower than 17.5, greater than 27.5
ggplot(data_train1, aes(residual.sugar)) + geom_histogram(binwidth = 5, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x = residual.sugar, fill = factor(classquality)))+geom_histogram(binwidth = 5)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(residual.sugar)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$residual.sugar, 5)) #bins intervals

cor(chlorides,classquality) #lower than 0.12
ggplot(data_train1, aes(chlorides)) + geom_histogram(binwidth = 0.08, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x = chlorides, fill = factor(classquality)))+geom_histogram(binwidth = 0.08)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(chlorides)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$chlorides, 0.08)) #bins intervals

cor(free.sulfur.dioxide,classquality) #between 7.5 and 112
ggplot(data_train1, aes(free.sulfur.dioxide)) + geom_histogram(binwidth = 15, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x =free.sulfur.dioxide, fill = factor(classquality)))+geom_histogram(binwidth = 15)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(free.sulfur.dioxide)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$free.sulfur.dioxide, 15)) #bins intervals

cor(total.sulfur.dioxide,classquality) #between 50 and 250
ggplot(data_train1, aes(total.sulfur.dioxide)) + geom_histogram(binwidth = 100, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x = total.sulfur.dioxide, fill = factor(classquality)))+geom_histogram(binwidth = 100)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(total.sulfur.dioxide)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$total.sulfur.dioxide, 100)) #bins intervals

cor(density,classquality) #less than 0.995, greater than 1.009
ggplot(data_train1, aes(density)) + geom_histogram(binwidth = 0.002, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x = density, fill = factor(classquality)))+geom_histogram(binwidth = 0.002)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(density)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$density, 0.002)) #bins intervals

cor(pH,classquality) #except between 2.77 and 2.83; expect between 3.62 and 3.68 (I don't know if it's correct)
ggplot(data_train1, aes(pH)) + geom_histogram(binwidth = 0.05, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x =pH, fill = factor(classquality)))+geom_histogram(binwidth = 0.05)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(pH)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$pH, 0.05)) #bins intervals

cor(sulphates,classquality) #except between 0.855 and 0.885 (I don't know if it's correct)
ggplot(data_train1, aes(sulphates)) + geom_histogram(binwidth = 0.03, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x = sulphates, fill = factor(classquality)))+geom_histogram(binwidth = 0.03)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(sulphates)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$sulphates, 0.03)) #bins intervals

cor(alcohol,classquality) #greater than 9.5
ggplot(data_train1, aes(alcohol)) + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(classquality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5)
ggplot(data_train1, aes(x = alcohol, fill = factor(classquality)))+geom_histogram(binwidth = 1)+ scale_fill_discrete(name = "classquality")
ggplot(data_train1, aes(alcohol)) + geom_density(aes(color = factor(classquality)))+scale_color_discrete(name="Qualità")
levels(cut_width(data_train1$alcohol, 1)) #bins intervals

#plot histograms with proportion

grid.arrange(ggplot(data_train1, aes(fixed.acidity)) + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(volatile.acidity)) + geom_histogram(binwidth = 0.2, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(citric.acid)) + geom_histogram(binwidth = 0.2, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(residual.sugar)) + geom_histogram(binwidth = 5, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(chlorides)) + geom_histogram(binwidth = 0.08, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(free.sulfur.dioxide)) + geom_histogram(binwidth = 15, position ="fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ncol = 2, nrow = 3, top = "Conditional distributions on values of classquality")


grid.arrange(ggplot(data_train1, aes(total.sulfur.dioxide)) + geom_histogram(binwidth = 100, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(density)) + geom_histogram(binwidth = 0.002, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(pH)) + geom_histogram(binwidth = 0.05, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(sulphates)) + geom_histogram(binwidth = 0.03, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ggplot(data_train1, aes(alcohol)) + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "classquality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             
             ncol = 2, nrow = 3, top = "Conditional distributions on values of classquality")

#plot histograms without proportion

grid.arrange(ggplot(data_train1, aes(x =fixed.acidity, fill = factor(classquality)))+geom_histogram(binwidth = 1)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x = volatile.acidity, fill = factor(classquality)))+geom_histogram(binwidth = 0.2)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x =citric.acid, fill = factor(classquality)))+geom_histogram(binwidth = 0.2)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x = residual.sugar, fill = factor(classquality)))+geom_histogram(binwidth = 5)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x = chlorides, fill = factor(classquality)))+geom_histogram(binwidth = 0.08)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x =free.sulfur.dioxide, fill = factor(classquality)))+geom_histogram(binwidth = 15)+ scale_fill_discrete(name = "classquality"), 
             
             ncol = 2, nrow = 3, top = "Conditional distributions on values of classquality")


grid.arrange(ggplot(data_train1, aes(x = total.sulfur.dioxide, fill = factor(classquality)))+geom_histogram(binwidth = 100)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x = density, fill = factor(classquality)))+geom_histogram(binwidth = 0.002)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x =pH, fill = factor(classquality)))+geom_histogram(binwidth = 0.05)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x = sulphates, fill = factor(classquality)))+geom_histogram(binwidth = 0.03)+ scale_fill_discrete(name = "classquality"), 
             
             ggplot(data_train1, aes(x = alcohol, fill = factor(classquality)))+geom_histogram(binwidth = 1)+ scale_fill_discrete(name = "classquality"), 
             
             ncol = 2, nrow = 3, top = "Conditional distributions on values of classquality")



# ------------------------------- #

#### 2) MODELING TRAINING DATA ####

# ------------------------------- #




# ---------------------------- #

#### A) Logistic Regression ####

# ---------------------------- #

logiReg = step(glm(classquality~.-quality, data = data_train1, family = binomial), 
            
            direction = "both") # we are using all the variables except quality

summary(logiReg)

glm.probs1 <- predict.glm(logiReg, type = "response") # The predict() function can be used to predict probabilities, 

# given values of the predictors. The type="response" option tells R to output probabilities 

# of the form P(Y = 1|X), as opposed to other information such as the logit.

head(glm.probs1) 



N <- dim(data_train1)[1] 

glm.pred1 <- rep("0", N) # assign "0" (Bad) for each unit with prob < 0.5

glm.pred1[glm.probs1 > 0.5] = "1" # assign "1" (Excellent) for units with P(Y="1")>0.5

glm.pred1 # look at the results



# Confusion matrix:

table(glm.pred1, data_train1$classquality)


confMat1 <- addmargins(table(glm.pred1, data_train1$classquality))

confMat1



delta1 <-(confMat1[1,2]+confMat1[2,1])/N*100 # misclassification error rate

delta1



tpr <- round(confMat1[2,2]/confMat1[3,2]*100, 2) 

tpr



fpr <- round(confMat1[2,1]/confMat1[3,1]*100, 2)

fpr



# ROC curve:

pred.t.LR_T=ROCR::prediction(glm.probs1, classquality) 



# Need to specify ROCR:: because If you have library(neuralnet) open,

# it overrides the "prediction" function in ROCR and generates this error. 

# Double check that neuralnet, or any other package that may use a "prediction" function, 

# are detached.



perf.t.LR_T=performance(pred.t.LR_T, measure = "tpr", x.measure = "fpr") 

plot(perf.t.LR_T,colorize=TRUE,lwd=2, print.cutoffs.at=c(0.2,0.5,0.8)) 

abline(a=0,b=1, lty=2)

perf <- performance(pred.t.LR_T, measure = "auc", x.measure = "fpr")



#AUC:

AUC <- performance(pred.t.LR_T, measure = "auc", x.measure = "fpr")

AUC@y.values[[1]]





#DATA VALIDATION SET:

data_validation = read.csv("winequality_white_validation.csv")

str(data_validation)

data_validation = data_validation[,-1] # remove the first column, with the index of the wine

sum(!complete.cases(data_validation)) # number of NA values = 0

plot_missing(data_validation)

attach(data_validation)

#ValidQuality <- ifelse(data_validation$quality < 6, "bad", "good") 
ValidQuality <- ifelse(data_validation$quality < 6, 0, 1) 

data_validation1 = data.frame(data_validation, ValidQuality) 

str(data_validation1) 

head(data_validation1$ValidQuality, 324) 



glm.probs1val <- predict.glm(logiReg, data_validation1, type = "response")

Nval<-dim(data_validation1)[1] 

Nval
glm.pred1val <- rep("0", Nval) 

glm.pred1val[glm.probs1val > 0.5] = "1" 

glm.pred1val 



#confusion matrix:

table(glm.pred1val, data_validation1$ValidQuality)

confMat1val <- addmargins(table(glm.pred1val, data_validation1$ValidQuality))

confMat1val


delta1val <-(confMat1val[1,2]+confMat1val[2,1])/N*100 

delta1val

tprval <- round(confMat1val[2,2]/confMat1val[3,2]*100, 2) 

tprval



fprval <- round(confMat1val[2,1]/confMat1val[3,1]*100, 2)

fprval


# ---------------------------- #

###  B) Classification trees ###

# ---------------------------- #

# Step 2
# Find the summary of the above classification tree and plot the
# classification tree. Explain your results.
# 

data_train1$classquality<-as.factor(data_train1$classquality)
str(data_train1)
tree.datatrain1=tree(classquality~.-quality,data_train1)


summary(tree.datatrain1) 
# Note also the misclassification error rate
plot(tree.datatrain1)
text(tree.datatrain1, cex=0.75, col="blue")
#
plot(tree.datatrain1)
text(tree.datatrain1,pretty=0, cex=0.75, col="blue")
# The argument pretty=0 instructs R to include the category names for any 
# qualitative predictors, rather than simply displaying a letter for each category.
tree.datatrain1
# for classification trees, yval gives the  majority class 
#
#
# Step 3: 
# Analysis on test set
#
# We split the observations into a training set and a test
# set, build the tree using the training set, and evaluate its performance on
# the test data. The predict() function can be used for this purpose. In the
# case of a classification tree, the argument type="class" instructs R to return
# the actual class prediction. 
set.seed(2)
data_validation = read.csv("winequality_white_validation.csv")

str(data_validation)

data_validation = data_validation[,-1] # remove the first column, with the index of the wine

sum(!complete.cases(data_validation)) # number of NA values = 0

plot_missing(data_validation)

attach(data_validation)

#ValidQuality <- ifelse(data_validation$quality < 6, "bad", "good")
ValidQuality <- ifelse(data_validation$quality < 6, 0,1)

data_validation1 = data.frame(data_validation, ValidQuality) 

str(data_validation1) 

colnames(data_validation1)[13] = "classquality"
str(data_validation1) 
data_validation1$classquality=as.factor(data_validation1$classquality)

tree.pred=predict(tree.datatrain1,data_validation1,type="class")
table(tree.pred,data_validation1$classquality)
confMat.tree=addmargins(table(tree.pred,data_validation1$classquality))
confMat.tree
delta.tree.test=(confMat.tree[1,2]+confMat.tree[2,1])/nL*1
delta.tree.test
accuracy.tree.test=1-delta.tree.test
accuracy.tree.test
#
# This approach leads to correct predictions for
# around 71.5% of the locations in the test data set.
#
# Next, we consider whether pruning the tree might lead to improved
# results. The function cv.tree() performs cross-validation in order to
# cv.tree()
# determine the optimal level of tree complexity; cost complexity pruning
# is used in order to select a sequence of trees for consideration. We use
# the argument FUN=prune.misclass in order to indicate that we want the
# classification error rate to guide the cross-validation and pruning process,
# rather than the default for the cv.tree() function, which is deviance. The
# cv.tree() function reports the number of terminal nodes of each tree considered
# (size) as well as the corresponding error rate and the value of the
# cost-complexity parameter used (k, which corresponds to Î± in (8.4)).
#
#
# Step 4: 
# Cross Validation and Pruning
#
set.seed(3)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)
names(cv.carseats)
cv.carseats
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b",lwd=2,col="blue",
     xlab="Number of terminal nodes", ylab="Deviance" )
plot(cv.carseats$k,cv.carseats$dev,type="b",lwd=2,col="blue",
     xlab="Number of terminal nodes", ylab="Deviance" )
par(mfrow=c(1,1))
opt.nodes<-cv.carseats$size[which.min(cv.carseats$dev)]
opt.nodes
prune.carseats=prune.misclass(tree.carseats,best=opt.nodes)
# Note that, despite the name, dev corresponds to the cross-validation error
# rate in this instance. The tree with 9 terminal nodes results in the lowest
# cross-validation error rate, with 50 cross-validation errors.
plot(prune.carseats)
text(prune.carseats,pretty=0)
#
# Step 5: 
# Apply the pruned tree to the test data
# We now apply the prune.misclass() function in order to prune the tree to 
# obtain the nine-node tree.
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
confMat.tree.pruned=addmargins(table(tree.pred,High.test))
confMat.tree.pruned
delta.tree.pruned.test=(confMat.tree.pruned[1,2]+confMat.tree.pruned[2,1])/nT*1
delta.tree.pruned.test
accuracy.tree.pruned.test=1-delta.tree.pruned.test
accuracy.tree.pruned.test
# Now 77% of the test observations are correctly classified, so not only has
# the pruning process produced a more interpretable tree, but it has also
# improved the classification accuracy.
#
#
# If we increase the value of best, we obtain a larger pruned tree with lower
# classification accuracy:
#
prune.carseats=prune.misclass(tree.carseats,best=15)
plot(prune.carseats)
text(prune.carseats,pretty=0)
# Ho well does this pruned tree perform on the test data set? Once again,
# we apply the predict() function.
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table(tree.pred,High.test)
confMat.tree.pruned=addmargins(table(tree.pred,High.test))
confMat.tree.pruned
delta.tree.pruned.test=(confMat.tree.pruned[1,2]+confMat.tree.pruned[2,1])/nT
delta.tree.pruned.test
accuracy.tree.pruned.test=1-delta.tree.pruned.test
accuracy.tree.pruned.test

#ENSEMBLE METHOD: BAGGING, RANDOM FOREST, BOOSTING

#BAGGING

data_train2 = data_train1[,-c(12)]
rf.1=randomForest(classquality ~ ., data = data_train2, mtry=11,importance=TRUE)
rf.1
varImpPlot(rf.1)
importance(rf.1)
#
# Prediction with bagging
#
pred <- predict(rf.1, newdata = data_validation1)
table(pred, data_validation1$classquality)
confMat.tree.bagging=addmargins(table(pred, data_validation1$classquality))
confMat.tree.bagging
delta.tree.bagging.test=(confMat.tree.bagging[1,2]+confMat.tree.bagging[2,1])/Nval*1
delta.tree.bagging.test
accuracy.tree.bagging.test=1-delta.tree.bagging.test
accuracy.tree.bagging.test

#RANDOM FOREST

data_train2 = data_train1[,-c(12)]
rf.2=randomForest(classquality ~ ., data = data_train2, importance=TRUE)
rf.2
varImpPlot(rf.2)
importance(rf.2)
#
# Prediction with random forest
#
pred2 <- predict(rf.2, newdata = data_validation1)
table(pred2, data_validation1$classquality)
confMat.tree.bagging.2=addmargins(table(pred2, data_validation1$classquality))
confMat.tree.bagging.2
delta.tree.bagging.test.2=(confMat.tree.bagging.2[1,2]+confMat.tree.bagging.2[2,1])/Nval*1
delta.tree.bagging.test.2
accuracy.tree.bagging.test.2=1-delta.tree.bagging.test.2
accuracy.tree.bagging.test.2


#BOOSTING
#classquality deve essere di tipo numerico cioè 0, 1

wine.boost <- gbm(classquality ~ . - quality, data = data_train1, distribution = 'bernoulli', n.trees = 5000)
par(mar=c(3,14,1,1))
summary(wine.boost,method = relative.influence, las = 1)
#
# Prediction with boosting training dataset
#           
pred3 <- predict.gbm(wine.boost, newdata = data_train1,n.trees=5000,type="response")
for(i in 1:length(pred3)) {
        if (pred3[i] > 0.5){
                pred3[i]=1
        } else {
                pred3[i]=0
        }
}
    
table(pred3, data_train1$classquality)
confMat.tree.boosting.3=addmargins(table(pred3, data_train1$classquality))
confMat.tree.boosting.3
delta.tree.boosting.test.3=(confMat.tree.boosting.3[1,2]+confMat.tree.boosting.3[2,1])/N*1
delta.tree.boosting.test.3
accuracy.tree.boosting.test.3=1-delta.tree.boosting.test.3
accuracy.tree.boosting.test.3

#
# Prediction with boosting validation dataset
#

pred4 <- predict.gbm(wine.boost, newdata = data_validation1,n.trees=5000,type="response")
for(i in 1:length(pred4)) {
        if (pred4[i] > 0.5){
                pred4[i]=1
        } else {
                pred4[i]=0
        }
}

table(pred4, data_validation1$classquality)
confMat.tree.boosting.4=addmargins(table(pred4, data_validation1$classquality))
confMat.tree.boosting.4
delta.tree.boosting.test.4=(confMat.tree.boosting.4[1,2]+confMat.tree.boosting.4[2,1])/Nval*1
delta.tree.boosting.test.4
accuracy.tree.boosting.test.4=1-delta.tree.boosting.test.4
accuracy.tree.boosting.test.4

# -------------------------- #

#### C) Neural Networks #####

# -------------------------- #



# To avoid this error: Error in if (ncol.matrix < rep) { : argument is of length zero

# or other similar errors, DO THIS:



#install.packages("devtools")

library(devtools)

# devtools::install_github("bips-hb/neuralnet")

library(neuralnet)

data_train_nn <- data_train1

str(data_train_nn)


data_validation_nn <- data_validation1

str(data_validation_nn)


set.seed(100)

NN = c()

names_nn = c()

error_nn = c()


tab_nn <- vector(mode = "list", length = 5)

for (i in 1:5){
        
        names_nn = c(names_nn, paste("nn", i, sep = ""))
        
}

#i=1 



# 1 hidden layer with 1 to 5 number of neurons:



for (i in 1:5){
        
        cat(paste("Computing the nn", i, "of 5 ..."), "\n" )
        
        name = names_nn[i]
        
        nn_i = neuralnet(classquality~fixed.acidity + volatile.acidity + 
                                 
                                 residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                                 
                                 density + sulphates + alcohol, data = data_train_nn, hidden = i+2, threshold = 0.1, stepmax = 1e7, linear.output = F, rep = 3)
        
        #i increased stepmax from the default value 1e5, to 1e7, because stepmax is the maximal count of
        
        #all gradient steps, and using value 1e5 gives the following warning:
        
        #Warning message:
        
        #Algorithm did not converge in 3 of 3 repetition(s) within the stepmax.
        
        NN[[name]] = nn_i
        
        i_best = which.min(nn_i$result.matrix[1,])
        
        yhat = round(nn_i$net.result[[i_best]])
        
        pred.nn = round(predict(nn_i, newdata = data_validation_nn, rep = i_best))
        
        tab_nn_i= addmargins(table(pred.nn, data_validation_nn$classquality))
        
        tab_nn[[i]] <- tab_nn_i
                
        error_nn_i = (tab_nn_i[1,2]+tab_nn_i[2,1])/tab_nn_i[3,3]
        
        error_nn = c(error_nn, error_nn_i)
        
}



dnn = data.frame(model = names_nn, hidden = c(3:7), misc.error = error_nn)

bmnn = dnn[which.min(dnn$misc.error), ]

bmnn$misc.error = bmnn$misc.error*100

names(bmnn) = c("Best nn model", "neurons", "Misclassification error on validation set (%)")

bmnn

plot(NN$nn4)

tab_nn[4]



# --------------------------------- #

#### 4) PREDICT TARGET VALUES #####

# --------------------------------- #

#As bagging and random forest models are equivalent, we do the test considering both of them

best_model_1 <- rf.1 #bagging model
best_model_2 <- rf.2 #random forest model

data_test <- read.csv("winequality_white_test.csv")

str(data_test)

data_test = data_test[,-c(1,13)] #delete the first and the last column

str(data_test)

pred1 <- predict(best_model_1, newdata=data_test,type="class")
pred2 <- predict(best_model_2, newdata=data_test,type="class")


N <- dim(data_test)[1] 


table(pred1)
table(pred2)


data_test_predicted_1 = data.frame(data_test, predicted_qualities_1=pred1)
data_test_predicted_2 = data.frame(data_test, predicted_qualities_2=pred2)

write.csv(data_test_predicted_1, "data_test_predicted_1.csv")
write.csv(data_test_predicted_2, "data_test_predicted_2.csv")


#per vedere la differenza tra i due modelli ovvero la diversa classificazione
pred1 <- as.numeric(levels(pred1))[pred1]
pred1

pred2 <- as.numeric(levels(pred2))[pred2]
pred2

table(pred1,pred2)
addmargins(table(pred1,pred2))
