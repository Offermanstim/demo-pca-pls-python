# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:13:31 2023

@author: Tim Offermans (tim.offermans@frieslandcampina.com)
"""

#%% INTRODUCTION
# This script offers a basic demonstration of how to use Principal Component 
# Analysis (PCA) and Partial Least Squares regression in Python. It also shows 
# the basics behind spectral data preprocessing and model validation.

# Import required packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess as pp
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import PLSRegression
from sklearn import model_selection

#%% PART I: PRINCIPAL COMPONENT ANALYSIS (PCA)

# We will start the exercises with an exploratory analysis of the 'Wine-data'. 
# This dataset is often used in data analysis courses, and was originally 
# measured to investigate whether certain Italian wines can be classified 
# according to their region of origin, using chemical properties measured on 
# them. This would allow wine merchants to better guarantee the quality of 
# their wines to their sellers.

# The Wine-data is saved as a matrix in the csv-file 'wine.csv', which  can be 
# loaded as follows:
wine = pd.read_csv('data_wine.csv')

# In chemometrics, it is the convention to save the samples as rows of the data 
# matrix and variables as the columns of the data matrix. The Wine dataset thus 
# contains 178 samples and 14 variables, which can be found respectively with 
# the following two codes:
wine.shape[0]
wine.shape[1]

# In Python, you can also view the first 10 rows (samples) of data matrix for 
# inspection in the command window with the following command:
wine.head(10)

# Note that (in Python), the first element has index 0 and not 1. Also note 
# that the first variable (column) gives the region of origin (cultivar) of the 
# sample. The number of actual chemical properties measured for each of the 178 
# wines is thus 13.

# If you view and inspect the data matrix carefully, you will find that there 
# are three different cultivars or regions. These are named 'Barolo', 
# 'Grignolino' and 'Barbera'. To check this, you can also list the different 
# cultivars by asking Python to extract the unique elements of the first 
# column in the data matrix:
np.unique(wine.iloc[:, 0])

# You can also index the column by name:
np.unique(wine.Cultivar)

# It is always a good idea to inspect the scales of the different variables, 
# that is, to inspect how large in values they typically are. You can of 
# course do this by eye by viewing the matrix as done above. However, you can 
# also let Python calculate the mean values of each column with the following 
# code. Note that the means are only calculated for the columns with chemical 
# measurements:
np.mean(wine.iloc[:,1:14])

# From these values, you can clearly see that Proline is the variable with the 
# highest mean value. To do a more careful inspection, you could also make a 
# boxplot for each (chemical) variable with the following code. This gives you 
# information about the median, the first and third quartiles, and on outliers:
plt.boxplot(wine.iloc[:, 1:14], labels=list(wine.columns[1:14]));
plt.title('Box-plots on Wine data');
plt.ylabel('Value');
plt.xticks(rotation=90);
plt.show();

# From a visual inspection, you can see that all cells in the data matrix are 
# filled with values, and that ther are thus no missing values (holes) in the 
# data. You can also check if there are no missing values, also known as 'NaN'
# (Not a Number) in the entire data matrix. Note that for this, we have to take 
# the numerical part of the (pandas) DataFrame and transform it to a (numpy) 
# data array.
any(np.isnan(wine.iloc[:, 1:].to_numpy().flatten()))

# Now that we understand the data matrix, we can inspect is using Principal 
# Component Analysis (PCA). We will use mean-centering but not scaling, and we 
# will model 13 components (which is the maximum):
X = wine.iloc[:, 1:]
model = PCA(n_components=13)
model.fit(X)

# To decide how many principal components of our model we should take into 
# account for analysis, we can look at the cumulative percentage of variance 
# of the original data that is explained by the principal components. This 
# plot can be made as follows:
plt.plot(model.explained_variance_ratio_)
plt.title('PCA explained variance plot');
plt.xlabel('Number of principal components')
plt.ylabel('Explained variance (%)');
plt.show();

# You can see here that the first principal component already describe 99% of 
# the of the original data. Based on this alone, it would seem that little more 
# of the data can be described by looking at more than one principal component. 
# Despite this, we will investigate the scores of the original samples on the 
# first two principal components using a score-plot.
scores = model.fit_transform(X)
plt.scatter(scores[wine.Cultivar == 'Barolo', 0], scores[wine.Cultivar == 'Barolo', 1], color='r')
plt.scatter(scores[wine.Cultivar == 'Grignolino', 0], scores[wine.Cultivar == 'Grignolino', 1], color='g')
plt.scatter(scores[wine.Cultivar == 'Barbera', 0], scores[wine.Cultivar == 'Barbera', 1], color='b')
plt.legend(['Barolo', 'Grignolino', 'Barbera'])
plt.title('PCA score-plot')
plt.xlabel('PC1 (' + str(round(model.explained_variance_ratio_[0]*100, 2)) + '%)')
plt.ylabel('PC2 (' + str(round(model.explained_variance_ratio_[1]*100, 2)) + '%)')
plt.show()

# The samples (wines) in this score-plot are colored based on the cultivar of 
# the wine. However, we can see that we cannot tell the three wine cultivars 
# completely apart based on the measured data, as the three clusters in the 
# score-plot show overlap. In fact, we cannot even completely separate two of 
# the three cultivars, although Barolo and Grignolino seem to differ the most.

#To get a better understanding of the model we should investigate the loadings, 
# which are the contributions of the original variables (properties) measured 
# for the wines on the newly defined principal components. We can superimpose 
# these quite easily on the score-plot to get a so-called biplot:
loadings = model.components_.T
for i in np.arange(0, len(loadings)):
    loadings[:, i] = loadings[:, i] / max(abs(loadings[:, i]))
    loadings[:, i] = loadings[:, i] * min(np.abs([min(scores[:, i]), max(scores[:, i])]))
plt.scatter(scores[wine.Cultivar == 'Barolo', 0], scores[wine.Cultivar == 'Barolo', 1], color='r')
plt.scatter(scores[wine.Cultivar == 'Grignolino', 0], scores[wine.Cultivar == 'Grignolino', 1], color='g')
plt.scatter(scores[wine.Cultivar == 'Barbera', 0], scores[wine.Cultivar == 'Barbera', 1], color='b')
plt.legend(['Barolo', 'Grignolino', 'Barbera'])
for i in np.arange(0, len(loadings)):
    plt.text(loadings[i, 0], loadings[i, 1], wine.columns[i+1], horizontalalignment='center', verticalalignment='center')
plt.title('PCA score-plot')
plt.xlabel('PC1 (' + str(round(model.explained_variance_ratio_[0]*100, 2)) + '%)')
plt.ylabel('PC2 (' + str(round(model.explained_variance_ratio_[1]*100, 2)) + '%)')
plt.show()

# We can now clearly see that there is quite some difference between how much 
# each variable contributes to the model, as the arrows (loadings) are really 
# different in length. Proline seems to completely dominate PC1, while 
# Magnesium completely dominates PC2.

# This however makes a lot of sense, as we've seen earlier that Proline is 
# measured on a much higher scale than the other properties. It therefore 
# has a much higher variance, and since PCA just tries to explain variance, 
# it completely focuses on Proline and almost just copies it into PC1. This 
# is also why PC1 explains so much variance, because Proline just makes up 
# for most of the variance in the original data. Magnesium is the second-
# highest in scale, and therefore makes up PC2. The worse part is that we 
# are still not really doing multivariate analysis: we are just plotting 
# Magnesium against Proline.

# We can try to solve this by autoscaling the data. This will unify the 
# variances of the different properties, and gives them equal opportunity to 
# contribute to the PCA-model:
X_as = scale(X, axis=0, with_mean=True, with_std=True)
model.fit(X_as)

# To see if autoscaling improved the separation of the cultivars, we have to 
# remake the score-plot:
scores = model.fit_transform(X_as)
plt.scatter(scores[wine.Cultivar == 'Barolo', 0], scores[wine.Cultivar == 'Barolo', 1], color='r')
plt.scatter(scores[wine.Cultivar == 'Grignolino', 0], scores[wine.Cultivar == 'Grignolino', 1], color='g')
plt.scatter(scores[wine.Cultivar == 'Barbera', 0], scores[wine.Cultivar == 'Barbera', 1], color='b')
plt.legend(['Barolo', 'Grignolino', 'Barbera'])
plt.title('PCA score-plot')
plt.xlabel('PC1 (' + str(round(model.explained_variance_ratio_[0]*100, 2)) + '%)')
plt.ylabel('PC2 (' + str(round(model.explained_variance_ratio_[1]*100, 2)) + '%)')
plt.show()

# There is still a little bit of overlap, but you can see that the separation 
# of the three cultivars improved a lot. You can also see that PC1 now explains 
# just 36.2% of the variance and PC2 19.2%, which shows that we have a better 
# balanced model. It is also interesting to see that to separate the three 
# cultivars, we really need both PC1 and PC2. PC1 separates Barolo from 
# Barbera, while PC2 separates Grignolino from the other two.

# To investigate the loadings for the new model, we can remake the biplot:
loadings = model.components_.T
for i in np.arange(0, len(loadings)):
    loadings[:, i] = loadings[:, i] / max(abs(loadings[:, i]))
    loadings[:, i] = loadings[:, i] * min(np.abs([min(scores[:, i]), max(scores[:, i])]))
plt.scatter(scores[wine.Cultivar == 'Barolo', 0], scores[wine.Cultivar == 'Barolo', 1], color='r')
plt.scatter(scores[wine.Cultivar == 'Grignolino', 0], scores[wine.Cultivar == 'Grignolino', 1], color='g')
plt.scatter(scores[wine.Cultivar == 'Barbera', 0], scores[wine.Cultivar == 'Barbera', 1], color='b')
plt.legend(['Barolo', 'Grignolino', 'Barbera'])
for i in np.arange(0, len(loadings)):
    plt.text(loadings[i, 0], loadings[i, 1], wine.columns[i+1], horizontalalignment='center', verticalalignment='center')
plt.title('PCA score-plot')
plt.xlabel('PC1 (' + str(round(model.explained_variance_ratio_[0]*100, 2)) + '%)')
plt.ylabel('PC2 (' + str(round(model.explained_variance_ratio_[1]*100, 2)) + '%)')
plt.show()

# We can now see that all the arrows are much more equal in length, showing 
# that all are contributing to the model, and also to the separation of the 
# cultivars. This model is also more informative about the measure data. It 
# tells us for instance that from all variables, the Nonflavonoid Phenols 
# are most related to the Alkalinity, as its arrowhead is closest to that of 
# the Alkalinity. It also tells us that from all cultivars, the Barbera has 
# on average the highest concentration of Malic Acid, as its clusters is 
# most in the direction that the loading of Malic Acid is pointing towards.

# We will end this first part of the demonstration by removing the data and 
# model from the Matlab workspace. It is good practice to clean your stuff 
# after you are done.
del(i, loadings, model, scores, wine, X, X_as)

#%% PART II: PARTIAL LEAST SQUARS (PLS)

# In this second part, we will work with data measured on Wheat kernels. Near-
# infrared transmission (NIT) spectra were measured for these kernels, as well 
# as the protein content using traditional wet chemical laboratory analysis. 
# The goal of the study is to predict the protein content of the kernels based 
# on the spectroscopic data using multivariate regression. This would be a 
# better method for protein content determination, as NIT spectroscopy is 
# easier, quicker and cheaper to perform than the traditional analysis.

# We will first load the data as CSV-file, which is originally from 
# http://models.life.ku.dk/wheat_kernels, the website of Rasmus Bro's 
# chemometrics group at the University of Copenhagen. You might want to check 
# out this website, as it contains very interesting data, software and 
# literature regarding chemometrics.
wheat = pd.read_csv('data_wheat.csv')

# This data matrix contains 523 samples (rows) and 101 variables (columns). The 
# first variable is the protein content, the remaining 100 are spectroscopic. 
# The range of the spectra can be read from the variable names (column 
# headers), and is 850 to 1048 nm. Note that the unit (nm) is not given in the 
# data.

# We can investigate the spectra by plotting them as line plots with the 
# following code:
wavelengths = []
for i in np.arange(wheat.shape[1]-1):
    wavelengths.append(int(wheat.columns[i+1].replace('_nm', '')))
wavelengths = np.array(wavelengths)
protein = wheat.iloc[:,0].to_numpy()
nit = wheat.iloc[:,1:].to_numpy()
plt.plot(wavelengths, np.transpose(nit))
plt.title('Wheat NIT data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.show()

# We can see that the spectral data shows more overall trends instead of sharp 
# peaks, which is rather typical for spectra measured in this near-infrared 
# domain. Also remarkable is the high variation in the overall baseline of the 
# spectra (they are 'smeared out' along the vertical axis).

# Having quickly investigated the spectra, we will now calibrate a Partial 
# Least Squares regression model that attempts to predict the protein content 
# from the NIT data. We call this 'regressing the protein content on the NIT 
# data'. The NIT data is thus the independent, predictor or X data, while the 
# protein data is the dependent, response or Y data. These terms are often used 
# in regression analysis literature.
model = PLSRegression(n_components=5, scale=False)
model.fit(nit, protein)

# We use up to five latent variables (components) for this model. The data is 
# always automatically mean-centered by the software. Scaling is not required 
# for this dataset (because all variables are in the same unit), but we could 
# do it by adding an input argument at the back so that it ends like: 
#'..., data=plsdata, scale=TRUE)'.

# We will start inspecting how well the PLS model can predict the protein 
# values from the NIT spectra, by plotting the predicted protein values against 
# the actual (reference) protein values of the samples:
plt.scatter(protein, model.predict(nit))
plt.xlabel('Reference')
plt.ylabel('Prediction')
plt.title('PLS model')
plt.show()

# For a model that has perfect prediction accuracy, the predicted and reference 
# values would be identical, and all the samples would in this plot lie on a 
# perfectly straight line. For our model, we can see a general upward trend, so 
# the model does pick up something, but it is far from a sharp, straight line 
# and so the prediction accuracy is not very good. You can also see an overall 
# curve-like shape for this plot, which in this case indicates that the model 
# is more likely to underestimate the protein values for samples of which the 
# actual protein value is higher.

# To quantify the prediction performance as a number, we can calculate ask for 
# the root mean squared error (RMSE) of the predictions:
rmse = np.mean((model.predict(nit) - protein)**2)**0.5

# You can sort-of interpret this number as a average prediction errors. It is 
# also in the same scale as the original protein values, so you should 
# interpret them taking into account the average values of the reference 
# protein values:
np.mean(protein)

# If you try to increase the number of latent variables modelled, you will see 
# that this RMSE becomes lower and thus that the prediction performance 
# increases. You might even consider to model many more latent variables, but 
# more about that in the bonus-round of this tutorial on validation.

# To estimate the prediction performance in a way that is scale-independent, we 
# can calculate the Pearson correlation between prediction and reference (R). 
# This value should be 1 for a perfect model, and is obtained as follows:
R = np.min(np.corrcoef(protein, np.transpose(model.predict(nit))))
R

# In regression literature, this value of R is often squared to give R2. This 
# value then represents the fraction of variance in the protein data that is 
# explained by the PLS model. This value should also be 1 (100%) for a perfect 
# model. In our case, the model explains only 56% of the variance, this is not 
# very high:
R**2

# So, the basic conclusion is that the PLS model captures some of the 
# properties of the NIT data that are predictive of the protein values, but not 
# a lot. We can however try to improve this by pre-processing the NIT data. 
# Spectroscopic data often suffers from baseline and scattering artefacts, 
# especially when measured in the near-infrared domain. These effects are not 
# related to the chemistry but do influence the spectra. Removing these effects 
# using dedicated data pre-processing methods often increases the performance 
# of prediction models calibrated on the data.

# We will preprocess the NIT data first by subtracting a baseline offset. This 
# corrects for additive baseline artefacts. Several methods are available for 
# such baseline subtraction. Here, we use Asymmetric Least Squares here (ALS), 
# as it generally performs well for most datasets. Essentially, this procedure 
# estimates the baseline for each individual NIT-spectrum, and subtracts that 
# from the spectrum.
nit = pp.bl_asls(nit)

# The baseline-corrected spectra can then be plotted as follows. You can see 
# that a large part of the offsets (the 'smearing our' over the vertical axis) 
# are now removed.
plt.plot(wavelengths, np.transpose(nit))
plt.title('Baseline-corrected wheat NIT data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.show()

# However, you can still see some of this 'smearing' is still left, especially 
# at the large peak around 1000 nm. These are caused by multiplicative scatter 
# effects, and can be removed using scatter correction methods. Several are 
# available, but we use Multiplicative Scatter Correction (MSC) as this methods 
# has good overall performance.
nit = pp.sc_msc(nit)

# When plotting the resulting pre-processed spectra, you can see that the 
# scatter effects are largely removed:
plt.plot(wavelengths, np.transpose(nit))
plt.title('Baseline- and scatter-corrected wheat NIT data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.show()

# Now that we have pre-processed the wheat spectra, we can try to recalibrate a 
# PLS model to see if the predictive performance is increased:
model = PLSRegression(n_components=5, scale=False)
model.fit(nit, protein)

# The plot showing the predicted protein values against the actual protein 
# values improved a lot: the samples are more one a tight line. The 'bend' 
# that we saw earlier is also gone, so the model no longer constructively 
# underestimates samples with high protein values.
plt.scatter(protein, model.predict(nit))
plt.xlabel('Reference')
plt.ylabel('Prediction')
plt.title('PLS model on preprocessed data')
plt.show()

# The RMSE for the model also improved, as it is lower:
rmse = np.mean((model.predict(nit) - protein)**2)**0.5
np.mean(protein)

# Both the correlation between prediction and reference protein (R) and the 
# explained variance in the protein values (R2) also improved, as they are 
# higher:
R = np.min(np.corrcoef(protein, np.transpose(model.predict(nit))))
R
R**2

# This shows that the baseline and scattering artefacts that were present in 
# the NIT data are indeed not predictive for the protein values, and that 
# removing them can improve a statistical model predicting the protein values 
# from the NIT data.

# You might be wondering which regions of the spectra are most predictive of 
# the protein values. We can get an idea of this by plotting the regression 
# vector of the PLS model. These numbers essentially tell us how much each 
# of the spectral variables contribute to the protein value prediction. They 
# can be conveniently plotted as follows.
plt.plot(wavelengths, model.coef_)
plt.title('PLS model on preprocessed data')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Regression coefficient')
plt.show()

# Unfortunately, there is not really a single sharp peak present, so we cannot 
# really identify a single wavelength that are corresponding to 
# the protein content. Instead, it seems to be the overall shape that is 
# predictive.

#%% BONUS ROUND: MODEL VALIDATION

# The models fitted above were not subjected to model validation, so we do not 
# know if they are overfitting the data. Overfitting means that the models 
# focus too much on the calibration data, and start to describe patterns that 
# are specific for the samples in the dataset and not for the total population 
# of the data in general. Such an overfitted model will predict very poorly for 
# new data, what is what we would want to use it for. For PLS, this can happen 
# when the model becomes too complicated and uses too many latent variables. 
# Using fewer latent variables will then reduce the performance on the 
# calibration data, but will increase the performance on new data, which is 
# what we want to achieve.

# To check if a model is overfitting, we can use model validation. We split our 
# samples in two sets: a calibration and validation set. The calibration set is 
# used to calibrate the model, which is then used to predict the validation 
# set. This set mimicks the unseen data. Because we have the to-be-predicted 
# data present for this validation set (the protein levels in our case), we can 
# check if the model still works for unseen data.

# To validate the latest model, calibrated on preprocessed spectra and using
# five latent variables, we can use the following code. There are multiple
# methods by which the data can be split in calibration and validation sets.
# We will use leave-one-out cross-validation. For more details on these 
# methods, please see the reference below:
results = model_selection.cross_validate(model, nit, protein, cv=model_selection.LeaveOneOut(), return_train_score=True, scoring='neg_root_mean_squared_error')

# With this code, Python returns the negative RMSE for each validation fold. We 
# can obtain the RMSE over all validation folds by calculating the RMSE of the 
# RMSEs. This only holds because the number of calibration and validation 
# samples is the same for each fold (when using leave-one-out cross-
# validation):
rmse_cal = np.mean(results.get('train_score')**2)**0.5
rmse_cal
rmse_val = np.mean(results.get('test_score')**2)**0.5
rmse_val

# You can see that the RMSE of the validation set is slightly worse than that 
# of the calibration set. This is normal. To check how many latent variables we 
# can use before the model starts to overfit, we should calibrate and validate 
# models for each number of latent variables and save the RMSE values. We can 
# do this as follows, where we consider up to 20 latent variables:
rmse_cal = []
rmse_val = []
for lv in np.arange(20)+1:
    print('Testing PLS model with ' + str(lv) + ' latent variable(s)...')
    model = PLSRegression(n_components=lv, scale=False)
    results = model_selection.cross_validate(model, nit, protein, cv=model_selection.LeaveOneOut(), return_train_score=True, scoring='neg_root_mean_squared_error')
    rmse_cal.append(np.mean(results.get('train_score')**2)**0.5)
    rmse_val.append(np.mean(results.get('test_score')**2)**0.5)

# Let's put these results in a figure:
plt.plot(rmse_cal)
plt.plot(rmse_val)
plt.legend(['Calibration', 'Validation'])
plt.title('PLS optimization')
plt.xlabel('#LVs')
plt.ylabel('RMSE')
plt.xticks(ticks=np.arange(20)+1)
plt.show()

# You can see that the calibration performance keeps improving when we add 
# latent variables. The validation performance however only keeps improving up 
# until we add the 12th latent variable. You can see that the validated error 
# slightly increases when add the 12th latent variable. This is thus were the 
# overfitting starts, and the optimal number of latent variables to include is 
# therefore 11.

# Technically, we now have still used our independent validation data to train 
# the model. We have used it for optimizing the number of latent variables, 
# which is part of model calibration. Officially, we should have had kept yet 
# another, completely independent test set apart. By applying the model with 
# 11 latent variables on that data, we can truly estimate the independent 
# performance of the optimized model. We then make sure that the selection of 
# the number of latent variables is also not overfitted. This is called a 
# double validation. A short demonstration of how to do this in Python is 
# given below:

# This code takes out 20% of the samples as independent test set.
nit_train, nit_test, protein_train, protein_test = model_selection.train_test_split(nit, protein, test_size=0.2)

# This is done at random, but to guarantee reproducibility of the analysis the 
# random number generator can be fixed so that it always returns the same 
# (pseudo)-random split:
nit_train, nit_test, protein_train, protein_test = model_selection.train_test_split(nit, protein, test_size=0.2, random_state=42)

# Now we will repeat the optimization of the number of latent variables by 
# using cross-validation on the training data. Note that we are using 5-fold 
# random cross-validation to save time, and are again fixing the random split 
# to guarentee reproducibility:
rmse_cal = []
rmse_opt = []
for lv in np.arange(20)+1:
    model = PLSRegression(n_components=lv, scale=False)
    results = model_selection.cross_validate(model, nit_train, protein_train, cv=5, return_train_score=True, scoring='neg_root_mean_squared_error')
    rmse_cal.append(np.mean(results.get('train_score')**2)**0.5)
    rmse_opt.append(np.mean(results.get('test_score')**2)**0.5)

# The optimal number of latent variables is that where the optimization RMSE is 
# minimal (this should be 11):
lv = np.argmin(rmse_opt)+1
lv

# To get the independent testing performance, we will recalibrate the PLS model
# using 11 latent variables, using the training data:
model = PLSRegression(n_components=lv, scale=False)
model.fit(nit_train, protein_train)

# Next, we will predict the testing data and calculate the RMSE, R, R2 and 
# show the prediction versus reference plot:
rmse = np.mean((model.predict(nit_test) - protein_test)**2)**0.5
rmse

R = np.min(np.corrcoef(protein_test, np.transpose(model.predict(nit_test))))
R
R**2
    
plt.scatter(protein_test, model.predict(nit_test))
plt.xlabel('Reference')
plt.ylabel('Prediction')
plt.title('PLS model')
plt.show()

# Note that this (truly) external testing can also be done in a cross-
# validation fashion, in which case two nested cross-validation loops are used. 
# This is called double cross-validation, and can be computationally very 
# intensive.

# Another note on terminology: 'test' data is truly independent, 'train' data 
# is the dependent data that is not part of 'test', and is further split into 
# 'calibration' data for calculating the model and 'optimization' for 
# optimizing the number of latent variables. Different people and fields may 
# use different terms. Especially using '(cross-)validation' instead of 
# 'optimization' is common.

# This is the end of the tutorial, all that remains is to clean our Python 
# environment again. I do hope that you have learned something new about 
# chemometrics today. If you want to delve deeper in the material covered in 
# this tutorial, I strongly recommend the literature listed at the end of 
# this script.
del(i, lv, model, nit, nit_test, nit_train, protein, protein_test, protein_train, R, results, rmse, rmse_cal, rmse_opt, rmse_val, wavelengths, wheat)

#%% RECOMMENDED LITERATURE:

# - PCA: Bro, Rasmus, and Age K. Smilde. "Principal component analysis." Analytical methods 6.9 (2014): 2812-2831.
# - PLS: Geladi, Paul, and Bruce R. Kowalski. "Partial least-squares regression: a tutorial." Analytica chimica acta 185 (1986): 1-17.
# - Pre-processing: Engel, Jasper, et al. "Breaking with trends in pre-processing?." TrAC Trends in Analytical Chemistry 50 (2013): 96-106.
# - Validation: Westad, Frank, and Federico Marini. "Validation of chemometric models-a tutorial." Analytica Chimica Acta 893 (2015): 14-24.