# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:37:17 2023

@author: Tim Offermans, FrieslandCampina (tim.offermans@frieslandcampina.com)
"""

import math
import numpy as np
from scipy import sparse

# Subtract per spectrum the minimum of the spectrum as a constant baseline.
def bl_offset(spectra):
    b = np.repeat(np.min(spectra, 1).reshape(spectra.shape[0], 1), spectra.shape[1], 1);
    spectra = spectra - b;
    return spectra

# Subtract per spectrum a polynomial fitted through the spectrum. A degree of 1 
# will for instance fit a straight line. A degree of 0 will subtract the mean 
# spectrum.
def bl_polynomial(spectra, degree):
    x = np.arange(0, spectra.shape[1])+1
    for i in np.arange(0, spectra.shape[0]):
        y = spectra[i, :]
        c = np.polyfit(x, y, degree)
        p = np.poly1d(c)
        b = p(x)
        spectra[i, :] = spectra[i, :] - b; 
    return spectra

# Subtract per spectrum a linear baseline fitted through the start and end of 
# the spectrum. This is not the same as the polynomial fit with a degree of 1, 
# as that will fit a line through all points (not just the start and finish).
def bl_linear(spectra):
    for i in np.arange(0, spectra.shape[0]):
        b = np.linspace(spectra[i, 0], spectra[i, -1], spectra.shape[1])
        spectra[i, :] = spectra[i, :] - b
    return spectra

# Use assymetric least squares smoothing per spectrum to fit its baseline and 
# subtract it from the spectrum.
def bl_asls(spectra, lam=1000000, p=0.001):
    for n in range(spectra.shape[0]):
        y = spectra[n, :]
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(10):
            W.setdiag(w)
            Z = W + D
            z = sparse.linalg.spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        spectra[n, :] = spectra[n, :] - z
    return spectra

# Standard Normal Variate transformation: subtracts, per spectrum, the mean of 
# the spectrum and then divides it by the spectrum standard deviation.
def sc_snv(spectra):
    b = np.repeat(np.mean(spectra, 1).reshape(spectra.shape[0], 1), spectra.shape[1], 1);
    spectra = spectra - b;
    s = np.repeat(np.std(spectra, 1).reshape(spectra.shape[0], 1), spectra.shape[1], 1);
    spectra = spectra / s;
    return spectra

# Multiplicative Scatter Correction, which fits a additive and multiplicative
# baseline between each spectrum and the average of all spectra. Independent 
# testing data should be processed based on the mean of the dependent training 
# data. This function can be used to process testing data alongside the 
# training data.
def sc_msc(spectra_train, spectra_test=None):
    ref = np.mean(spectra_train, 0)
    for i in range(spectra_train.shape[0]):
        p = np.polyfit(ref, spectra_train[i, :], 1)
        spectra_train[i, :] = (spectra_train[i, :] - p[1]) / p[0]
    if np.any(spectra_test!=None):
        for i in range(spectra_test.shape[0]):
            p = np.polyfit(ref, spectra_test[i, :], 1)
            spectra_test[i, :] = (spectra_test[i, :] - p[1]) / p[0]
        return [spectra_train, spectra_test]
    else:
        return spectra_train

# Smooths using a moving mean over a certain window width (also known as a 
# boxcar filter).
def sm_movmean(spectra, width):
    spectra_sm = spectra
    for i in np.arange(0, spectra.shape[1]):
        j = np.arange(i + 1 - math.ceil(width/2), i + math.ceil(width/2))
        k = np.where((j >= 0) * (j <= (spectra.shape[1]-1)))
        s = spectra[:, j[k]]
        spectra_sm[:, i] = np.mean(s, 1)
    return spectra_sm    

# Smooths using a moving median over a certain window width.
def sm_movmedian(spectra, width):
    spectra_sm = spectra
    for i in np.arange(0, spectra.shape[1]):
        j = np.arange(i + 1 - math.ceil(width/2), i + math.ceil(width/2))
        k = np.where((j >= 0) * (j <= (spectra.shape[1]-1)))
        s = spectra[:, j[k]]
        spectra_sm[:, i] = np.median(s, 1)
    return spectra_sm    

# The same as using a moving median, but the spectral variables are weighed 
# over the window according to a Gaussian function. That is, the center of the 
# window has more weight than the edges.
def sm_gaussian(spectra, width):
    spectra_sm = spectra
    w = np.exp(-np.linspace(-np.exp(1), np.exp(1), width)**2)
    for i in np.arange(0, spectra.shape[1]):
        j = np.arange(i + 1 - math.ceil(width/2), i + math.ceil(width/2))
        k = np.where((j >= 0) * (j <= (spectra.shape[1]-1)))
        s = spectra[:, j[k]]
        ws = np.repeat(w[k].reshape(1, -1), spectra.shape[0], 0)
        spectra_sm[:, i] = np.sum(s * ws, 1) / sum(w[k]);
    return spectra_sm    

# Set the mean value of each spectral variable (column) to zero.
def sc_mean(spectra):
    spectra = spectra - np.repeat(np.mean(spectra, 0, keepdims=True), spectra.shape[0], 0)
    return spectra

# Set the standard deviation each spectral variable (column) to one.
def sc_std(spectra):
    spectra = spectra / np.repeat(np.std(spectra, 0, keepdims=True), spectra.shape[0], 0)
    return spectra

# Sets the minimum and maximum of each spectral varirable (column) to zero and 
# one, respectively.
def sc_range(spectra):
    spectra = spectra - np.repeat(np.min(spectra, 0, keepdims=True), spectra.shape[0], 0)
    spectra = spectra / np.repeat(np.max(spectra, 0, keepdims=True), spectra.shape[0], 0)
    return spectra