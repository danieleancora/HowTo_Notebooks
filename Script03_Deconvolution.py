#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:37:32 2021

@author: danieleancora
"""

# %% IMPORT USEFUL LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import filters

from scipy import signal


# %% DECONVOLUTION BY SPECTRAL DIVISION
# we load a standard image
camera = data.shepp_logan_phantom()
camera = data.camera()

# intensity normalization
camera = np.float64(camera[1:,1:])
camera /= camera.max()


# kernel size and properties
kernelsize = camera.shape[0]
center = np.uint32(np.floor(kernelsize/2.))

# ideal system response
delta = np.zeros([kernelsize,kernelsize])
delta[center,center] = 1

# Gaussian optical response
psf = filters.gaussian(delta, sigma=5)


# we filter the image in Fourier domain
camera_blurred = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(camera)*np.fft.fft2(psf))))


# function to compute the spectral division
def spectraldivision(blurred, psf):
    F_blurred = np.fft.fft2(blurred)
    F_psf = np.fft.fft2(psf)
    
    F_spectraldivision = F_blurred / F_psf
    deblurred = np.abs(np.fft.fftshift(np.fft.ifft2(F_spectraldivision)))
    
    return deblurred

# we compute the spectral division to deconvolve the image
camera_spectraldiv = spectraldivision(camera_blurred, psf)

# plotting the results
plt.figure(figsize=[10,10])
plt.subplot(131), plt.imshow(camera), plt.axis('off'), plt.title('Original image')
plt.subplot(132), plt.imshow(camera_blurred), plt.axis('off'), plt.title('Blurred Image')
plt.subplot(133), plt.imshow(camera_spectraldiv), plt.axis('off'), plt.title('Spectral division')
plt.tight_layout()


# %% THE ROLE OF THE NOISE
# make random noise
alpha = 1e-11
noise = np.random.rand(camera.shape[0], camera.shape[1])
noise = np.random.poisson(lam=1., size=np.shape(camera))


# add it to the blurred object
camera_blurred_noisy = camera_blurred + alpha * noise

# we compute the spectral division to deconvolve the image with noise...
camera_spectraldiv = spectraldivision(camera_blurred_noisy, psf)


# plotting the results
plt.figure(figsize=[10,10])
plt.subplot(131), plt.imshow(camera), plt.axis('off'), plt.title('Original image')
plt.subplot(132), plt.imshow(camera_blurred_noisy), plt.axis('off'), plt.title('Noisy and Blurred Image')
plt.subplot(133), plt.imshow(camera_spectraldiv), plt.axis('off'), plt.title('Spectral division')
plt.tight_layout()

