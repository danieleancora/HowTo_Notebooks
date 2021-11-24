#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 19:09:33 2021

@author: danieleancora
"""

# %% IMPORT USEFUL LIBRARIES
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import filters

from scipy import signal


# %% DECONVOLUTION WITH RICHARDSON-LUCY
# we load a standard image
camera = data.camera()
# camera = data.shepp_logan_phantom()

# intensity normalization
camera = np.float32(camera[1:,1:])
camera /= camera.max()


# kernel size and properties
kernelsize = camera.shape[0]
center = np.uint32(np.floor(kernelsize/2.))

# ideal system response
delta = np.zeros([kernelsize,kernelsize])
delta[center,center] = 1

# Gaussian optical response
psf = filters.gaussian(delta, sigma=2)


# blurring the signal
camera_blurred = signal.convolve(camera, psf, 'same')


# function to compute the euclidean distance.
def euclideandistance(image1, image2):
    distance = ((image1-image2)**2).sum()
    
    return distance

# this computes one kind of signal to noise ratio
def snrIntensity_db(signal, noise):
    snr = 20*np.log10(np.mean(signal) / np.mean(noise))
    return snr

# function that implements the deconvolution
def deconvolutionRL(blurred, psf, iterations=10):
    # reconstruction initialization as the blurred item
    deconvolved = blurred.copy()
    # deconvolved = np.ones_like(blurred)/(blurred.shape[0]**2)
    distance = np.zeros([iterations,])
    
    for i in range(iterations): 
        # useful quantities
        forwardblur = signal.convolve(deconvolved, psf, 'same')
        ratio = blurred / forwardblur
        #deconvolved = np.maximum(ratio, 1e-6, dtype = 'float32')

        distance[i] = euclideandistance(blurred, forwardblur)
        
        # iterative update
        deconvolved *= signal.correlate(ratio, psf, 'same') 

    return deconvolved, distance


# running the deconvolution
camera_deconvolved, distance = deconvolutionRL(camera_blurred, psf, iterations=100)


# plotting the results
plt.figure(figsize=[10,10])
plt.subplot(131), 
plt.imshow(camera, vmin=0, vmax=1), plt.axis('off'), plt.title('Original image')
plt.subplot(132), 
plt.imshow(camera_blurred, vmin=0, vmax=1), plt.axis('off'), plt.title('Blurred image')
plt.subplot(133),
plt.imshow(camera_deconvolved, vmin=0, vmax=1), plt.axis('off'), plt.title('Deconvolved image')
plt.tight_layout()


# plotting the results
profile = 256
plt.figure(figsize=[10,10])

plt.subplot(211)
plt.plot(camera[profile,200:300])
plt.plot(camera_blurred[profile,200:300])
plt.plot(camera_deconvolved[profile,200:300])
plt.legend(['original', 'blurred', 'deconvolved'])
plt.title('Image profile')
plt.xlabel('Position')
plt.ylabel('Intensity value [a.u.]')

plt.subplot(212)
plt.plot(distance)
plt.yscale('log')
plt.title('Distance during reconstruction')
plt.xlabel('# Iterations')
plt.ylabel('Euclidean distance')


# plotting the results
plt.figure(figsize=[10,10])
plt.subplot(231), plt.imshow(camera, vmin=0, vmax=1), plt.axis('off'), plt.title('Original image')
plt.subplot(232), plt.imshow(camera_blurred), plt.axis('off'), plt.title('Blurred image')
plt.subplot(233), plt.imshow(camera_deconvolved, vmin=0, vmax=1), plt.axis('off'), plt.title('Deconvolved image')

camera_fft_abs = np.abs(np.fft.fftshift(np.fft.fft2(camera)))
camera_blurred_fft_abs = np.abs(np.fft.fftshift(np.fft.fft2(camera_blurred)))
camera_deconvolved_fft_abs = np.abs(np.fft.fftshift(np.fft.fft2(camera_deconvolved)))

plt.subplot(234), plt.imshow(np.log(camera_fft_abs)), plt.axis('off'), plt.title('FFT Original image')
plt.subplot(235), plt.imshow(np.log(camera_blurred_fft_abs)), plt.axis('off'), plt.title('FFT Blurred image')
plt.subplot(236), plt.imshow(np.log(camera_deconvolved_fft_abs)), plt.axis('off'), plt.title('FFT Deconvolved image')
plt.tight_layout()


# %% ADDING THE NOISE!!!
# blurring the signal
alpha = 0.01
noise = np.random.rand(camera.shape[0], camera.shape[1])
camera_blurred = signal.convolve(camera, psf, 'same') + alpha * noise


# running the deconvolution
camera_deconvolved, distance = deconvolutionRL(camera_blurred, psf, iterations=100)


# plotting the results
plt.figure(figsize=[10,10])
plt.subplot(221), 
plt.imshow(camera, vmin=0, vmax=1), plt.axis('off'), plt.title('Original image')
plt.subplot(222), 
plt.imshow(psf), plt.axis('off'), plt.title('Blurring PSF')
plt.subplot(223), 
plt.imshow(camera_blurred, vmin=0, vmax=1), plt.axis('off'), plt.title('Blurred image')
plt.subplot(224),
plt.imshow(camera_deconvolved, vmin=0, vmax=1), plt.axis('off'), plt.title('Deconvolved image')
plt.tight_layout()


# plotting the results
profile = 256
plt.figure(figsize=[10,10])

plt.subplot(211)
plt.plot(camera[profile,200:300])
plt.plot(camera_blurred[profile,200:300])
plt.plot(camera_deconvolved[profile,200:300])
plt.legend(['original', 'blurred', 'deconvolved'])
plt.title('Image profile')
plt.xlabel('Position')
plt.ylabel('Intensity value [a.u.]')

plt.subplot(212)
plt.plot(distance)
plt.yscale('log')
plt.title('Distance during reconstruction')
plt.xlabel('# Iterations')
plt.ylabel('Euclidean distance')


# %% ERROR STUDY
plt.figure(figsize=[10,5])
for i in range(10):
    noise = np.random.rand(camera.shape[0], camera.shape[1])
    alpha = i * 0.01
    camera_blurred = signal.convolve(camera, psf, 'same') + alpha * noise
    camera_deconvolved, distance = deconvolutionRL(camera_blurred, psf, iterations=100)
    #distance /= distance.max()
    plt.plot(distance)
    
plt.legend([ x*0.01 for x in range(10) ])
plt.yscale('log')

