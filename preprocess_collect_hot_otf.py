import sys
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import Imputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import FastICA
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import scipy as sp
from scipy.linalg import solve
from scipy import signal
import scipy.optimize as sop
from scipy.interpolate import interp1d
from scipy import arange, array, exp
from scipy.linalg import solveh_banded
from scipy.signal import find_peaks_cwt
import pywt
import peakutils
import glob
import os, errno
import shutil
import time
import matplotlib.pyplot as plt

dir = 'C:/Users/seo39/Drive/OBS/STO2/'
dir_base = dir + 'REF_ICA/'
dir_data_in = dir + 'DATA/LEVEL0.6/SCANS/'
dir_data_out = dir_base+'DATA/'
folder_name = os.listdir(dir_data_in)
scans = np.sort(np.array(folder_name,dtype=int))

#shutil.rmtree(dir_data_out, ignore_errors=True)
#os.mkdir(dir_data_out)

nchan = 1024
lin =2

hdu_hot_otf =[]
for i0 in scans:
	dir_data = dir_data_in+'{0:05d}'.format(i0)+'/'
	print(i0)
	# find HOT0 and REF scans
	file_REH = glob.glob(dir_data+'HOT*')
	file_REC = glob.glob(dir_data+'REF*')
	file_HOT = glob.glob(dir_data+'HOT*')
	file_OTF = glob.glob(dir_data+'OTF*')
	
	
	if (len(file_HOT) > 0) & (len(file_OTF) > 0):
		nfile = len(file_HOT)
		for j0 in range(0,nfile):
			hdu = fits.open(file_HOT[j0])
			hdu_hot_otf.append(hdu)
	
hot_otf_size = len(hdu_hot_otf)
time_hot_otf = np.zeros(hot_otf_size)
hot_otf_t = np.zeros([hot_otf_size,nchan])
print(hot_otf_size)
for i3 in range(0,hot_otf_size):
	print(i3)
	hdu_one = hdu_hot_otf[i3].copy()
	data = hdu_one[1].data
	int_time = np.float(hdu_one[1].header['OBSTIME'])
	obs_time = np.float(hdu_one[1].header['UNIXTIME'])
	intensity = data.field('DATA')/int_time
	time_hot_otf[i3] = obs_time
	hot_otf_t[i3,:] = intensity[lin,:]

hdu.close()
	
hdu = fits.PrimaryHDU(time_hot_otf)
hdu.writeto(dir_data_out+'time_hot_otf.fits')
hdu = fits.PrimaryHDU(hot_otf_t)
hdu.writeto(dir_data_out+'hot_otf_t.fits')
