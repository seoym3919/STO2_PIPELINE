import sys
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import Imputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import FastICA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import scipy as sp
from scipy.linalg import solve
from scipy import signal
import scipy.optimize as sop
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
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
import progressbar

"""

Collections of subroutines

"""

def silentremove(filename):
# remove files without raising error
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def validate_raw_spec(file_OTF,line,mask=None):
	hdu_OTF = fits.open(file_OTF[0])
	data_OTF = hdu_OTF[1].data
	int_OTF = data_OTF.field('DATA')
	nchan = len(int_OTF[line,:])
	if (mask is None):
		mask =np.ones(nchan, dtype = bool)
	good_raw_spec = np.zeros(len(file_OTF), dtype = bool)
	for i0 in range(0,len(file_OTF)):
		hdu_OTF = fits.open(file_OTF[i0])
		data_OTF = hdu_OTF[1].data
		int_OTF = data_OTF.field('DATA')
		good_raw_spec[i0] = (int_OTF[line,mask].sum() > 0.) & np.isfinite(int_OTF[line,mask].sum())
	return good_raw_spec
			
def wT_lowfreq(arr):
	"""
	# Find lowfrequency structure in spectra. Wipe out any structure less than 8 channel frequency.
	# Do not use this if channel is not evenly spaced. 
	"""
	coeff = pywt.wavedec(arr,'db5','zpd',5)
	coeff[1][:] = 0.
	coeff[2][:] = 0.
	coeff[3][:] = 0.
	coeff[4][:] = 0.
	coeff[5][:] = 0.
	arr_wT = pywt.waverec(coeff,'db5','zpd')
	return arr_wT

def fixdata_raw(arr, badpix, buffer=50, limit = 1.,deg = 3, verbose = False):
	"""
	This function fixes bad channels in data with the polynomial fitting
	input:
	    arr : the data arry
	    buffer : number of channel for buffer windows
	    limit : sigma level in finding bad channels in the sigma-clipping 
	output:
	    arr : array with bad channels fixed
	"""
	size = arr.shape
	mask_window = np.zeros(size[0],dtype = bool)
	mask_window[np.min(badpix)-buffer:np.max(badpix)+buffer] = True
	mask_bad = np.zeros(size[0],dtype = bool)
	mask_bad[badpix] = True 
	mask_buffer = mask_window & np.invert(mask_bad)
	std_buffer = np.std(arr[mask_buffer])
	mean_buffer = np.mean(arr[mask_buffer])
	#sigma clip
	badbad = (arr[mask_bad] < mean_buffer-limit*std_buffer) | (arr[mask_bad] > mean_buffer+limit*std_buffer)
	#update
	mask_bad[badpix] = badbad[:]
	mask_good = np.invert(mask_bad) & mask_window
	# set x-axis values
	x = np.arange(len(arr))
	x_in = x[mask_good]
	y_in = arr[mask_good]
	p0 = np.ones(deg+1)*0.1
	err_poly = lambda p0, x_in, y_in: y_in - free_poly(p0, x_in)
	p1, cov_x, infodict, mesg, ier = sop.leastsq(err_poly, p0[:], args=(x_in, y_in),full_output=1)
	y_polyfit = free_poly(p1,x[mask_bad])
	arr[mask_bad] = y_polyfit[:]
	if verbose:
		print(np.arange(size[0])[mask_bad],flush = True)
	return arr			


def fixdata(arr, badpix, buffer=25, limit = 1.):
	"""
	This function fixes bad channels in data with the univariate spline
	input:
	    arr : the data arry
	    buffer : buffer area to treat bad channels 
	    badpix : channels containing bad channels
	    limit : sigma level in finding bad channels in the sigma-clipping 
	output:
	    arr : array with bad channels fixed
	"""
	mask_bad = np.zeros(len(arr),dtype = bool)
	mask_bad[badpix] = True
	mask_treat = ExpandMask(mask_bad,buffer)
	mask_good = np.invert(mask_bad) & mask_treat
	
	# set x-axis values
	vv = np.arange(len(arr))

	xx = vv[mask_good]
	yy = arr[mask_good]
	param = np.polyfit(xx,yy,deg=3)
	func0 = lambda p, x: p[0]*x**3+p[1]*x**2+p[2]*x**1+p[3]
	arr_fit_good = func0(param,vv[mask_good]) 

	sig = np.std(arr[mask_good]-arr_fit_good)
	med = np.median(arr[mask_good]-arr_fit_good)

	arr_fit_bad = func0(param,vv[mask_bad]) 
	
	badchan = (arr[mask_bad]-arr_fit_bad >= med+limit*sig) | (arr[mask_bad]-arr_fit_bad <= med-limit*sig)
	
	arr_cut = arr[mask_bad].copy()
	arr_cut[badchan] = arr_fit_bad[badchan]
	arr[mask_bad] = arr_cut[:]
	
	return arr		

def ExpandMask(input, iters):
	"""
	Expands the True area in an array 'input'.
	Expansion occurs in the horizontal and vertical directions by one
	cell, and is repeated 'iters' times.
	"""
	Len = len(input)
	output = input.copy()
	for iter in range(0,iters):
		for y in range(1,Len-1):
			if (input[y]): 
				output[y-1] = True
				output[y]   = True
				output[y+1] = True
		input = output.copy()
	return output	
	
def free_poly(p, arr):
	y = np.zeros(len(arr))
	for i0 in range(0,len(arr)):
		y[i0] = (np.array(p[:])*arr[i0]**np.arange(len(p))).sum() 
	return y

def smoothing(x,window_len=11,window='hanning'):
	"""smooth the data using a window with requested size.
	This method is based on the convolution of a scaled window with the signal.
	The signal is prepared by introducing reflected copies of the signal 
	(with the window size) in both ends so that transient parts are minimized
	in the begining and end part of the output signal.
	input:
		x: the input signal 
		window_len: the dimension of the smoothing window; should be an odd integer
		window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.
	output:
		the smoothed signal
	example:
	t=linspace(-2,2,0.1)
	x=sin(t)+randn(len(t))*0.1
	y=smooth(x)
	see also: 
	numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
	scipy.signal.lfilter
	TODO: the window parameter could be the window itself if an array instead of a string
	NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
	"""
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len<3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y[np.int((window_len-1)/2):np.int(len(x)+(window_len-1)/2)]


def find_nearest(array,value):
	# find nearest index
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def find_SKY_start_end(file_OTF,time_tot,start_time,hh,y_factor_all,hot_all,Thot):
	# find the first and last OTF dump within a single raster OTF scan. 
	# Either the first and the last OTF dump will serve as a sky (OFF) spectrum if there is no Sky_REF available 
	for i1 in range(0,len(file_OTF)):
		hdu_OTF = fits.open(file_OTF[i1])
		data_OTF = hdu_OTF[1].data
		int_time_OTF = np.float(hdu_OTF[1].header['OBSTIME'])
		obs_time_OTF = np.float(hdu_OTF[1].header['UNIXTIME'])
		int_OTF = data_OTF.field('DATA')/int_time_OTF
		int_OTF[lin,:] = fixdata_raw(int_OTF[lin,:],badpix)
		int_OTF[lin,badpix] = wT_lowfreq(smoothing(int_OTF[lin,:]))[badpix]
		int_OTF[lin,1023] = int_OTF[lin,1022]
		y_fit = np.zeros(nchan)
		hot_fit = np.zeros(nchan)
		for j in range(0,nchan):
			f0 = interp1d((time_tot-start_time)/1.e3, y_factor_all[:,j], kind='linear')
			y_fit[j] = f0((obs_time_OTF-start_time)/1.e3)
			f1 = interp1d((time_tot-start_time)/1.e3, hot_all[:,j], kind='linear')
			hot_fit[j] = f1((obs_time_OTF-start_time)/1.e3)
		# Tsys
		Tsys = (Thot - y_fit * 45.) / (y_fit - 1.)
		TA_dev = np.zeros(size[0])
		TA_amp_max = np.zeros(size[0]) 
		for j2 in range (0,size[0]):
			REF = hot_fit[:]/hh[j2,:]/y_fit[:]
			TA_nocorrect = Tsys*(int_OTF[lin,:]-REF)/REF
			TA_nocorrect = TA_nocorrect-np.median(TA_nocorrect)
			TA_nocorrect = TA_nocorrect*2.
			TA_dev[j2] = (TA_nocorrect**2).sum()/np.float(nchan)
			TA_amp_max[j2] = np.max(np.abs(TA_nocorrect))
		distance = np.sqrt(np.log10(TA_amp_max)**2+np.log10(TA_dev)**2)
		dist_min = np.min(distance)
		dist_min_ind = np.argmin(distance)
		REF = hot_fit[:]/hh[dist_min_ind,:]/y_fit[:]
		TA_test = Tsys*(int_OTF[lin,:]-REF)/REF
		if np.max(TA_test) < 250.:
			break
		
		
	best_ind = np.argsort(distance)[0:num_base_sample]
	hh_match_group = hh[best_ind,:]
	SKY_start = hot_fit[:]/hh[best_ind[0],:]/y_fit[:]
	obs_time_OTF_start = obs_time_OTF
	#
	# Find the last good spectra within a scan and find the best hot vs hot ratio
	for i1 in range(len(file_OTF)-1,0,-1):
		hdu_OTF = fits.open(file_OTF[i1])
		data_OTF = hdu_OTF[1].data
		int_time_OTF = np.float(hdu_OTF[1].header['OBSTIME'])
		obs_time_OTF = np.float(hdu_OTF[1].header['UNIXTIME'])
		int_OTF = data_OTF.field('DATA')/int_time_OTF
		int_OTF[lin,:] = fixdata_raw(int_OTF[lin,:],badpix)
		int_OTF[lin,badpix] = wT_lowfreq(smoothing(int_OTF[lin,:]))[badpix]
		int_OTF[lin,1023] = int_OTF[lin,1022]
		y_fit = np.zeros(nchan)
		hot_fit = np.zeros(nchan)
		for j in range(0,nchan):
			f0 = interp1d((time_tot-start_time)/1.e3, y_factor_all[:,j], kind='linear')
			y_fit[j] = f0((obs_time_OTF-start_time)/1.e3)
			f1 = interp1d((time_tot-start_time)/1.e3, hot_all[:,j], kind='linear')
			hot_fit[j] = f1((obs_time_OTF-start_time)/1.e3)
		# Tsys
		Tsys = (Thot - y_fit * 45.) / (y_fit - 1.)
		TA_dev = np.zeros(size[0])
		TA_amp_max = np.zeros(size[0]) 
		for j2 in range (0,size[0]):
			REF = hot_fit[:]/hh[j2,:]/y_fit[:]
			TA_nocorrect = Tsys*(int_OTF[lin,:]-REF)/REF
			TA_nocorrect = TA_nocorrect-np.median(TA_nocorrect)
			TA_nocorrect = TA_nocorrect*2.
			TA_dev[j2] = (TA_nocorrect**2).sum()/np.float(nchan)
			TA_amp_max[j2] = np.max(np.abs(TA_nocorrect))
		distance = np.sqrt(np.log10(TA_amp_max)**2+np.log10(TA_dev)**2)
		dist_min = np.min(distance)
		dist_min_ind = np.argmin(distance)
		best_ind = np.argsort(distance)[0:num_base_sample]
		REF = hot_fit[:]/hh[dist_min_ind,:]/y_fit[:]
		TA_test = Tsys*(int_OTF[lin,:]-REF)/REF
		if np.max(TA_test) < 250.:
			break
	
	SKY_end = hot_fit[:]/hh[best_ind[0],:]/y_fit[:]
	obs_time_OTF_end = obs_time_OTF
	OTF_time_start_end= (np.array([obs_time_OTF_start,obs_time_OTF_end])-obs_time_OTF_start)/1.e3
	SKY_start_end = np.array([SKY_start,SKY_end])

	return obs_time_OTF_start,OTF_time_start_end,SKY_start_end 


def read_preprocess_data(dir_data_in):
	hdu_time_hot_otf = fits.open(dir_data_in+'time_hot_otf.fits')
	time_hot_otf = hdu_time_hot_otf[0].data

	hdu_hh = fits.open(dir_data_in+'hot_vs_hot.fits')
	hothot = hdu_hh[0].data
	
	hdu_time_tot = fits.open(dir_data_in+'time_tot.fits')
	time_tot = hdu_time_tot[0].data

	hdu_y_factor = fits.open(dir_data_in+'y_interp.fits')
	y_factor_all = hdu_y_factor[0].data

	hdu_hot = fits.open(dir_data_in+'hot_interp.fits')
	hot_all = hdu_hot[0].data

	hdu_hot_otf = fits.open(dir_data_in+'hot_otf_interp.fits')
	hot_otf_all = hdu_hot_otf[0].data

	hdu_sky = fits.open(dir_data_in+'sky_interp.fits')
	sky_all = hdu_sky[0].data

	return time_hot_otf, hothot, time_tot, y_factor_all, hot_all, hot_otf_all, sky_all



def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=1e6,max_iters=50, conv_thresh=1e-5, verbose=False):
	'''Computes the asymmetric least squares baseline.
	* http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
	smoothness_param: Relative importance of smoothness of the predicted response.
	asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
	Setting p=1 is effectively a hinge loss.
	'''
	smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
	# Rename p for concision.
	p = asymmetry_param
	# Initialize weights.
	w = np.ones(intensities.shape[0])
	for i in range(max_iters):
		z = smoother.smooth(w)
		mask = intensities > z
		new_w = p*mask + (1-p)*(~mask)
		conv = np.linalg.norm(new_w - w)
		if verbose:
			print (i+1, conv)
		if conv < conv_thresh:
			break
		w = new_w
	else:
		print('ALS did not converge in %d iterations' % max_iters)
	return z


class WhittakerSmoother(object):
	def __init__(self, signal, smoothness_param, deriv_order=1):
		self.y = signal
		assert deriv_order > 0, 'deriv_order must be an int > 0'
		# Compute the fixed derivative of identity (D).
		d = np.zeros(deriv_order*2 + 1, dtype=int)
		d[deriv_order] = 1
		d = np.diff(d, n=deriv_order)
		n = self.y.shape[0]
		k = len(d)
		s = float(smoothness_param)
		
		# Here be dragons: essentially we're faking a big banded matrix D,
		# doing s * D.T.dot(D) with it, then taking the upper triangular bands.
		diag_sums = np.vstack([np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant') for i in range(1, k+1)])
		upper_bands = np.tile(diag_sums[:,-1:], n)
		upper_bands[:,:k] = diag_sums
		for i,ds in enumerate(diag_sums):
			upper_bands[i,-i-1:] = ds[::-1][:i+1]
		self.upper_bands = upper_bands
		
	def smooth(self, w):
		foo = self.upper_bands.copy()
		foo[-1] += w  # last row is the diagonal
		return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)

	
	
def gauss_func(x,amp,center,sigma):
	# Gaussian function
	y = amp*np.exp(-0.5*(x-center)**2/sigma**2)
	return y


def multi_gauss_func(x,par,ngauss):
	"""
	# returns multiple Gaussian curves, requires gauss_func routine
	"""
	if ngauss == 0:
		print('Number of Gaussian curves is 0',flush=True)
	if len(par) < 3 * ngauss:
		print('Number of elements are not sufficient for  '+'{0:02d}'.format(ngauss)+' Gaussian curves.',flush=True)
		print('Number of parameter is '+'{0:02d}'.format(len(par)),flush=True)

	xsize = len(x)
	gauss_array = np.zeros([ngauss,xsize])
	for i0 in range(0,ngauss):
		gauss_array[i0,:] = gauss_func(x, par[i0*3],par[i0*3+1],par[i0*3+2])

	multi_gauss = gauss_array.sum(axis =0) 
	return multi_gauss

	
def multi_gauss_func_fixed_center(x,par,center,ngauss):
	"""
	# returns multiple gaussian curves when center positions of Gaussian curves are fixed
	"""
	if ngauss == 0:
		print('Number of Gaussian curves is 0',flush=True)

	if (len(par) < 2 * ngauss):
		print('Number of elements are not sufficient for  '+'{0:02d}'.format(ngauss)+' Gaussian curves.',flush=True)
		print('Number of parameter is '+'{0:02d}'.format(len(par)),flush=True)

	xsize = len(x)
	gauss_array = np.zeros([ngauss,xsize])
	for i0 in range(0,ngauss):
		gauss_array[i0,:] = gauss_func(x, par[i0*2],center[i0],par[i0*2+1])

	multi_gauss = gauss_array.sum(axis =0) 
	return multi_gauss

	
def peak_fit(x,y,peak_pos,npeaks, init_amp = 0.1, init_width = 1.):
	"""
	# fits gaussian to peaks found by peakutils. Peak positions should be inputed and are fixed in the fitting.
	# input:
	#	x: 1D x axis array (velocity or frequency array for spectra)
	#	y: 1D y axis array (intensity array for spectra)
	#	peak_pos: position of peaks in velocity. Should be obtained from peak-detecting algorithm. Peak position will be fixed for the fitting. 
	#	npeaks: number of peaks 
	# output:
	#	gauss_peaks: summation of fitted gaussian curves
	#	p1: paramters for Gaussian curves. Only has amplitudes and widths of the curves.   
	"""
	xsize = len(x)
	ysize = len(x)
	peak_pos_size = len(peak_pos)
	if xsize != ysize:
		print('x and y arryas should have same dimensiona and size')

	if peak_pos_size == 0:
		print('peak position is not provided. Fitting cannot continue')


		
	# create input parameter array for fitting
	par = np.zeros(npeaks*2)
	for i0 in range(0,npeaks):
		par[i0*2] = init_amp
		par[i0*2+1] = init_width
	
	# make error function and fit using scipy least square fitting	
	errfunc = lambda par, x, y: y - multi_gauss_func_fixed_center(x, par[0:npeaks*2], peak_pos, npeaks)
	p1, cov_x, infodict, mesg, ier = sop.leastsq(errfunc, par, args=(x, y),full_output=1,ftol=1.5e-10,xtol=1.5e-10)
	gauss_peaks = multi_gauss_func_fixed_center(x, p1, peak_pos, npeaks)
	return gauss_peaks, p1


def masking_peaks(x, center, sigma, factor = 3., buffer = 0.):
	"""
	# Create mask for the Gaussian peaks the and baseline
	# input:
	#	x: input x-axis array
	#	center: mu values of the Gaussian curves
	#	sigma: sigma values of the Gaussian curves
	#	factor: controls mask width,  factor * sigma
	#	buffer: additional width to peak mask
	# output
	#	mask_peaks: boolean array for Gaussian peaks
	#	mask_base:boolean array for baseline
	"""
	if len(center) != len(sigma):
		print('Error: the dimension of center and sigma is not same',flush=True)
	if len(center) == 0:
		print('Error: empty data',flush=True)
	mask_temp = np.zeros([len(center),len(x)],dtype = bool)
	mask_peaks = np.zeros(len(x),dtype = bool)
	for i0 in range(0,len(center)):
		mask_temp[i0,:] = (x <= center[i0] + factor * sigma[i0] + buffer) & (x >= center[i0] - factor * sigma[i0] - buffer)
	
	for j0 in range(0,len(x)):
		mask_peaks[j0] = mask_temp[:,j0].any()
	mask_base = np.invert(mask_peaks)
	return mask_peaks, mask_base


def find_peaks(y,npeaks,steps = 0.01):
	"""
	# Find the strongest peaks upto npeaks 
	# input:
	#	y: input array to find peaks
	#	npeaks: number of peaks
	#	steps: interval of limit to find peaks in normalized spectra (highest peak value is 1)
	# output:
	#	peak_indexes: array containing peak index 
	"""
	limit = 1.
	peak_indexes = []
	while (len(peak_indexes) < npeaks) & (limit > 0.):		
		limit = limit - steps
		peak_indexes = peakutils.peak.indexes(y,limit,20)
	# limit number of peaks to npeaks if it happen to find more than npeaks.  
	if len(peak_indexes) >= npeaks+1:
		temp = y[peak_indexes]
		temp_sort_index = np.argsort(temp)
		peak_indexes = peak_indexes[temp_sort_index[::-1][0:npeaks]]	
	return peak_indexes


def find_window(x, arr, npeaks, win_peak, win_limit, factor, buffer, plot = False, scan= 0, group=0, dir = None, verbose = False):
	"""
	# Finds windows using spectra in array
	# input:
	#	x: velocity array in 1D
	#	arr: spectra intensity array in 2D. Dimension is [# of spectra, # of channels]
	#	npeaks: number of peaks to be found
	#	win_peak: window to find peaks, boolean array
	#	win_limit: force additional limit to window. Sets region of interest. boolean array
	#	factor: controls width of peaks; half width of window is factor * sigma
	#	buffer: additional width to the windows. width of window + 2* buffer
	#	plot: if true, it will generate plot for standard deviation and Gaussian fitting
	#	scan: scan number to be written in file name and title of plot
	#	dir: output directory. If it is none, it will save plot in the local folder
	# output:
	#	win_gauss_fit:
	#	win_baseline_fit:
	#	npeaks_roi:
	"""
	if dir == None:
		dir = './'
	size = arr.shape
	nspec = size[0]
	nchan = size[1]
	# estimate standard deviation for each channel
	stdev = np.zeros(nchan)
	for j0 in range (0,nchan):
		stdev[j0] = np.std(arr[:,j0])
	# get lowfrequency structure using the wavelet transform
	stdev_wT = wT_lowfreq(stdev)
	# estimate baseline using the asymmetric least square smoothing. This will eliminate broad features due to large fringes
	als_stdev = als_baseline(stdev_wT,asymmetry_param=0.03)
	# subtract baseline
	stdev_smooth = stdev_wT - als_stdev
	# setting  region for peak-finding
	x_peakfind = x[win_peak]
	stdev_smooth_peakfind = stdev_smooth[win_peak]
	# find peaks
	peak_indexes = find_peaks(stdev_smooth_peakfind, npeaks, steps = 0.01)
	# number of peaks within region of interest
	peaks_in_vel = x_peakfind[peak_indexes]
	npeaks_roi = len(peaks_in_vel[(peaks_in_vel >= np.min(x[win_limit])) & (peaks_in_vel <= np.max(x[win_limit]))])
	gauss_fit, p1 = peak_fit(x,stdev_smooth, peaks_in_vel, npeaks, init_amp = 0.1, init_width = 1.)
	sigma = np.zeros(npeaks)
	for j0 in range(0,npeaks):
		sigma[j0] = np.abs(p1[j0*2+1])
		if (sigma[j0] > np.abs(p1[j0*2])* 5.):
			sigma[j0] = 0.
	mask_peaks, maks_base = masking_peaks(x, peaks_in_vel, sigma, factor = factor, buffer = buffer)
	win_gauss_fit = mask_peaks & win_limit
	# Get channels for baseline fitting by inverting windows for the Gaussian fit
	win_baseline_fit = np.invert(win_gauss_fit)
	if plot:
		plt.axis([np.min(x),np.max(x),-5, 2.*np.max(stdev)])
		plt.xlabel('Velocity [km/s]')
		plt.ylabel('Standard Deviation [K]')
		plt.title('Scan = '+'{0:05d}'.format(scan)+' Group = '+'{0:02d}'.format(group))
		plt.plot(x,stdev, label = 'Standard Deviation')
		plt.plot(x,gauss_fit, label = 'Gaussian')
		#plt.plot(x,als_stdev, label = 'ALS')
		plt.plot(x_peakfind,stdev_smooth_peakfind, label = 'wT')
		plt.plot(x_peakfind[peak_indexes], stdev_smooth_peakfind[peak_indexes], 'ro',label = 'Peaks')
		if len(x[win_gauss_fit]) > 0:
			y = x[win_gauss_fit].copy()
			y[:] = 0.
			plt.plot(x[win_gauss_fit],y,'y.',markersize=1)
		plt.legend(frameon=False)
		plt.savefig(dir+'stdev_'+'{0:05d}'.format(scan)+'_group_'+'{0:02d}'.format(group)+'.png',dpi=250)
		plt.close("all")	
	return win_gauss_fit, win_baseline_fit, npeaks_roi


def find_single_window(x, arr, badpix, win_peak, win_limit, factor, buffer, plot = False, scan= 0, dir = None, verbose = True):
	"""
	# Finds windows using spectra in array
	# input:
	#	x: velocity array in 1D
	#	arr: spectra intensity array in 2D. Dimension is [# of spectra, # of channels]
	#	win_peak: window to find peaks, boolean array
	#	win_limit: force additional limit to window. Sets region of interest. boolean array
	#	factor: controls width of peaks; half width of window is factor * sigma
	#	buffer: additional width to the windows. width of window + 2* buffer
	#	plot: if true, it will generate plot for standard deviation and Gaussian fitting
	#	scan: scan number to be written in file name and title of plot
	#	dir: output directory. If it is none, it will save plot in the local folder
	# output:
	#	win_gauss_fit:
	#	win_baseline_fit:
	#	npeaks_roi:
	"""
	if dir == None:
		dir = './'
	size = arr.shape
	nspec = size[0]
	nchan = size[1]
	# estimate standard deviation for each channel
	stdev = np.zeros(nchan)
	for j0 in range (0,nchan):
		stdev[j0] = np.std(arr[:,j0])
	stdev = fixdata(stdev, badpix, buffer=25, limit = 1.)
	# get lowfrequency structure using the wavelet transform
	stdev_wT = wT_lowfreq(stdev)
	# estimate baseline using the asymmetric least square smoothing. This will eliminate broad features due to large fringes
	als_stdev = als_baseline(stdev,asymmetry_param=0.03)
	# subtract baseline
	stdev_smooth = stdev_wT - als_stdev
	x_peakfind = x[win_peak]
	stdev_smooth_peakfind = stdev_smooth[win_peak]
	# find the largest peak within the region of interest
	peak_index = find_peaks(stdev_smooth_peakfind, 1, steps = 0.01)
	# number of peaks within region of interest
	peaks_in_vel = x_peakfind[peak_index]
	mask_neighbor = (x <= peaks_in_vel+10.) & (x >= peaks_in_vel-10.)
	gauss_fit, p1 = peak_fit(x[mask_neighbor],stdev_smooth[mask_neighbor], peaks_in_vel, 1, init_amp = 0.1, init_width = 1.)
	sigma = [np.abs(p1[1])]
	if sigma > np.abs(p1[0])* 3.:
		sigma = [0.1]
	mask_peaks, maks_base = masking_peaks(x, peaks_in_vel, sigma, factor = factor, buffer = buffer)
	win_gauss_fit = mask_peaks & win_limit
	# Get channels for baseline fitting by inverting windows for the Gaussian fit
	win_baseline_fit = np.invert(win_gauss_fit)
	if plot:
		plt.axis([np.min(x),np.max(x),-2, 2.*np.max(stdev)])
		plt.xlabel('Velocity [km/s]')
		plt.ylabel('Standard Deviation [K]')
		plt.title('Scan = '+'{0:05d}'.format(scan))
		plt.plot(x,stdev, label = 'Standard Deviation',lw =0.25, drawstyle = 'steps',color='k')
		plt.plot(x[mask_neighbor],gauss_fit+als_stdev[mask_neighbor], label = 'Gaussian',lw =1, color = '#ff4400')
		plt.plot(x,stdev_wT, label = 'wT',lw =0.85, color = '#f400f4')
		plt.plot(x_peakfind[peak_index], (stdev_smooth_peakfind+als_stdev[win_peak])[peak_index], 'ro',label = 'Peaks',markersize =2.5)
		plt.legend(frameon=False)
		plt.savefig(dir+'stdev_'+'{0:05d}'.format(scan)+'.png',dpi=250)
		plt.close("all")	
	return win_gauss_fit, win_baseline_fit

	
def DeFringe_ICA(spectra, n_comps= 4, plot=False, scan = 0, group = 0, dir= None):
	# Defringing algorithm using a fast deflational independant component analysis (ICA)
	# The deflational ICA analysis hidden components/features within a given spectral group and orders them based on the amplitude of the features.
	# The strongest feature is the first component. 
	# For a massive data reduction, the asymmetric least square is not very effective since the parameter should be modified frequently 
	# and it is hard to find adequate value autopmatically.  
	if dir is None:
		dir = './'
	size = spectra.shape
	cube_ica_ready = spectra.transpose()
	ica = FastICA(n_components=n_comps,max_iter=1000, algorithm='deflation',random_state=0)
	S_ = ica.fit_transform(cube_ica_ready) 
	A_ = ica.mixing_
	TA_defringed= np.zeros([size[0],nchan]) 
	for j0 in range(0,size[0]):
		main_feature = np.argmax(np.abs(A_[j0,:]))
		ica_comp = np.vstack([S_.transpose()])
		TA_defringed[j0,:] = spectra[j0]-smoothing(smoothing(S_[:,main_feature]*A_[j0,main_feature]))	
	if plot:
		for j0 in range(0,size[0]):
			main_feature = np.argmax(np.abs(A_[j0,:]))
			plt.plot(spectra[j0],color='k',lw=0.7, drawstyle='steps')
			plt.plot(smoothing(smoothing(S_[:,main_feature]*A_[j0,main_feature])),'b',lw=0.5, drawstyle='steps')
			plt.axis([0,1024,-100,100])
			plt.xlabel('Channel')
			plt.ylabel('Ta')
			plt.savefig(dir+'ICA_DeFringe'+'{0:05d}'.format(scan)+'_'+'{0:05d}'.format(group)+'_'+'{0:05d}'.format(j0)+'.png',dpi=250)
			plt.close('all')	
	return TA_defringed, S_, A_


	
def DeBase_ICA(spectra, win_line, sigma_limit = 2.5, buffer = 50, plot_comp = False, plot_line = False, scan = 0, group = 0, dir= None):
	# Baseline correction using a fast parallel independant component analysis (ICA).
	# The ICA separate a group of spectra based on spectral features, which isolates noises and signals.
	# While it does not isolates the signals completely from noises and sometimes breaks down signals into multiple components,
	# the ICA is quite good to take out white noises and standing waves from gaussian signals, whcih enhances SNR in ICA components.
	# We do the sigma clipping to the ICA components, which are smoothed for further increase of SNR, 
	# to generate low-frequency baseline of each ICA components and recombine the baselines of the ICA components to generate spectrum baseline.
	# This algorithm may be replaced with the asymmetric least square if the standing wave has 
	# a frequency either > a few channel width or < 1/6 of total bandwidth.
	if dir is None:
		dir = './'
	size = spectra.shape
	cube_ica_ready = spectra.transpose()
	ica = FastICA(n_components=size[0],max_iter=1000, algorithm='deflation',random_state=0)
	S_ = ica.fit_transform(cube_ica_ready)
	A_ = ica.mixing_
	# Find low frequency structure for each component
	S_lowfreq = np.zeros([1024,size[0]])
	ica_comp = np.vstack([S_.transpose()])
	for k in range(0,size[0]):
		S_lowfreq[:,k] = wT_lowfreq(S_[:,k])
	# Find outliers
	xx = np.arange(size[1])
	center_smoothing = np.zeros([size[0],size[1],3])
	outliers_mask = np.zeros([size[0],size[1]], dtype = bool)
	# subplots sharing both x/y axes
	f, ax = plt.subplots(size[0], sharex=True, sharey=False,figsize=(15,size[0]*1.5))
	f.tight_layout()
	ax[0].plot(ica_comp[0,:])
	ax[0].set_title('ICA components')
	for k in range(1,size[0]):
		yy = S_[:,k].copy()
		yy_lowfreq = wT_lowfreq(yy)
		xy = np.vstack([xx,yy_lowfreq]).transpose()
		# Isolation forest to find spiky features
		clf = IsolationForest(max_samples=20)
		clf.fit(xy)
		y_pred_train = clf.predict(xy)
		outlier_mask1 = y_pred_train == -1
		# smoothing to find line feature
		center_smoothing[k,:,0] = smoothing(yy,window_len=101)
		center_smoothing[k,:,1] = smoothing(yy,window_len=201)
		center_smoothing[k,:,2] = smoothing(yy,window_len=301)
		# estimate deviation
		devi1 = (yy_lowfreq - center_smoothing[k,:,0])**2
		devi2 = (yy_lowfreq - center_smoothing[k,:,1])**2 
		devi3 = (yy_lowfreq - center_smoothing[k,:,2])**2
		# sum of deviation at each channel
		devi_tot = devi1+devi2+devi3
		# deviation mean and standard deviation
		devi_std = np.std(devi_tot)
		devi_mean = np.median(devi_tot)
		# find outliers and create outlier mask
		outlier_mask2=(devi_tot >= devi_mean+2.5*devi_std)
		outlier_mask2 = ExpandMask(outlier_mask2, buffer)
		outliers_mask[k,:] = (outlier_mask1 | outlier_mask2) & win_line
		# find outliers and create outlier mask
		line_mask_new = np.zeros(size[1],dtype=bool)
		line_mask_new[outliers_mask[k,:]] = True
		base_mask = np.invert(line_mask_new)
		# linear interpolation in the ourliers
		func0 = interp1d(xx[base_mask], S_lowfreq[base_mask,k], kind = 'slinear')
		S_lowfreq[line_mask_new,k] = func0(xx[line_mask_new])
		#
		ax[k].plot(yy, drawstyle='steps')
		ax[k].plot(yy_lowfreq,color='r',lw=0.5)
		ax[k].plot(center_smoothing[k,:,0],color='#68efa9',lw=0.75)
		ax[k].plot(center_smoothing[k,:,1],color='y',lw=0.75)
		ax[k].plot(center_smoothing[k,:,2],color='#ddef04',lw=0.75)
		ax[k].scatter(xx[outliers_mask[k,:]],wT_lowfreq(yy)[outliers_mask[k,:]],color='r')
		ax[k].plot(S_lowfreq[:,k],color='g',lw=0.5)
	# Fine-tune figure; make subplots xclose to each other
	f.subplots_adjust(hspace=0)
	plt.xlabel('Channel')
	if plot_comp:
		plt.savefig(dir+'ICA_comps'+'{0:05d}'.format(scan)+'_'+'{0:05d}'.format(group)+'.png',dpi=250)
	plt.close('all')
	#
	# plot each line and reconstructed baseline
	if plot_line:
		for k in range(0,size[0]):
			plt.plot(spectra[k],color='k',lw=0.7, drawstyle='steps')
			#sort_index = np.argsort(A_[k,:])
			plt.plot(np.dot(S_lowfreq,A_[k,:]),'b',lw=0.5, drawstyle='steps')
			plt.axis([0,1024,-20,50])
			plt.xlabel('Channel')
			plt.ylabel('Ta')
			plt.savefig(dir+'ICA_recon'+'{0:05d}'.format(scan)+'_'+'{0:05d}'.format(group)+'_'+'{0:05d}'.format(k)+'.png',dpi=250)
			plt.close('all')
	return S_lowfreq, A_


def CalPower(spectra, mask = None):
	# estimates total power. Beware results is not in physical dimension. 
	size = spectra.shape
	if mask is None:
		mask = np.ones(size[1], dtype = bool)
	power = np.zeros(size[0]) 
	for i in range(0,size[0]):
		power[i] = (spectra[i,mask]**2).sum()*np.float(size[1])/(np.ones(size[1])[mask]).sum() 
	return power

	
def phase_portrait(x,array, plot = False, scan=0, group =0, dir = None):
	# produce the phase portrait, I vs dI/dv, of a spectrum
	#
	if dir is None:
		dir = './'
	#
	nelements,nchan  = array.shape
	dTdv = np.zeros([nelements,nchan])
	for i1 in range(0,nelements):
		dTdv[i1,:] = FiniteDiff(x,wT_lowfreq(array[i1,:]))
	TA_1d = array.reshape(nelements*nchan)
	dTdv_1d = dTdv.reshape(nelements*nchan)
	h2d,hx,hy = np.histogram2d(TA_1d,dTdv_1d,bins=[100,100])
	#
	center_idx = np.unravel_index(h2d.argmax(), h2d.shape)
	center = np.array([hx[center_idx[0]],hy[center_idx[1]]])
	#
	if plot:
		for i1 in range(0,nelements):
			plt.plot(array[i1,:],dTdv[i1,:],'r.',markersize =1, alpha=0.2)
			plt.xlabel('TA [K]')
		plt.ylabel('dTA/dV')
		plt.title('Phaseall Scan = '+'{0:05d}'.format(scan)+' Group ='+'{0:05d}'.format(group))
		plt.savefig(dir+'Phase_'+'{0:05d}'.format(scan)+'_group_'+'{0:05d}'.format(group)+'.png',dpi=250)
		plt.close("all")
	#
	return center, h2d, hx, hy


def FiniteDiff(x,y):
	# Returns the finite differential. This is forward FD. 
	delx= x[2]-x[1]
	size = len(y)
	y_rshift = np.zeros(size)
	y_lshift = np.zeros(size)
	y_lshift[0:size-1] = y[1:size]
	y_lshift[size-1] = y[size-1]
	y_rshift[1:size] = y[0:size-1]
	y_rshift[0] = y[0]
	dydx = 0.5*(y_lshift-y_rshift)/delx
	return dydx	


"""	
=======================================================================================================================
=======================================================================================================================

Main Program Start

=======================================================================================================================
=======================================================================================================================
"""
"""
=======================================================================================================================
 Preprocess 1: setting up directories and parameters for reduction
=======================================================================================================================
"""
# Basic directory setup	
dir = '/home/seoyoung/WORK/OBS/STO2/'
dir_base = dir + 'LM2/'
dir_data_in = dir_base+'DATA/'
dir_data_out = dir_base+'LINES/'
dir_image_out = dir_base+'LINES/'
dir_scans = dir+'DATA/LEVEL0.6/SCANS/'
#
# Number of channels
nchan =1024
#
# channel index
xx = np.arange(1024)
#
# Bad channels
badpix = [480,481,482,483,484,485,486,487,488,489,490]
#
# Channel numbers where signals are expected
line_chan_low = 500
line_chan_high = 555 
#
# Line selection 0: NII, 1: NII, 2: CII
lin = 2
#
# Number of spectra for baseline analysis in phase portrait
num_base_sample = 50
#
# calibration switches
expand_hot_ratio= False
expand_factor = 40
win_peak = (xx > 150) & (xx < 900)
win_limit = (xx > 150) & (xx < 900)	
#
# data output switch
zero_cal_data_out = True
first_cal_data_out = True
#
# plot switch
zero_cal_image_out = True
first_cal_image_out = True
#
# criteria for decision tree
# TP_xxx related to total power
# PP_xxx related to phase portrait
TP_min_crit = 1.e5
TP_std_crit_bad = 5.e4
TP_std_crit_good = 3.e3
TP_std_crit_new = 1.e4
PP_ta = 4. 
PP_dtadv = 0.5
#
#
"""
=======================================================================================================================
 Preprocess 2: sorting scan types (OTF or reference), read time for HOT in OTF, and read HOT_ref/HOT_OTF library
=======================================================================================================================
"""
# Setting up masks for sorting OTF and reference scan numbers
folder_name = os.listdir(dir_scans)
scans = np.sort(np.array(folder_name,dtype=int))
OTF_mask = np.zeros(len(scans),dtype = bool)
REF_mask = np.zeros(len(scans),dtype = bool)
#
# Sort which scan number is OTF or reference scan. Spiral maps are not included. 
count = 0 
for i0 in scans:
	dir_data = dir_scans+'{0:05d}'.format(i0)+'/'
	# find HOT0 and REF scans
	file_REH = glob.glob(dir_data+'HOT*')
	file_REC = glob.glob(dir_data+'REF*')
	file_OTF = glob.glob(dir_data+'OTF*')
	if len(file_OTF) > 1:
		OTF_mask[count] = True
	if (len(file_REC) > 0) & (len(file_REH) > 0):
		REF_mask[count] = True
	count = count + 1
OTF_scans = scans[OTF_mask]     # OTF_scans contain scan numbers of OTF scans
REF_scans = scans[REF_mask]     # REF_scans contain scan numbers of reference scans
#
#
# read preprocess data including 
time_hot_otf, hothot, time_tot, y_factor_all, hot_all, hot_otf_all, sky_all = read_preprocess_data(dir_data_in)
start_time = np.min(time_tot)
#
#hdu_time_hot_otf = fits.open(dir_data_in+'time_hot_otf.fits')
#time_hot_otf = hdu_time_hot_otf[0].data
#
# read HOT_ref/HOT_OTF
if expand_hot_ratio:            # If True, interpolate hot_otf/hot_ref library for better 
	size0 = hothot.shape
	hh = np.zeros([size0[0]*expand_factor,size0[1]]) 
	for j0 in range(0,size0[1]):
		func = interp1d(np.arange(size0[0]),hothot[:,j0])
		hh[:,j0] = func(np.arange(size0[0]*expand_factor)/np.float(size0[0]*expand_factor-1)*np.float(size0[0]-1))
	size = hh.shape
else:                           # If false, use the original hot_otf/hot_ref library
	hh = hothot
	size = hothot.shape
#
"""
=======================================================================================================================
 Reduction starts from here
=======================================================================================================================
"""
#
#single_scan = 3843
#for i0 in range(single_scan,single_scan+1):
mask_rerun = (OTF_scans <= 4000) & (OTF_scans >= 1)
for i0 in OTF_scans[mask_rerun]:
	print('Processing scan '+'{0:05d}'.format(i0)+'...',flush = True)
	#read Thot from HOT spectrum header
	dir_REF = dir_scans+'{0:05d}'.format(find_nearest(REF_scans,i0))+'/'     # Directory to spectra at reference positions 
	file_REH = glob.glob(dir_REF+'HOT*')                                     # Get HOT ref file name
	hdu_REH = fits.open(file_REH[0])                                         # Read ambient spectra at reference position 
	Thot=np.float(hdu_REH[1].header['CONELOAD'])                             # Read ambient temperature 
	# find OTF and HOT files
	dir_data = dir_scans+'{0:05d}'.format(i0)+'/'                            # Data directory for a given scan number
	file_OTF = np.sort(glob.glob(dir_data+'OTF*'))                           # Get OTF dump file names within a given scan number
	file_hot = glob.glob(dir_data+'HOT*')                                    # Get\ HOT dump file names within a given scan number 
	# determine whether there is hot spectrum in OTF scan
	if len(file_hot) > 0:   
		hot_otf_exist = True                                                 # If hot_otf exists, use hot_otf for Ta conversion
	else:
		hot_otf_exist = False                                                # If hot_otf does not exist, find sky spectra from OTF spectra

	# Create directory for saving spectra plots
	dir_line_image = dir_image_out+'{0:05d}'.format(i0)+'/'
	shutil.rmtree(dir_line_image, ignore_errors=True)              # clean out the directory
	os.mkdir(dir_line_image)                                       # create the plot directory 

	# Set up mask for line, base, and bad channels	
	mask_line = np.zeros(nchan,dtype = bool)                       # Line mask (boolean array)
	mask_line[line_chan_low:line_chan_high] = True
	mask_bad = np.zeros(nchan,dtype = bool)                        # Bad channel mask (boolean array)
	mask_bad[badpix] = True
	mask_base = np.invert(mask_line | mask_bad)                    # Good channel mask (boolean array)

	#OTF data index
	OTF_ind = np.arange(len(file_OTF)) 

	# find good raw data
	good_raw_spec = OTF_ind[validate_raw_spec(file_OTF,2,mask=mask_base)]
	#
	# If there is no hot in OTF, find Sky spectra for the first and the last OTF dumps that make the flatest baseline.
	if hot_otf_exist == False:
		obs_time_OTF_start, OTF_time_start_end, SKY_start_end= find_SKY_start_end(file_OTF,time_tot,start_time,hh,y_factor_all,hot_all,Thot)
	#
	# Set up array for saving position, spectra, total power, and observation time
	pos_all = np.zeros([len(file_OTF),2])         # position information, size: (number of spectra, (l,b))
	TA_original = np.zeros([len(file_OTF),nchan]) # spectra in Ta
	TPower = np.zeros([len(file_OTF),2])          # total power defined as Sum(Ta^2), total power is used for determining group of spectra
	obs_time = np.zeros(len(file_OTF))            # observation time for each spectra
	bar = progressbar.ProgressBar(max_value=np.max(good_raw_spec)) # progress bar for direct run. If a job is submitted in que, this should be commented out
	#
	'''
	#===
	# Start the zeroth calibration:  Convert spectra in counts to Ta
	#===
	'''
	print('Processing the zeroth level reduction', flush = True)
	for i1 in good_raw_spec:                                       # iterating only good spectra
		hdu_OTF = fits.open(file_OTF[i1])                          # read SDFITS file
		data_OTF = hdu_OTF[1].data                                 # save data to data_OFT, level 0 data is in count unit
		int_time_OTF = np.float(hdu_OTF[1].header['OBSTIME'])      # read integration time
		int_OTF = data_OTF.field('DATA')/int_time_OTF              # estimate couns per second
		int_OTF[lin,:] = fixdata_raw(int_OTF[lin,:],badpix)        # fix bad pixels with 2nd order polynomial
		int_OTF[lin,badpix] = wT_lowfreq(int_OTF[lin,:])[badpix]   # wavelet low frequency filter for discontinuity reduction at the boundary of bad pixel regions 
		int_OTF[lin,1023] = int_OTF[lin,1022]                      # bad channel fix at channel 1023
		#
		obs_time_OTF = np.float(hdu_OTF[1].header['UNIXTIME'])     # OTF observation time
		obs_time[i1] = (obs_time_OTF -start_time)/3600.            # relative observation time w.r.t. the start of mission
		#
		# Get velocity axis
		n_pixl = data_OTF.field('MAXIS1')[lin]                     # number of pixels, vv: velocity array
		vv = (np.float(hdu_OTF[1].header['CRVAL1']) + (1 + np.arange(n_pixl) - data_OTF.field('CRPIX1')[lin]) * data_OTF.field('CDELT1')[lin])/1.e3
		#
		# Get spatial position of CII in RADec(pos2) and Galactic coord.(pos3)
		pos2 = SkyCoord( np.float(hdu_OTF[1].header['UDP_RA'])*u.deg, np.float(hdu_OTF[1].header['UDP_DEC'])*u.deg, frame='icrs')
		pos3 = pos2.transform_to('galactic')
		pos_all[i1,0] = pos3.l.degree                              # save galactic longitude in pos_all array
		pos_all[i1,1] = pos3.b.degree                              # save galactic latitude in pos_all array
		#
		# ambient scans and y factor interpoaltion at OTF time
		y_fit = np.zeros(nchan)                                    # y_fit: array to save interpolated hot/sky
		hot_fit = np.zeros(nchan)                                  # hot_fit: array to save interpolated hot_ref
		hot_otf_fit = np.zeros(nchan)                              # hot_fit: array to save interpolated hot_otf
		for j in range(0,nchan):
			f0 = interp1d((time_tot-start_time)/1.e3, y_factor_all[:,j], kind='linear')
			y_fit[j] = f0((obs_time_OTF-start_time)/1.e3)
			f1 = interp1d((time_tot-start_time)/1.e3, hot_all[:,j], kind='linear')
			hot_fit[j] = f1((obs_time_OTF-start_time)/1.e3)
			f2 = interp1d((time_tot-start_time)/1.e3, hot_otf_all[:,j], kind='linear')
			hot_otf_fit[j] = f2((obs_time_OTF-start_time)/1.e3)
		#
		# Tsys calculation
		y_fit = smoothing(y_fit,window_len = 3, window='flat')     # smooth y_fit 3 channels before Tsys calculation for noise reduction
		Tsys = (Thot - y_fit * 45.) / (y_fit - 1.)
		# Sky spectrum estimation. If there is hot_otf, use hot_otf. If not, use SKY spectra chosen either the first or last OTF dump. See line 978
		if hot_otf_exist:
			REF = hot_otf_fit/y_fit
		else:
			REF = np.zeros(nchan)
			if ((obs_time_OTF-obs_time_OTF_start)/1.e3 >= OTF_time_start_end[0]) & ((obs_time_OTF-obs_time_OTF_start)/1.e3 <= OTF_time_start_end[1]):
				for j in range(0,nchan):
					func_ref_interp =  interp1d(OTF_time_start_end, SKY_start_end[:,j], kind='linear')
					REF[j] = func_ref_interp((obs_time_OTF-obs_time_OTF_start)/1.e3)
			else:
				for j in range(0,nchan):
					func_ref_extrap =  interp1d(OTF_time_start_end, SKY_start_end[:,j], kind='nearest', fill_value='extrapolate')
					REF[j] = func_ref_extrap((obs_time_OTF-obs_time_OTF_start)/1.e3)
		#
		#
		TA = Tsys*(int_OTF[lin,:]-REF)/REF                         # Ta conversion
		TA = TA-np.median(TA)                                      # constant offset correction
		TA = TA*2.                                                 # Double side band correction
		TA = fixdata(TA, badpix, buffer=25, limit = 1.)            # bad pixel correction
		TA_original[i1,:] = TA[:]                                  # save results in TA_original
		TPower[i1,0] = TA.sum()                                    # total sum of intensity
		TPower[i1,1] = (TA**2).sum()                               # total intensity power
		bar.update(i1)                                             # progress bar update


	# if there is no good spectra, continue to next spectra
	if len(good_raw_spec) == 0:
		print('No good spectrum in this scan. Proceeding to next scan.',flush = True)
		continue
	# if there is any good spectra, then proceed	
	#===
	# Clustering spectra which are in adjoint along time or within a single raster OTF.  
	# DBSCAN is used to find adjoint OTF dumps along the time automatically.
	# The automatic clustering is essential step for the STO2 data since the characteristics of the spectra baseline significantly chages 
	# whenever there are REF observations or any adjustment in the receiver system.
	# Only spectra within a single OTF scan share a similar charactersitics, thus clustering adjoint OTF dumps is required before characterizing
	# a group of spectra
	#===
	print('Determining group of spectra using DBSCAN', flush = True)
	if len(file_hot) > 2:
		x = (obs_time[good_raw_spec]-np.min(obs_time[good_raw_spec]))/(np.max(obs_time[good_raw_spec])-np.min(obs_time[good_raw_spec])) # x is obervation time
		y = np.ones(len(x))                                        # dummy array since DBSCAN requires 2D array
		data = np.array([x,y]).transpose()
		db = DBSCAN(eps=0.011, min_samples=2).fit(data)            # run initial DBSCAN with eps =0.011
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)  # setting up boolean array for label of clustered spectra
		core_samples_mask[db.core_sample_indices_] = True          # cluster spectra == True
		labels = db.labels_                                        # label # for each cluster
		cut = (labels == -1)                                       # -1 indicates outliers
		num_outlier = len(labels[cut])                             # number of outliers within a raster line
		eps_inc = 0.01
		while (num_outlier >= 2):                                  # if there are outliers more than 2 spectra, redo the clustering with increasing tolerance label. Too many outliers indicate tolerance for DBSCAN is too small and creating an excessive number of clusters. 
			db = DBSCAN(eps=eps_inc, min_samples=2).fit(data)
			core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
			core_samples_mask[db.core_sample_indices_] = True
			labels = db.labels_
			cut = (labels == -1)
			num_outlier = len(labels[cut])
			eps_inc = eps_inc+0.01
		
		# Number of clusters in labels, ignoring outlier if present.
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		# Output spectra label as labels_xxxxx.fits
		labels_tot = np.zeros(len(file_OTF))-1.
		labels_tot[good_raw_spec] = labels[:]
		silentremove(dir_data_out+'labels_'+'{0:05d}'.format(i0)+'.fits')		
		hdu = fits.PrimaryHDU(labels_tot)
		hdu.writeto(dir_data_out+'labels_'+'{0:05d}'.format(i0)+'.fits')
		#
		# Black removed and is used for noise instead.
		unique_labels = set(labels)                                  # save unique label numbers
		colors = plt.cm.prism(np.linspace(0, 1, len(unique_labels))) # color for unique label number for plot below
		data_original = np.array([obs_time[good_raw_spec],TPower[good_raw_spec,1]/1.e4]).transpose() # data_original contains observation time and Total power/10,000
		# Plot clusters of spectra group in the observation time vs total power space. Each cluster is distinguished with different color.
		for k, col in zip(unique_labels, colors):
			if k == -1:
				# Outliers are marked as black (python color code 'k').
				col = 'k'
			class_member_mask = (labels == k)
			xy = data_original[class_member_mask & core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], '.', color=col, markersize=3)
			xy = data_original[class_member_mask & ~core_samples_mask]
			plt.plot(xy[:, 0], xy[:, 1], '.', color='k', markersize=1)	
		plt.xlabel('Time [hours]')
		plt.ylabel(r'Total Power [10$^4$]')
		plt.savefig(dir_image_out+'TP_all'+'{0:05d}'.format(i0)+'.png',dpi=250)
		plt.close("all")
		for k in range(0,n_clusters_):
			for j in good_raw_spec[labels == k]:
				plt.axis([np.min(vv),np.max(vv),-50,50])
				plt.plot(vv, smoothing(TA_original[j,:],window_len=3,window='flat'), drawstyle='steps',lw=0.75)
			plt.xlabel('Velocity [km/s]')
			plt.ylabel('Ta [K]')
			plt.title('TA Scan = '+'{0:05d}'.format(i0)+' Group  = '+'{0:02d}'.format(k))
			plt.savefig(dir_image_out+'TA_all'+'{0:05d}'.format(i0)+'_group_'+'{0:02d}'.format(k)+'.png',dpi=250)
			plt.close("all")
	else:   # For there is only one cluster when a scan contains less than 2 OFT_HOT spectra
		# Just one cluster when len(file_hot) <= 2
		n_clusters_ = 1
		# label output
		labels_tot = np.zeros(len(file_OTF))-1.
		labels_tot[good_raw_spec] = 0
		silentremove(dir_data_out+'labels_'+'{0:05d}'.format(i0)+'.fits')		
		hdu = fits.PrimaryHDU(labels_tot)
		hdu.writeto(dir_data_out+'labels_'+'{0:05d}'.format(i0)+'.fits')
		# Total Power of each spectra
		plt.plot(obs_time, TPower[:,1]/1.e4,'r.',markersize =3)
		plt.xlabel('Time [hours]')
		plt.ylabel(r'Total Power [10$^4$]')
		plt.title('Total Power Scan = '+'{0:05d}'.format(i0))
		plt.savefig(dir_image_out+'TP_all'+'{0:05d}'.format(i0)+'.png',dpi=250)
		plt.close("all")
	print('Number of group is ',n_clusters_,flush = True)
	#
	# plot only good raw spectra
	for i1 in good_raw_spec:
		plt.axis([np.min(vv),np.max(vv),-50,50])
		plt.plot(vv, smoothing(TA_original[i1,:],window_len=3,window='flat'), drawstyle='steps',lw=0.5)
	plt.xlabel('Velocity [km/s]')
	plt.ylabel('Ta [K]')
	plt.title('TAall Scan = '+'{0:05d}'.format(i0))
	plt.savefig(dir_image_out+'TA_all'+'{0:05d}'.format(i0)+'.png',dpi=250)
	plt.close("all")
	#
	# write the zeroth calibration data including Ta, velocity, position. The zeroth calibration is count --> Ta 
	if zero_cal_data_out:
		silentremove(dir_data_out+'TA_all'+'{0:05d}'.format(i0)+'.fits')	
		hdu = fits.PrimaryHDU(TA_original[:,:])
		hdu.writeto(dir_data_out+'TA_all'+'{0:05d}'.format(i0)+'.fits')
		silentremove(dir_data_out+'vv_'+'{0:05d}'.format(i0)+'.fits')		
		hdu = fits.PrimaryHDU(vv[:])
		hdu.writeto(dir_data_out+'vv_'+'{0:05d}'.format(i0)+'.fits')
		silentremove(dir_data_out+'pos_all'+'{0:05d}'.format(i0)+'.fits')		
		hdu = fits.PrimaryHDU(pos_all[:,:])
		hdu.writeto(dir_data_out+'pos_all'+'{0:05d}'.format(i0)+'.fits')
	#
	# plot spectra from the zeroth calibration
	if zero_cal_image_out:
		for i1 in good_raw_spec:
			plt.axis([np.min(vv),np.max(vv),-50,50])
			plt.plot(vv, smoothing(TA_original[i1,:],window_len=3,window='flat'), drawstyle='steps',lw=0.5)
			plt.xlabel('Velocity [km/s]')
			plt.ylabel('Ta [K]')
			plt.savefig(dir_line_image+'TA'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i1)+'.png',dpi=250)
			plt.close("all")
	#
	'''
	#===
	# Start the level 1 calibration. The level 1 calibration contins the decision tree, which evaluates the characteristics of clustered spectrum 
	# group, a large fringe correction using a deflational ICA, and a baseline correction using the asymmetric least square and the wavelet
	# transform. 	
	#===
	'''
	print('Processing the first level reduction',flush = True)
	TA_1stcal = np.zeros([len(file_OTF),nchan])
	reduction = np.zeros(len(file_OTF), dtype = int) -1
	spec_index = np.arange(0,len(file_OTF), dtype = int)
	for k in range(0,n_clusters_):
		# total power minimum and standar deviation of total power within a single cluster. Values are used in the decision tree
		TP_min = np.min(TPower[labels_tot == k, 1])
		TP_std = np.std(TPower[labels_tot == k, 1])
		# Create the phase portrait of spectra
		center, h2d, hx, hy = phase_portrait(vv, TA_original[labels_tot == k,:], plot = True, scan=i0, group =k, dir = dir_image_out)
		#
		# Standard deviation at each channel within a single cluster
		std = np.zeros(nchan)
		win_line, win_baseline = find_single_window(vv, TA_original[labels_tot == k,:], badpix, win_peak, win_limit, 3., 0.)
		for j in range(0,nchan):
			std[j] = np.std(TA_original[labels_tot == k,j])
		std = fixdata(std, badpix, buffer=25, limit = 1.)
		std_lowfreq = wT_lowfreq(std) 
		std = smoothing(std,window_len=5, window='flat')
		plt.axis([np.min(vv),np.max(vv),0,np.max(std)*1.1])
		plt.plot(vv, std, drawstyle='steps',lw=0.75)
		plt.plot(vv, std_lowfreq, drawstyle='steps',lw=0.75)
		plt.plot(vv[win_line], std[win_line], drawstyle='steps',color='#ff0000',lw=0.75)
		plt.xlabel('Velocity [km/s]')
		plt.ylabel('Ta [K]')
		plt.title('Standard Deviation through Spectra Scan = '+'{0:05d}'.format(i0)+' Group  = '+'{0:02d}'.format(k))
		plt.savefig(dir_image_out+'StD_'+'{0:05d}'.format(i0)+'_group_'+'{0:02d}'.format(k)+'.png',dpi=250)
		plt.close("all")
		#===
		# Proceed to Decision Tree
		# Determine whether spectra within a given cluster has large fringes. 
		# TP_min less than 50000 is relatively good. Dev for flat with rms =4 K is 15000
		#===
		TA_in_range = (center[0] >= -PP_ta) & (center[0] <= PP_ta)
		dTAdv_in_range = (center[1] >= -PP_dtadv) & (center[1] <= PP_dtadv)
		TP_std_in_range = TP_std < TP_std_crit_bad
		fringe_is_small = (TA_in_range & dTAdv_in_range) & TP_std_in_range
		#reduce data with small fringes
		if fringe_is_small:
			if TP_std >= TP_std_crit_good:
#				win_line, win_baseline = find_single_window(vv, TA_original[labels_tot == k,:], badpix, win_peak, win_limit, 3., 0.)
				reduction[labels_tot == k] = 1
				win_line, win_baseline, npeaks_roi = find_window(vv, TA_original[labels_tot == k,:], 2, win_peak, win_limit, 3.5, 0., plot = False, scan= i0, group=k, dir = dir_image_out)
				# Synthesize baseline
				S_lowfreq, A_ = DeBase_ICA(TA_original[labels_tot == k], win_line, sigma_limit = 2.5, buffer = 50, plot_comp = False, plot_line = False, scan = i0, group = k, dir= dir_image_out)
				spectra = TA_original[labels_tot == k].copy()
				size_group = spectra.shape
				# Debase
				spec_DeBased = np.zeros([size_group[0],size_group[1]])
				for jj in range(0,size_group[0]):
					spec_DeBased[jj] = spectra[jj]-np.dot(S_lowfreq,A_[jj,:])
				TA_1stcal[labels_tot == k,:] = spec_DeBased[:,:]		
			else:
				reduction[labels_tot == k] = 2
				TA_defringed, S_, A_ = DeFringe_ICA(TA_original[labels_tot == k,:],n_comps=4, plot=False, scan = i0, group = k, dir= dir_line_image)
				TA_1stcal[labels_tot == k,:] = TA_defringed[:,:]
		else:
			TA_defringed, S_, A_ = DeFringe_ICA(TA_original[labels_tot == k,:],n_comps=4,plot=True, scan = i0, group = k, dir= dir_line_image)
			win_line, win_baseline = find_single_window(vv, TA_defringed, badpix, win_peak, win_limit, 3., 0.)
			TPower_new = CalPower(TA_defringed,mask=win_baseline)
			# Recheck data quality 
			# if data still have very large deviation, it is bad data. Reduction is False
			if np.std(TPower_new) >  TP_std_crit_new: 
				reduction[labels_tot == k] = -2
				TA_1stcal[labels_tot == k,:] = TA_defringed[:,:]
			#
			# if data have small deviation then start to debase 
			else:
				reduction[labels_tot == k] = 3
				# find spectra with the lowest baseline within a group. This is to correct false absorption feature
				lowest_spec_idx = np.argmin(TA_defringed.sum(axis=1))	
				lowest_spec = smoothing(smoothing(TA_defringed[lowest_spec_idx,:]),window_len=101)
				base_corrected_spec = TA_defringed[:,:] - lowest_spec[None,:]
				nelements, nchan1 = base_corrected_spec.shape 
				for jj in range(0,nelements):
					base_corrected_spec[jj,:] = base_corrected_spec[jj,:] - np.median(base_corrected_spec[jj,:])
				# find line and setup 
				win_line, win_baseline, npeaks_roi = find_window(vv, base_corrected_spec, 2, win_peak, win_limit, 3.5, 0., plot = True, scan= i0, group=k, dir = dir_image_out)
				# Run ICA for Debase
				S_lowfreq, A_ = DeBase_ICA(base_corrected_spec, win_line, sigma_limit = 2.5, buffer = 50, plot_comp = True, plot_line = False, scan = i0, group = k, dir= dir_image_out)
				spectra = base_corrected_spec.copy()
				size_group = spectra.shape
				# Debase
				spec_DeBased = np.zeros([size_group[0],size_group[1]])
				for jj in range(0,size_group[0]):
					spec_DeBased[jj] = spectra[jj]-np.dot(S_lowfreq,A_[jj,:])
				TA_1stcal[labels_tot == k,:] = spec_DeBased[:,:]
		print('Group = ',k,', TP min & TP standard deviation = ',TP_min,', ',TP_std,', Reduction tag = ',reduction[labels_tot == k][0],flush=True)
	#
	# 1stcal plot all spec
	for k in range(0,n_clusters_):
		spectra = TA_1stcal[labels_tot == k,:]
		size_group = spectra.shape
		for jj in range(0,size_group[0]):
			plt.axis([np.min(vv),np.max(vv),-50,50])
			plt.plot(vv, smoothing(spectra[jj,:],window_len=3,window='flat'), drawstyle='steps',lw=0.5)
			plt.xlabel('Velocity [km/s]')
			plt.ylabel('Ta [K]')
		plt.savefig(dir_image_out+'TA1stcal_group'+'{0:05d}'.format(i0)+'_'+'{0:02d}'.format(k)+'.png',dpi=250)
		plt.close("all")
	#
	# 1stcal plot each spec
	if first_cal_image_out:
		for i1 in good_raw_spec:
			plt.axis([np.min(vv),np.max(vv),-50,50])
			plt.plot(vv, smoothing(TA_1stcal[i1,:],window_len=3,window='flat'), drawstyle='steps',lw=0.5)
			plt.xlabel('Velocity [km/s]')
			plt.ylabel('Ta [K]')
			plt.savefig(dir_line_image+'TA1stcal_'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i1)+'.png',dpi=250)
			plt.close("all")
	#	
	# 1stcal data out
	if first_cal_data_out:
		#
		silentremove(dir_data_out+'TA_1stcal'+'{0:05d}'.format(i0)+'.fits')	
		hdu = fits.PrimaryHDU(TA_1stcal[:,:])
		hdu.writeto(dir_data_out+'TA_1stcal'+'{0:05d}'.format(i0)+'.fits')
		#
		silentremove(dir_data_out+'reduction'+'{0:05d}'.format(i0)+'.fits')	
		hdu = fits.PrimaryHDU(reduction)
		hdu.writeto(dir_data_out+'reduction'+'{0:05d}'.format(i0)+'.fits')


		for  i1 in good_raw_spec:
			silentremove(dir_data_out+'TA_1stcal_SDFITS'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i1)+'.fits')	
			hdu_temp = fits.open(file_OTF[i1])
			hdu_temp[1].data['data'][2][:] = TA_1stcal[i1,:]
			hdu_temp[1].header.set('GROUP','{0:04d}'.format(reduction[i1]))
			hdu_temp[1].header.set('CALLINE','CII_0')
			hdu_temp[1].header.set('BADPIX','480-490')
			hdu_temp.writeto(dir_data_out+'TA_1stcal_SDFITS'+'{0:05d}'.format(i0)+'_'+'{0:05d}'.format(i1)+'.fits')
#			hdu_temp.close()

	
	print('Reduction done for scan ',i0,flush = True)




