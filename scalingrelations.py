############################################# H2 Project Scaling Relations #############################################

# Importing necessary modules:
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import emcee
import corner
from scipy import special
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from pylab import *


# FUNCTIONS ############################################################################################################


# Linear function y = mx + b for fitting Star Formation Rate (SFR) vs. MH2 and SFR vs. MHI planes:
def linfunc(xdata, m, c):
    y = m*xdata + c
    return y


########## Functions for SFR vs. MH2 linear scaling relation:


# Prior (uniform, uninformative):
def log_prior_SFRMH2(theta):
    m, const, lnf = theta
    if 0 < m < 1 and 8 < const < 10 and -5 < lnf < 5:
        return 0.0
    return -np.inf


# Posterior probability:
def log_probability_SFRMH2(theta, x, y, x2, y2, S, S2, w, w2):
    m, const, lnf = theta
    v = np.array([-m, 1.0])
    # calculating sigma squared for detections:
    sigma2 = np.dot(np.dot(S, v), v) + np.exp(2 * lnf) 
    #note: sigma calculation is: [sigma^ 2 = slope times xerr^2 + yerr^2 + scatter(f)^2]
    #note2: this comes [sigma^2 = (measurement error)^2 + (instrinsic scatter)^2]
    # calculating sigma for non-detections:
    sigma = (np.dot(np.dot(S2, v), v) + np.exp(2 * lnf))**0.5
    # calculating difference between value (y) and model for detections:
    deltaN = y - (m * x) - const
    # calculating model for non-detections:
    model = (m * x2) + const
    # calculating likelihood for all detections:
    ll1 = -0.5 * np.sum(deltaN ** 2 / sigma2 + np.log(sigma2) + np.log(1/(w*w)))
    # considering non-detections:
    I = np.zeros(len(x2))
    for i in range(0,len(x2)):
        # integrate gaussian likelihood from 0 mass to upper limit mass for each non-detection:
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((y2[i] - model[i]) / ((2 ** 0.5) * sigma[i])) + 1))
        # note1: the root 2pi factor comes from the fact that we ignore this factor in the ll1 likelihood, so we have to
            # account for eliminating this factor on this likelihood calculation as well otherwise we cant add ll1+ll2
        # note2: the error function is used to integrate the Gaussian function within our limits, this significantly
            # reduces the time it takes to compute all the integrals for all the upper limits
    # calculating likelihood for all non-detections:
    ll2 = np.sum(I + np.log(w2))
    return ll1 + ll2 + log_prior_SFRMH2(theta) # calculating posterior, product of detection & non detection likelihoods and prior


# Plotting corner:
def plot_corner_SFRMH2(samples_input):
    samples_input[:, 2] = np.exp(samples_input[:, 2]) # taking exponent of ln(scatter) to get scatter
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["slope", "y-intercept", "scatter"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('sfrH2corner.pdf', format='pdf', dpi=300, transparent=False)


########## Functions for SFR vs. MHI linear scaling relation:


# Prior (uniform, uninformative):
def log_prior_SFRMHI(theta):
    m, const, lnf = theta
    if 0 < m < 10 and 6 < const < 12 and -5 < lnf < 5:
        return 0.0
    return -np.inf


# Posterior probability:
def log_probability_SFRMHI(theta, x, y, x2, y2, S, S2, w, w2):
    m, const, lnf = theta
    v = np.array([-m, 1.0])
    # calculating sigma squared for detections:
    sigma2 = np.dot(np.dot(S, v), v) + np.exp(2 * lnf)
    # calculating sigma for non-detections:
    sigma = (np.dot(np.dot(S2, v), v) + np.exp(2 * lnf))**0.5
    deltaN = y - (m * x) - const # for detections
    model = (m * x2) + const # for non-detections
    # calculating likelihood for all detections:
    ll1 = -0.5 * np.sum(deltaN ** 2 / sigma2 + np.log(sigma2) + np.log(1/(w*w)))
    I = np.zeros(len(x2))
    # considering non-detections:
    for i in range(0,len(x2)):
        # integrate gaussian likelihood from 0 mass to upper limit mass for each non-detection:
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((y2[i]-model[i]) / ((2 ** 0.5) * sigma[i])) + 1))
    # calculating likelihood for all non-detections:
    ll2 = np.sum(I + np.log(w2))
    return ll1 + ll2 + log_prior_SFRMHI(theta) # calculating posterior, product of detection & non detection likelihoods and prior


# Plotting corner:
def plot_corner_SFRMHI(samples_input):
    samples_input[:, 2] = np.exp(samples_input[:, 2])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["slope", "y-intercept", "scatter"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('sfrHIcorner.pdf', format='pdf', dpi=300, transparent=False)


########## Functions for both SFR vs. MH2 and SFR vs. MHI relations:


# Building x- and y-error matrix (in our case it is diagonal since errors aren't correlated)
def S_error(x_err,y_err):
    N = len(x_err)
    S = np.zeros((N, 2, 2))
    for n in range(N):
        L = np.zeros((2, 2))
        L[0, 0] = np.square(x_err[n])
        L[1, 1] = np.square(y_err[n])
        S[n] = L
    return S


# Defining error shading for fit on scaling relation plot:
def shading_linear(sampler_input, samples_input, x_input):
    lnprob = sampler_input.lnprobability[:, 200:].reshape(-1) # taking all log probabilities (except first 200)
    posterior_percentile = np.percentile(lnprob, 31.7) # taking only probabilities within 1sigma
    onesigma = samples_input[np.where(lnprob > posterior_percentile)] # taking samples from these 1sigma probabilities
    # Building error region for shading fit
    y_fits = []
    for i in range(len(onesigma)):
        y_fits.append(linfunc(x_input, onesigma[i][0], onesigma[i][1]))
    y_fits = np.array(y_fits)
    y_max = []
    y_min = []
    for i in range(len(x_input)): # for each x interval, find max and min of fits to shade between
        y_max.append(max(y_fits[:, i]))
        y_min.append(min(y_fits[:, i]))
    y_max = np.array(y_max)
    y_min = np.array(y_min)
    return y_max, y_min


# Sorting xCOLD GASS data (SFR vs. MH2):
def sort_xCOLDGASS():
    # loading data:
    xCOLDGASS = fits.open('Survey_Data/xCOLDGASS_PubCat.fits') # obtained from http://www.star.ucl.ac.uk/xCOLDGASS
    xCOLDGASS = Table(xCOLDGASS[1].data).to_pandas()
    flag_CO, sfr, sfr_err = xCOLDGASS['FLAG_CO'].values, xCOLDGASS['LOGSFR_BEST'].values, xCOLDGASS['LOGSFR_ERR'].values
    weights = xCOLDGASS['WEIGHT'].values
    # defining an array of MH2 for both detections and non detections:
    indUpperLimit = np.where(flag_CO == 2)  # indices for galaxies with upper limits
    indNoUpperLimit = np.where(flag_CO == 1)  # indices for galaxies without upper limits
    mass_H2 = np.zeros(len(xCOLDGASS), dtype=float)
    mass_H2[indNoUpperLimit] = xCOLDGASS['LOGMH2'].values[indNoUpperLimit]
    mass_H2[indUpperLimit] = xCOLDGASS['LIM_LOGMH2'].values[indUpperLimit]
    # defining array for error in MH2, including 1sigma error for upper limits:
    error_H2 = np.zeros(len(xCOLDGASS), dtype=float)
    error_H2[indNoUpperLimit] = xCOLDGASS['LOGMH2_ERR'].values[indNoUpperLimit]
    error_H2[indUpperLimit] = 0.14 # 1sigma error for upper limits
    # defining indices for detections and non detections, and where values are finite:
    ind_det = np.where((sfr > -6) & (mass_H2 > 6) & (flag_CO == 1))
    ind_nondet = np.where((sfr > -6) & (mass_H2 > 6) & (flag_CO == 2))
    # building matrix of x and y errors:
    S_det = S_error(sfr_err[ind_det], error_H2[ind_det])
    S_nondet = S_error(sfr_err[ind_nondet], error_H2[ind_nondet])

    x_scale = np.linspace(-3,2,100) # x range

    # running emcee:
    ndim, nwalkers = 3, 100 # we will be sampling 3 parameters with 100 walkers
    initial=np.array([0.7,9.0,-1]) # initial guess where walkers will start
    pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)] # generating first proposal by taking initial guess and adding noise
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_SFRMH2,
                                      args=(sfr[ind_det], mass_H2[ind_det], sfr[ind_nondet], mass_H2[ind_nondet],
                                            S_det, S_nondet, weights[ind_det], weights[ind_nondet]))
    sampler.run_mcmc(pos, 1000) # obtain 1000 samples
    samples = sampler.chain[:, 200:, :].reshape((-1, ndim)) # get rid of first 200 samples (not converged yet) and flatten list
    
    plot_corner_SFRMH2(samples) # plotting corner
    shading = shading_linear(sampler, samples, x_scale) # get error shading within 1sigma posteriors
    y_best = linfunc(x_scale, np.median(samples[:,0]),np.median(samples[:,1])) # best fit, taken with median of samples

    # plotting data and emcee fit results:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_scale, y_best, color='k', linewidth=1.5, label='Best fit (MCMC)', zorder=2)
    ax.errorbar(sfr[ind_det], mass_H2[ind_det], yerr=error_H2[ind_det], xerr=sfr_err[ind_det],
                fmt='o',
                markersize=4, linewidth=0.4, mew=1, capsize=2,
                capthick=0.5, mec='midnightblue', mfc="cornflowerblue", ecolor='cornflowerblue',
                label="xCOLD GASS detections", zorder=3)
    ax.errorbar(sfr[ind_nondet], mass_H2[ind_nondet], xerr=sfr_err[ind_nondet], fmt='v', markersize=5,
                linewidth=0.4,
                mew=1, capsize=2,
                capthick=0.5, mec='darkred', mfc="salmon", ecolor='salmon',
                label="xCOLD GASS non-detections", zorder=3)
    ax.fill_between(x_scale, shading[0], shading[1], facecolor='darkgray', zorder=1)
    leg = ax.legend(fancybox=True, prop={'size': 12})
    leg.get_frame().set_alpha(1.0)
    ax.set_xlabel("$\mathrm{log\, SFR\, [M_{\odot}\, yr^{-1}]}$", fontsize=16)
    ax.set_ylabel("$\mathrm{log\, M_{H_{2}}\, [M_\odot]}$", fontsize=16)
    ax.set_xlim([-3, 2])
    ax.set_ylim([7, 11])
    plt.savefig('sfrH2.pdf', format='pdf', dpi=300, transparent=False)

    
# Sorting xGASS data (SFR vs. MHI):
def sort_xGASS():
    # Loading xGASS data:
    xGASS_representative = fits.open('Survey_Data/xGASS_representative_sample.fits') # obtained from http://xgass.icrar.org/data.html
    xGASS_representative = Table(xGASS_representative[1].data).to_pandas()
    xGASS_errors = fits.open('Survey_Data/xGASS_RS_final_Serr_180903.fits') # obtained through private communication with xGASS team
    xGASS_errors = Table(xGASS_errors[1].data).to_pandas()
    xGASS = pd.merge(xGASS_representative, xGASS_errors, left_index = True, right_index = True, how = 'outer') # merging all data
    # obtaining desired quantities:
    sfr, mass_HI, flag, weights = np.log10(xGASS['SFR_best'].values), xGASS['lgMHI_x'].values, xGASS['HIsrc_x'].values, xGASS['weight'].values
    # Taking indices of detections and non detections, and finite values of SFR:
    ind_det = np.where((xGASS['SFR_best'] > -80) & (flag != 4)) # detection
    ind_nondet = np.where((xGASS['SFR_best'] > -80) & (flag == 4)) # non detection
    # Creating error arrays:
    xerr = xGASS['SFRerr_best'].values / (xGASS['SFR_best'].values * np.log(10))  # propagating error into log SFR
    errMHI = np.zeros(len(xGASS))
    yerr = np.zeros(len(xGASS))
    errMHI[ind_det] = ((2.356 * 10E5) / (1 + xGASS['zHI_x'].values[ind_det])) * ((xGASS['Dlum'].values[ind_det]) ** 2) \
                      * (xGASS['Serr'].values[ind_det]) # calculating MHI error from HI line flux error, standard propagation of uncertainties
    yerr[ind_det] = errMHI[ind_det] / ((10 ** mass_HI[ind_det]) * np.log(10)) # propagating MHI error to log
    yerr[ind_nondet] = 0.09  # non-detections errors (from standard propagation of uncertainties)
    # Building error matrices:
    S_det = S_error(xerr[ind_det], yerr[ind_det])
    S_nondet = S_error(xerr[ind_nondet], yerr[ind_nondet])

    # running emcee:
    ndim, nwalkers = 3, 100 # probing 3 parameter spaces with 100 walkers
    initial = np.array([0.7, 9.0, -1]) # initial guess
    pos = [initial + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)] # generate first proposal by adding random noise to initial guess
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_SFRMHI,
                                      args=(sfr[ind_det], mass_HI[ind_det], sfr[ind_nondet], mass_HI[ind_nondet],
                                            S_det, S_nondet, weights[ind_det], weights[ind_nondet]))
    sampler.run_mcmc(pos, 1000) # get 1000 samples
    samples = sampler.chain[:, 200:, :].reshape((-1, ndim)) # discard first 200 samples (not converged yet) and flatten list
    plot_corner_SFRMHI(samples)  # plotting corner

    x_scale = np.linspace(-3.2,2,100) # x range

    y_best = linfunc(x_scale, np.median(samples[:, 0]), np.median(samples[:, 1]))  # best fit from medians
    shading = shading_linear(sampler, samples, x_scale)  # shading error region on fit

    # plotting result:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(sfr[ind_det], mass_HI[ind_det], yerr=yerr[ind_det],
                xerr=xerr[ind_det], fmt='o',
                markersize=4, linewidth=0.4, mew=1, capsize=2,
                capthick=0.5, mec='midnightblue', mfc="cornflowerblue", ecolor='cornflowerblue',
                label="xGASS detections", zorder=3)
    ax.errorbar(sfr[ind_nondet], mass_HI[ind_nondet], xerr=xerr[ind_nondet], fmt='v',
                markersize=5, linewidth=0.4,
                mew=1, capsize=2,
                capthick=0.5, mec='darkred', mfc="salmon", ecolor='salmon',
                label="xGASS non-detections", zorder=3)
    ax.plot(x_scale, y_best, color='k', linewidth=1.5, label='Best fit (MCMC)', zorder=2)
    ax.fill_between(x_scale, shading[0], shading[1], facecolor='darkgray', zorder=1)
    leg = ax.legend(fancybox=True, prop={'size': 12})
    leg.get_frame().set_alpha(1.0)
    ax.set_xlabel("$\mathrm{log\, SFR\, [M_{\odot}\, yr^{-1}]}$", fontsize=16)
    ax.set_ylabel("$\mathrm{log\, M_{HI}\, [M_\odot]}$", fontsize=16)
    ax.set_xlim([-3, 2])
    ax.set_ylim([7, 11])
    plt.savefig('sfrHI.pdf', format='pdf', dpi=300, transparent=False)


########## Functions for main sequence of star forming galaxies (Stellar mass vs. SFR):


# Second order polynomial fitting:
def func(x,a,b,c):
    return (a * (x**2)) + (b * x) + c


# Prior (uniform, uninformative):
def log_prior_MS(theta):
    a, b, c, lnf = theta
    if -1.5 < a < 1.5 and 0 < b < 4 and -20 < c < -1 and -5 < lnf < 10:
        return 0.0
    return -np.inf


# Posterior probability:
def log_probability_MS(theta, x, y, xerr, yerr):
    a, b, c, lnf = theta
    # calulating sigma squared using propagation of uncertainties:
    sigma2 = np.square(xerr)*np.square(2*a*x + b) + np.square(yerr) + np.exp(2 * lnf)
    # calculating difference between data and model:
    deltaN = y - (a * (x**2)) - (b * x) - c
    # calculating log likelihood function:
    ll = -0.5 * np.sum(deltaN ** 2 / sigma2 + np.log(sigma2))
    return ll + log_prior_MS(theta) # returns posterior (prior times likelihood)


# Defining error shading for fit on scaling relation plot:
def shading_MS(sampler_input, samples_input, x_input):
    lnprob = sampler_input.lnprobability[:, 200:].reshape(-1)
    posterior_percentile = np.percentile(lnprob, 31.7)
    onesigma = samples_input[np.where(lnprob > posterior_percentile)]
    y_fits = []
    for i in range(len(onesigma)):
        y_fits.append(func(x_input, onesigma[i][0], onesigma[i][1], onesigma[i][2]))
    y_fits = np.array(y_fits)
    y_max = []
    y_min = []
    for i in range(len(x_input)):
        y_max.append(max(y_fits[:, i]))
        y_min.append(min(y_fits[:, i]))
    y_max = np.array(y_max)
    y_min = np.array(y_min)
    return y_max, y_min


# Plotting corner:
def plot_corner_MS(samples_input):
    samples_input[:, 3] = np.exp(samples_input[:, 3]) # taking exponent of log scatter
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["a", "b", "c", "f"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2]),
                          np.median(samples_input[:, 3])),
                  truth_color="k", quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('MScorner.pdf', format='pdf', dpi=300, transparent=False)

    
# Sorting GAMA data (Main Sequence of star forming galaxies, stellar mass vs. SFR):
def sort_GAMA_starforming():
    GAMA = pd.read_csv('Survey_Data/GAMA_sample.dat', comment='#', header=None, sep=r"\s*", engine="python") # loading GAMA data
    GAMA.columns = ['ID', 'z', 'logM*', 'logM*err', 'logSFR', 'logSFRerr', 'ColorFlag'] # taking columns
    GAMA = GAMA[GAMA['logSFR'] > -7] # taking finite values
    GAMA = GAMA[GAMA['logM*'] > 7]
    GAMA = GAMA[GAMA['ColorFlag'] == 1] # selecting main sequence of star forming galaxies only
    mass_sf, mass_sferr, sfr_sf, sfr_sferr = GAMA['logM*'].values, GAMA['logM*err'].values,\
                                             GAMA['logSFR'].values, GAMA['logSFRerr'].values

    x_scale = np.linspace(7, 12, 100)  # array of x values for main sequence fit

    # creating bins (in stellar mass) along main sequence:
    bins = np.linspace(7.5, 11, 15)
    m_bin, sfr_bin, err_bin = [], [], []
    for i in range(1, len(bins)):
        # print (bins)
        inbin = GAMA[(GAMA['logM*'] >= bins[i - 1]) & (GAMA['logM*'] < bins[i])]
        m_bin.append((bins[i] + bins[i - 1]) / 2)
        sfr_bin.append(np.median(inbin['logSFR']))
        err_bin.append(np.std(inbin['logSFR']))
    m_bin, sfr_bin, err_bin = np.array(m_bin), np.array(sfr_bin), np.array(err_bin)

    # running emcee:
    ndim, nwalkers = 4, 100 # probe 4 parameters with 100 walkers
    initial = np.array([-0.024, 1.3, -12.75, -1]) # initial guess
    pos = [initial + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)] # first proposal: initial guess + random noise
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_MS, args=(mass_sf, sfr_sf, mass_sferr, sfr_sferr))
    sampler.run_mcmc(pos, 1000) # get 1000 samples with MCMC
    samples = sampler.chain[:, 200:, :].reshape((-1, ndim)) # discard first 200 samples (not converged yet) and flatten list
    shading = shading_MS(sampler, samples, x_scale) # creating shading on fit
    y_best = func(x_scale, np.median(samples[:, 0]), np.median(samples[:, 1]), np.median(samples[:, 2])) # best fit

    plot_corner_MS(samples) # plotting corner

    # plot results:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(mass_sf, sfr_sf, yerr=sfr_sferr, xerr=mass_sferr, fmt="o", markersize=1, linewidth=0.5, mew=1, capsize=0,
                mec='lightsteelblue', mfc="lightsteelblue", ecolor='lightsteelblue', label="GAMA star forming",
                zorder=1)
    ax.errorbar(m_bin, sfr_bin, yerr=err_bin, fmt="o", markersize=6, linewidth=1, mew=1.5, capsize=3,
                capthick=1, mec='midnightblue', mfc="cornflowerblue", ecolor='midnightblue', label="GAMA binned",
                zorder=4)
    ax.plot(x_scale, y_best, c="k", linewidth=1.5, label='Best fit (MCMC)', zorder=3)
    ax.fill_between(x_scale, shading[0], shading[1], facecolor='plum', zorder=2)
    leg = ax.legend(fancybox=True, prop={'size': 14})
    leg.get_frame().set_alpha(1.0)
    ax.set_ylabel("$\mathrm{log\, SFR\, [M_{\odot}\, yr^{-1}]}$", fontsize=16)
    ax.set_xlabel("$\mathrm{log\, M_{*}\, [M_\odot]}$", fontsize=16)
    ax.set_xlim([7, 12])
    plt.savefig('MS.pdf', format='pdf', dpi=300, transparent=False)

# MAIN PROGRAM #########################################################################################################

sort_xCOLDGASS()
sort_xGASS()
sort_GAMA_starforming()
