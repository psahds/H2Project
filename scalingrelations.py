import atpy
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import emcee
import corner
from scipy import special
import pandas as pd

# FUNCTIONS ############################################################################################################


# Linear function y = mx + b for fitting SFr vs. MH2 and SFR vs. MHI planes:
def linfunc(xdata, m, c):
    y = m*xdata + c
    return y


# Functions for SFR vs. MH2 linear scaling relation:
def log_prior_SFRMH2(theta):
    m, const, lnf = theta
    if 0 < m < 1 and 8 < const < 10 and -5 < lnf < 5:
        return 0.0
    return -np.inf


def log_probability_SFRMH2(theta, x, y, S, fl):
    m, const, lnf = theta
    v = np.array([-m, 1.0])
    # finding indices for detections and non-detections:
    det = np.where(fl == 1)
    nondet = np.where(fl == 2)
    # calculating sigma squared for both detections and non detections:
    sigma2 = np.dot(np.dot(S, v), v) + np.exp(2 * lnf)
    sigma = sigma2**0.5
    # calculating difference between value (y) and model, as well as calculating model:
    deltaN = y - (m * x) - const
    model = (m * x) + const
    # calculating likelihood for all detections:
    ll1 = -0.5 * np.sum(deltaN[det] ** 2 / sigma2[det] + np.log(sigma2[det]))
    # considering non-detections:
    I = np.zeros(len(nondet[0]))
    for i in range(0,len(nondet[0])):
        # integrate gaussian from 0 mass to upper limit mass for each non-detection:
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((y[nondet][i] - model[nondet][i]) / ((2 ** 0.5) * sigma[nondet][i])) + 1))
        # note1: the root 2pi factor comes from the fact that we ignore this factor in the ll1 likelihood, so we have to
            # account for eliminating this factor on this likelihood calculation as well otherwise we cant add ll1+ll2
        # note2: the error function is used to integrate the Gaussian function within our limits, this significantly
            # reduces the time it takes to compute all the integrals for all the upper limits
    # calculating likelihood for all non-detections:
    ll2 = np.sum(I)
    return ll1 + ll2 + log_prior_SFRMH2(theta) # combining detection & non detection results


def plot_corner_SFRMH2(samples_input):
    samples_input[:, 2] = np.exp(samples_input[:, 2]) # taking exponent of ln(scatter) to get scatter
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["slope", "y-intercept", "scatter"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('sfrH2corner.pdf', format='pdf', dpi=300, transparent=False)


# Prior and probability functions for SFR vs. MHI linear scaling relation:
def log_prior_SFRMHI(theta):
    m, const, lnf = theta
    if 0 < m < 10 and 6 < const < 12 and -5 < lnf < 5:
        return 0.0
    return -np.inf


def log_probability_SFRMHI(theta, x, y, S, fl):
    m, const, lnf = theta
    v = np.array([-m, 1.0])
    # finding indices for detections and non-detections:
    det = np.where(fl != 4)
    nondet = np.where(fl == 4)
    # calculating sigma squared for both detections and non detections:
    sigma2 = np.dot(np.dot(S, v), v) + np.exp(2 * lnf)
    sigma = sigma2**0.5
    deltaN = y - (m * x) - const
    model = (m * x) + const
    # calculating likelihood for all detections:
    ll1 = -0.5 * np.sum(deltaN[det] ** 2 / sigma2[det] + np.log(sigma2[det]))
    I = np.zeros(len(nondet[0]))
    # considering non-detections:
    for i in range(0,len(nondet[0])):
        # integrate gaussian from 0 mass to upper limit mass for each non-detection:
        I[i] = np.log(((2 * np.pi) ** 0.5) *
                      0.5 * (special.erf((y[nondet][i]-model[nondet][i]) / ((2 ** 0.5) * sigma[nondet][i])) + 1))
    # calculating likelihood for all non-detections:
    ll2 = np.sum(I)
    return ll1 + ll2 + log_prior_SFRMHI(theta) # combining detection & non detection results


def plot_corner_SFRMHI(samples_input):
    samples_input[:, 2] = np.exp(samples_input[:, 2])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["slope", "y-intercept", "scatter"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2])),
                  truth_color="k",
                  quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('sfrHIcorner.pdf', format='pdf', dpi=300, transparent=False)


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
    for i in range(len(xsc)): # for each x interval, find max and min of fits to shade between
        y_max.append(max(y_fits[:, i]))
        y_min.append(min(y_fits[:, i]))
    y_max = np.array(y_max)
    y_min = np.array(y_min)
    return y_max, y_min


# Sorting xCOLD GASS data:
def sort_xCOLDGASS(catalogue, x):
    # defining an array of MH2 for both detections and non detections:
    indUpperLimit = np.where(catalogue.FLAG_CO == 2)  # indices for galaxies with upper limits
    indNoUpperLimit = np.where(catalogue.FLAG_CO == 1)  # indices for galaxies with upper limits
    mass_H2 = np.zeros(len(catalogue), dtype=float)
    mass_H2[indNoUpperLimit] = catalogue.LOGMH2[indNoUpperLimit]
    mass_H2[indUpperLimit] = catalogue.LIM_LOGMH2[indUpperLimit]  # total list of MH2 mass, including upper limits
    # defining array for error in MH2, including 1sigma error for upper limits:
    error_H2 = np.zeros(len(catalogue), dtype=float)
    error_H2[indNoUpperLimit] = catalogue.LOGMH2_ERR[indNoUpperLimit]
    error_H2[indUpperLimit] = 0.14 # 1sigma error for upper limits
    # defining indices:
    ind_xcg = np.where((catalogue.LOGSFR_BEST > -6) & (catalogue.LOGSFR_BEST < 6) & (mass_H2 > 4) & (mass_H2 < 20))
    indxcg_det = np.where((catalogue.LOGSFR_BEST > -6) & (catalogue.LOGSFR_BEST < 6) &
                          (mass_H2 > 4) & (mass_H2 < 20) & (catalogue.FLAG_CO == 1))
    indxcg_nondet = np.where((catalogue.LOGSFR_BEST > -6) & (catalogue.LOGSFR_BEST < 6) &
                             (mass_H2 > 4) & (mass_H2 < 20) & (catalogue.FLAG_CO == 2))
    # building matrix of x and y errors:
    S_xcg = S_error(catalogue.LOGSFR_ERR[ind_xcg], error_H2[ind_xcg])
    flag_CO = catalogue.FLAG_CO

    # running emcee:
    ndim, nwalkers = 3, 100
    initial=np.array([0.7,9.0,-1])
    pos = [initial + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    samplerh2 = emcee.EnsembleSampler(nwalkers, ndim, log_probability_SFRMH2,
                                      args=(catalogue.LOGSFR_BEST[ind_xcg], mass_H2[ind_xcg], S_xcg, flag_CO[ind_xcg]))
    samplerh2.run_mcmc(pos, 1000)
    samplesh2 = samplerh2.chain[:, 200:, :].reshape((-1, ndim))
    plot_corner_SFRMH2(samplesh2) # plotting corner
    # error shading for fit:
    shadingh2 = shading_linear(samplerh2, samplesh2, xsc)
    # calculating best fit function from medians of samples:
    ysch2 = linfunc(xsc, np.median(samplesh2[:,0]),np.median(samplesh2[:,1]))

    # plotting data and emcee fit results:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, ysch2, color='k', linewidth=1.5, label='Best fit (MCMC)', zorder=2)
    ax.errorbar(catalogue.LOGSFR_BEST[indxcg_det], mass_H2[indxcg_det], yerr=error_H2[indxcg_det], xerr=catalogue.LOGSFR_ERR[indxcg_det],
                fmt='o',
                markersize=6, linewidth=0.7, mew=1.5, capsize=3,
                capthick=0.5, mec='midnightblue', mfc="cornflowerblue", ecolor='midnightblue',
                label="xCOLD GASS detections", zorder=4)
    ax.errorbar(catalogue.LOGSFR_BEST[indxcg_nondet], mass_H2[indxcg_nondet], xerr=catalogue.LOGSFR_ERR[indxcg_nondet], fmt='v', markersize=6,
                linewidth=0.7,
                mew=1.5, capsize=3,
                capthick=0.5, mec='midnightblue', mfc="cornflowerblue", ecolor='midnightblue',
                label="xCOLD GASS non-detections", zorder=3)
    ax.fill_between(x, shadingh2[0], shadingh2[1], facecolor='plum', zorder=1)
    leg = ax.legend(fancybox=True, prop={'size': 12})
    leg.get_frame().set_alpha(1.0)
    ax.set_xlabel("$\mathrm{log\, SFR\, [M_{\odot}\, yr^{-1}]}$", fontsize=16)
    ax.set_ylabel("$\mathrm{log\, M_{H_{2}}\, [M_\odot]}$", fontsize=16)
    ax.set_xlim([-3, 2])
    ax.set_ylim([7, 10.5])
    plt.savefig('sfrH2.pdf', format='pdf', dpi=300, transparent=False)


def sort_xGASS(catalogue, x):
    # loading variables from table
    sfr_HI = np.log10(catalogue.SFR_best)
    mass_HI = catalogue.lgMHI
    flag_HI = catalogue.HIsrc
    # defining indices:
    ind_xg = np.where((sfr_HI > -6) & (sfr_HI < 6) & (mass_HI > 4) & (mass_HI < 20))
    indxg_det = np.where((sfr_HI > -6) & (sfr_HI < 6) & (mass_HI > 4) & (mass_HI < 20) & (flag_HI != 4))
    indxg_nondet = np.where((sfr_HI > -6) & (sfr_HI < 6) & (mass_HI > 4) & (mass_HI < 20) & (flag_HI == 4))
    # creating error arrays
    xerrxg1 = catalogue.SFRerr_best / (catalogue.SFR_best * np.log(10))  # propagating error into log SFR
    yerrxg1 = np.zeros(len(sfr_HI))
    yerrxg1[indxg_det] = 0.2
    yerrxg1[indxg_nondet] = 0.14  # non-detections errors
    # building error matrix:
    S_xg = S_error(xerrxg1[ind_xg], yerrxg1[ind_xg])

    # running emcee:
    ndim, nwalkers = 3, 100
    initial = np.array([0.7, 9.0, -1])
    pos = [initial + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    samplerHI = emcee.EnsembleSampler(nwalkers, ndim, log_probability_SFRMHI,
                                      args=(sfr_HI[ind_xg], mass_HI[ind_xg], S_xg, flag_HI[ind_xg]))
    samplerHI.run_mcmc(pos, 1000)
    samplesHI = samplerHI.chain[:, 200:, :].reshape((-1, ndim))
    plot_corner_SFRMHI(samplesHI) # plotting corner
    yschI = linfunc(xsc, np.median(samplesHI[:, 0]), np.median(samplesHI[:, 1])) # best fit from medians
    shadinghI = shading_linear(samplerHI, samplesHI, xsc) # shading error region on fit

    # plotting result:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, yschI, color='k', linewidth=1.5, label='Best fit (MCMC)', zorder=2)
    ax.errorbar(sfr_HI[indxg_det], mass_HI[indxg_det], yerr=yerrxg1[indxg_det],
                xerr=xerrxg1[indxg_det], fmt='o',
                markersize=6, linewidth=0.7, mew=1.5, capsize=3,
                capthick=0.5, mec='midnightblue', mfc="cornflowerblue", ecolor='midnightblue',
                label="xCOLD GASS detections", zorder=4)
    ax.errorbar(sfr_HI[indxg_nondet], mass_HI[indxg_nondet], xerr=xerrxg1[indxg_nondet], fmt='v',
                markersize=6, linewidth=0.7,
                mew=1.5, capsize=3,
                capthick=0.5, mec='midnightblue', mfc="cornflowerblue", ecolor='midnightblue',
                label="xCOLD GASS non-detections", zorder=3)
    ax.fill_between(x, shadinghI[0], shadinghI[1], facecolor='plum', zorder=1)
    leg = ax.legend(fancybox=True, prop={'size': 12})
    leg.get_frame().set_alpha(1.0)
    ax.set_xlabel("$\mathrm{log\, SFR\, [M_{\odot}\, yr^{-1}]}$", fontsize=16)
    ax.set_ylabel("$\mathrm{log\, M_{HI}\, [M_\odot]}$", fontsize=16)
    ax.set_xlim([-3, 2])
    ax.set_ylim([7, 11])
    plt.savefig('sfrHI.pdf', format='pdf', dpi=300, transparent=False)


# For main sequence of star forming galaxies:
def func(x,a,b,c):
    return (a * (x**2)) + (b * x) + c


def log_prior_MS(theta):
    a, b, c, lnf = theta
    if -1.5 < a < 1.5 and 0 < b < 4 and -16 < c < -1 and -5 < lnf < 10:
        return 0.0
    return -np.inf


def log_probability_MS(theta, x, y, xerr, yerr):
    a, b, c, lnf = theta
    # calulating sigma squared using propagation of uncertainties:
    sigma2 = np.square(xerr)*np.square(2*a*x + b) + np.square(yerr) + np.exp(2 * lnf)
    deltaN = y - (a * (x**2)) - (b * x) - c
    ll = -0.5 * np.sum(deltaN ** 2 / sigma2 + np.log(sigma2)) # calculating likelihood
    return ll + log_prior_MS(theta)


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
    for i in range(len(xsc)):
        y_max.append(max(y_fits[:, i]))
        y_min.append(min(y_fits[:, i]))
    y_max = np.array(y_max)
    y_min = np.array(y_min)
    return y_max, y_min


def plot_corner_MS(samples_input):
    samples_input[:, 3] = np.exp(samples_input[:, 3])
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    corner.corner(samples_input, labels=["a", "b", "c", "f"],
                  truths=(np.median(samples_input[:, 0]), np.median(samples_input[:, 1]), np.median(samples_input[:, 2]),
                          np.median(samples_input[:, 3])),
                  truth_color="k", quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('MScorner.pdf', format='pdf', dpi=300, transparent=False)


def sort_GAMA_starforming(x_input):
    GAMA = pd.read_csv('GAMA_sample.dat', comment='#', header=None, sep=r"\s*", engine="python") # loading GAMA data
    GAMA.columns = ['ID', 'z', 'logM*', 'logM*err', 'logSFR', 'logSFRerr', 'ColorFlag'] # taking columns
    GAMA = GAMA[np.isfinite(GAMA['logSFR'])]
    GAMA = GAMA[np.isfinite(GAMA['logM*'])]
    GAMA = GAMA[GAMA['logSFR'] > -7]
    GAMA = GAMA[GAMA['logM*'] > 7]
    GAMA = GAMA[GAMA['ColorFlag'] == 1] # selecting main sequence of star forming galaxies only
    mass_sf = np.array(GAMA['logM*'])
    mass_sferr = np.array(GAMA['logM*err'])
    sfr_sf = np.array(GAMA['logSFR'])
    sfr_sferr = np.array(GAMA['logSFRerr'])

    # creating bins along main sequence:
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
    ndim, nwalkers = 4, 100
    initial = np.array([-0.024, 1.3, -12.75, -1])
    pos = [initial + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_MS, args=(mass_sf, sfr_sf, mass_sferr, sfr_sferr))
    sampler.run_mcmc(pos, 1000)
    samples = sampler.chain[:, 200:, :].reshape((-1, ndim))
    shade = shading_MS(sampler, samples, x_input) # creating shading on fit
    y_new = func(x_input, np.median(samples[:, 0]), np.median(samples[:, 1]), np.median(samples[:, 2])) # best fit

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
    ax.plot(x_input, y_new, c="k", linewidth=1.5, label='Best fit (MCMC)', zorder=3)
    ax.fill_between(x_input, shade[0], shade[1], facecolor='plum', zorder=2)
    leg = ax.legend(fancybox=True, prop={'size': 14})
    leg.get_frame().set_alpha(1.0)
    ax.set_ylabel("$\mathrm{log\, SFR\, [M_{\odot}\, yr^{-1}]}$", fontsize=16)
    ax.set_xlabel("$\mathrm{log\, M_{*}\, [M_\odot]}$", fontsize=16)
    ax.set_xlim([7, 12])
    plt.savefig('MS.pdf', format='pdf', dpi=300, transparent=False)

# MAIN PROGRAM #########################################################################################################

# loading xCOLD GASS and xGASS catalogues:
xcg = atpy.Table('xCOLDGASS_PubCat.fits') # loading xCOLD GASS data
xg = atpy.Table('xxGASS_MASTER_CO_170620_final.fits') # loading xGASS data

# x-values array:
xsc = np.linspace(-3,2,100) # array of x values for linear fits
xsc2 = np.linspace(7, 12, 100) # array of x values for main sequence fit

sort_xCOLDGASS(xcg, xsc)
sort_xGASS(xg, xsc)
sort_GAMA_starforming(xsc2)