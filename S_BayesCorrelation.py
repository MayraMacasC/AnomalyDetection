# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:56:15 2019

@author: Mayra
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(palette='husl') 

import pymc3 as pm
import numpy as np
import theano.tensor as T

from theano.printing import Print

def mad(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)


def covariance(sigma, rho):
    C = T.alloc(rho, 2, 2)
    C = T.fill_diagonal(C, 1.)
    S = T.diag(sigma)
    return S.dot(C).dot(S)


def analyze_robust(data):
    with pm.Model() as model:
        # priors might be adapted here to be less flat
        mu = pm.Normal('mu', mu=0., tau=0.000001, shape=2, testval=np.median(data, axis=1))
        sigma = pm.Uniform('sigma', lower=0, upper=1000, shape=2, testval=mad(data.T, axis=0))
        rho = pm.Uniform('r', lower=-1., upper=1., testval=0.5)
        sigma_p = Print('sigma')(sigma)

        cov = pm.Deterministic('cov', covariance(sigma_p, rho))
        nu = pm.Exponential('nu_minus_one', lam=1./29.) + 1
        mult_t = pm.MvStudentT('mult_t', nu=nu, mu=mu, Sigma=cov, observed=data.T)

    return model

def process_model (data):
    #__spec__ = None
    robust_model = analyze_robust(data)
    with robust_model:
        robust_trace = pm.sample(5, tune=2, cores=1)
    pm.traceplot(robust_trace, varnames=['r', 'sigma', 'mu'])
    pm.autocorrplot(robust_trace, varnames=['r', 'nu_minus_one', 'sigma', 'mu'])
    pm.plot_posterior(robust_trace, varnames=['r'])
    sumt = pm.summary(robust_trace, ['r'])
    return (sumt.iat[0,0])