import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
from utils import LoadParams, rescale
from constants import *



# def getCoeff(nbls, Fdrive, Adrive, phi, Qm):

#     # Set time vector
#     T = 1 / Fdrive
#     t = np.linspace(0, T, NPC_FULL)
#     dt = t[1] - t[0]

#     # Run STIM ON simulation and retrieve deflection and gas content vectors from last cycle
#     (_, y_on, _) = nbls.runMech(Adrive, Fdrive, phi, Qm)
#     (_, Z, _) = y_on
#     deflections = Z[-NPC_FULL:]

#     # Compute membrane capacitance and potential vectors
#     capacitances = np.array([nbls.Capct(ZZ) for ZZ in deflections])
#     elastance_integral = np.trapz(1 / capacitances, dx=dt)
#     Vmeff = Qm * elastance_integral / T

#     return Vmeff


def target(x):
    return np.interp(x, Qm, Vmeff)


def posterior(bo, x, xmin=0, xmax=150.e-5):
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma = bo.gp.predict(x, return_std=True)
    return mu, sigma

def plot_gp(bo, x, y):

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(bo.X)), fontdict={'size':30})

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    mu, sigma = posterior(bo, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((0., 150.e-5))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})

    utility = bo.util.utility(x, bo.gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((0., 150.e-5))
    # acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)



filepath = 'C:/Users/admin/Google Drive/PhD/NBLS model/Output/lookups 0.35MHz charge extended/lookups_a32.0nm_f350.0kHz_A100.0kPa_dQ1.0nC_cm2.pkl'
filepath0 = 'C:/Users/admin/Google Drive/PhD/NBLS model/Output/lookups 0.35MHz charge extended/lookups_a32.0nm_f350.0kHz_A0.0kPa_dQ1.0nC_cm2.pkl'

with open(filepath, 'rb') as fh:
    lookup = pickle.load(fh)
    Qm = lookup['Q']
    Vmeff = lookup['V_eff']

with open(filepath0, 'rb') as fh:
    lookup = pickle.load(fh)
    Vmbase = lookup['V_eff']

Vmeff = -(Vmeff - Vmbase)


nQ = 100
x = np.linspace(0., 150., nQ).reshape(-1, 1) * 1e-5
y = np.empty(nQ)
for i in range(nQ):
    y[i] = target(x[i])
fig, ax = plt.subplots()
ax.set_xlabel('$Q_m\ (nC/cm^2)$')
ax.set_ylabel('$V_{m, eff}\ (mV)$')
ax.plot(x * 1e5, y)


bo = BayesianOptimization(target, {'x': (0., 150.e-5)})
bo.maximize(init_points=10, n_iter=0, acq='ei', kappa=1)
plot_gp(bo, x, y)

# bo.maximize(init_points=10, n_iter=0, acq='ei', kappa=5)
# plot_gp(bo, x, y)
for i in range(5):
    bo.maximize(init_points=0, n_iter=1, acq='ei', kappa=1)
plot_gp(bo, x, y)

plt.show()


