# Creates all figures and table data in Section 5.1

# Import modules and libraries
import DrumAnalysis as DA
import numpy as np
from scipy.stats import beta 
import scipy.integrate.quadrature as quad
import matplotlib.pyplot as plt

# Set parameters for either fast computation of TV metrics (maxiter=100)
# or to recreate data in paper (maxiter=1500)
print('='*50)
print('='*50)
s = 'Input the maximum number of iterations to use in adaptive quadrature' 
s += ' for estimating TV metrics. \n\nUsing 1500 will recreate all table data.'
s += '\n\nSmaller values (such as 100) will produce less accurate data (but'
s += ' very quickly) that still give the same idea about the order of errors.'
print(s)
print('='*50)
print('='*50)
maxiter = int(input('maximum number of iterations: '))

# Set plotting parameters
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'axes.linewidth': 2})

# Load data sets
init_param_set = DA.load_data_set('prior_data_set_20_1.mat')
init_obs_set = DA.load_data_set('prior_data_set_20_1.mat')
actual_param_set = DA.load_data_set('observed_data_set_20_01.mat')
actual_obs_set = DA.load_data_set('observed_data_set_20_01.mat')

# Use fingerprinted eigen-modes to un-mix predicted eigenvalue data
(ms_init,ns_init,eigs_init) = DA.un_mix(init_obs_set['ms'],
                                      init_obs_set['ns'],
                                      init_obs_set['eig'])

# Use un-mixed predicted eigenvalue data to un-mix observed data in
# the same order as predicted data sorted. 
(ms_actual,ns_actual,eigs_actual) = DA.un_mix(actual_obs_set['ms'],
                                            actual_obs_set['ns'],
                                            actual_obs_set['eig'],
                                            ms_init,ns_init)

# Visualize correlations between eigenvalue data
DA.plot_eigenvalue_pairs(eigs_init, eigs_actual, 0, 1, filename='Example-1-eigen-correlation-1')
DA.plot_eigenvalue_pairs(eigs_init, eigs_actual, 15, 4, filename='Example-1-eigen-correlation-2')

# Learn the QoI through feature extraction (using PCA) after
# first performing a standard scaling of data
X_init_std, scaler = DA.standard_scaling_xform(eigs_init)
PC_vals, PC_vecs = DA.PCA(X_init_std)

# Visualize spectral gaps
DA.plot_gap(0, PC_vals, filename='Example-1-QoI-Gap-1')
DA.plot_gap(1, PC_vals, filename='Example-1-QoI-Gap-2')

# Visualize principal vectors
DA.plot_PC_vecs(0, PC_vals, PC_vecs, filename='Example-1-PC-vector-1')
DA.plot_PC_vecs(1, PC_vals, PC_vecs, filename='Example-1-PC-vector-2')

# Transform observed eigenvalue data into learned QoI
print('------------ Using both QoI --------------')
QoI_num = [0, 1]
qs_init_xform = DA.QoI_xform(X_init_std,PC_vecs,PC_vals,QoI_num)
X_actual_std = DA.standard_scaling_xform(eigs_actual,scaler)
qs_actual_xform = DA.QoI_xform(X_actual_std,PC_vecs,PC_vals,QoI_num)

# Visualize the improved correlation between QoI (both predicted and observed)
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 0, 1, filename='Example-1-QoI-correlations')

# Compute the predicted and observed densities on the QoI
pi_Q_kde = DA.kernel_density_estimate(qs_init_xform)
pi_obs_kde = DA.kernel_density_estimate(qs_actual_xform)

# Construct the ratio of observed to predicted densities and print diagnostic
r = DA.compute_r(pi_obs_kde,pi_Q_kde,qs_init_xform)

# Visualize the ratio over parameter space
plt.figure(figsize=(10,10))
a0_init = init_param_set['a0'][0]
a1_init = init_param_set['a1'][0]
scatter = plt.scatter(a0_init, a1_init, c=r, cmap='hot')
plt.colorbar(scatter,fraction=0.05, pad=0.04)
plt.xlabel('$a_0$')
plt.ylabel('$a_1$')
plt.xlim([0.8,1.2])
plt.ylim([0.1,0.2])
plt.title('$\mathcal{R}(a(r))$')
plt.tight_layout()
plt.savefig('Example-1-ratios-QoI-both.png')

# Compute and visualize the various marginal densities
a0_update_dens = DA.kernel_density_estimate(a0_init, r)
a1_update_dens = DA.kernel_density_estimate(a1_init, r)

a0_obs = actual_param_set['a0beta'] # samples of first parameter responsible for observed data
a1_obs = actual_param_set['a1beta'] # samples of second parameter responsible for observed data
a0_obs_samples = actual_param_set['a0'][0]
a1_obs_samples = actual_param_set['a1'][0]

a0_true_beta = beta(a=a0_obs[0,0],b=a0_obs[0,1],loc=0.8,scale=0.4)
a0_true_kde = DA.kernel_density_estimate(a0_obs_samples)
a1_true_beta = beta(a=a1_obs[0,0], b=a1_obs[0,1],loc=0.1,scale=0.1)
a1_true_kde = DA.kernel_density_estimate(a1_obs_samples)

plt.figure(figsize=(10,10))
x = np.linspace(0.8,1.2,100)
plt.plot(x,a0_true_beta.pdf(x), linestyle='solid', linewidth=4, label='Exact pdf')
plt.plot(x,a0_update_dens.pdf(x), linestyle='dashed', linewidth=4, label='Estimated pdf')
plt.plot(x,a0_true_kde.pdf(x), linestyle='dashdot', linewidth=4, label='Exact KDE')
plt.plot(x,1/0.4*np.ones(len(x)), linestyle='dotted', linewidth=4, label='Initial')
plt.legend()
plt.title('Estimated and Exact Variation in $a_0$')
plt.tight_layout()
plt.savefig('Example-1-both-QoI-a0-pdfs.png')

plt.figure(figsize=(10,10))
x = np.linspace(0.1,0.2,100)
plt.plot(x,a1_true_beta.pdf(x), linestyle='solid', linewidth=4, label='Exact pdf')
plt.plot(x,a1_update_dens.pdf(x), linestyle='dashed', linewidth=4, label='Estimated pdf')
plt.plot(x,a1_true_kde.pdf(x), linestyle='dashdot', linewidth=4, label='Exact KDE')
plt.plot(x,1/0.1*np.ones(len(x)), linestyle='dotted', linewidth=4, label='Initial')
plt.legend()
plt.title('Estimated and Exact Variation in $a_1$')
plt.tight_layout()
plt.savefig('Example-1-both-QoI-a1-pdfs.png')

print('------- Computing TV metrics for $a_0$ --------')
print('-- Using adaptive quadrature (may take a few minutes) --')

def a0_init_soln(x):
    return np.abs(a0_update_dens.pdf(x) - 1/0.4)

TV = quad(a0_init_soln,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from update = ', TV)

def a0_init_error(x):
    return np.abs(a0_true_beta.pdf(x) - 1/0.4)

TV = quad(a0_init_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from Beta = ', TV)

def a0_init_error_kde(x):
    return np.abs(a0_true_kde.pdf(x) - 1/0.4)

TV = quad(a0_init_error_kde,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from KDE of Beta = ', TV)

def a0_dens_error(x):
    return np.abs(a0_true_beta.pdf(x) - a0_update_dens.pdf(x))

TV = quad(a0_dens_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ update from Beta = ', TV)

def a0_dens_estimate_error(x):
    return np.abs(a0_true_kde.pdf(x) - a0_update_dens.pdf(x))

TV = quad(a0_dens_estimate_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ update from KDE of Beta = ', TV)

def a0_sampling_error(x):
    return np.abs(a0_true_kde.pdf(x) - a0_true_beta.pdf(x))

TV = quad(a0_sampling_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ Beta from KDE of Beta = ', TV)

print('------- Computing TV metrics for $a_1$ --------')
print('-- Using adaptive quadrature (may take a few minutes) --')

def a1_init_soln(x):
    return np.abs(a1_update_dens.pdf(x) - 1/0.1)

TV = quad(a1_init_soln,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from update = ', TV)

def a1_init_error(x):
    return np.abs(a1_true_beta.pdf(x) - 1/0.1)

TV = quad(a1_init_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from Beta = ', TV)

def a1_init_error_kde(x):
    return np.abs(a1_true_kde.pdf(x) - 1/0.1)

TV = quad(a1_init_error_kde,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from KDE of Beta = ', TV)

def a1_dens_error(x):
    return np.abs(a1_true_beta.pdf(x) - a1_update_dens.pdf(x))

TV = quad(a1_dens_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ update from Beta = ', TV)

def a1_dens_estimate_error(x):
    return np.abs(a1_true_kde.pdf(x) - a1_update_dens.pdf(x))

TV = quad(a1_dens_estimate_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ update from KDE of Beta = ', TV)

def a1_sampling_error(x):
    return np.abs(a1_true_kde.pdf(x) - a1_true_beta.pdf(x))

TV = quad(a1_sampling_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ Beta from KDE of Beta = ', TV)

# Now redo using only QoI #1
print('------------ Using first QoI --------------')
QoI_num = [0]
qs_init_xform = DA.QoI_xform(X_init_std,PC_vecs,PC_vals,QoI_num)
X_actual_std = DA.standard_scaling_xform(eigs_actual,scaler)
qs_actual_xform = DA.QoI_xform(X_actual_std,PC_vecs,PC_vals,QoI_num)

# Compute the predicted and observed densities on the QoI
pi_Q_kde = DA.kernel_density_estimate(qs_init_xform)
pi_obs_kde = DA.kernel_density_estimate(qs_actual_xform)

# Construct the ratio of observed to predicted densities and print diagnostic
r = DA.compute_r(pi_obs_kde,pi_Q_kde,qs_init_xform)

# Visualize the ratio over parameter space
plt.figure(figsize=(10,10))
a0_init = init_param_set['a0'][0]
a1_init = init_param_set['a1'][0]
scatter = plt.scatter(a0_init, a1_init, c=r, cmap='hot')
plt.colorbar(scatter,fraction=0.05, pad=0.04)
plt.xlabel('$a_0$')
plt.ylabel('$a_1$')
plt.xlim([0.8,1.2])
plt.ylim([0.1,0.2])
plt.title('$\mathcal{R}(a(r))$')
plt.tight_layout()
plt.savefig('Example-1-ratios-QoI-first.png')

# Compute and visualize the various marginal densities
a0_update_dens = DA.kernel_density_estimate(a0_init, r)
a1_update_dens = DA.kernel_density_estimate(a1_init, r)

plt.figure(figsize=(10,10))
x = np.linspace(0.8,1.2,100)
plt.plot(x,a0_true_beta.pdf(x), linestyle='solid', linewidth=4, label='Exact pdf')
plt.plot(x,a0_update_dens.pdf(x), linestyle='dashed', linewidth=4, label='Estimated pdf')
plt.plot(x,a0_true_kde.pdf(x), linestyle='dashdot', linewidth=4, label='Exact KDE')
plt.plot(x,1/0.4*np.ones(len(x)), linestyle='dotted', linewidth=4, label='Initial')
plt.legend()
plt.title('Estimated and Exact Variation in $a_0$')
plt.tight_layout()
plt.savefig('Example-1-first-QoI-a0-pdfs.png')

plt.figure(figsize=(10,10))
x = np.linspace(0.1,0.2,100)
plt.plot(x,a1_true_beta.pdf(x), linestyle='solid', linewidth=4, label='Exact pdf')
plt.plot(x,a1_update_dens.pdf(x), linestyle='dashed', linewidth=4, label='Estimated pdf')
plt.plot(x,a1_true_kde.pdf(x), linestyle='dashdot', linewidth=4, label='Exact KDE')
plt.plot(x,1/0.1*np.ones(len(x)), linestyle='dotted', linewidth=4, label='Initial')
plt.legend()
plt.title('Estimated and Exact Variation in $a_1$')
plt.tight_layout()
plt.savefig('Example-1-first-QoI-a1-pdfs.png')

print('------- Computing TV metrics for $a_0$ --------')
print('-- Using adaptive quadrature (may take a few minutes) --')

TV = quad(a0_init_soln,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from update = ', TV)

TV = quad(a0_init_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from Beta = ', TV)

TV = quad(a0_init_error_kde,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from KDE of Beta = ', TV)

TV = quad(a0_dens_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ update from Beta = ', TV)

TV = quad(a0_dens_estimate_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ update from KDE of Beta = ', TV)

TV = quad(a0_sampling_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ Beta from KDE of Beta = ', TV)

print('------- Computing TV metrics for $a_1$ --------')
print('-- Using adaptive quadrature (may take a few minutes) --')

TV = quad(a1_init_soln,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from update = ', TV)

TV = quad(a1_init_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from Beta = ', TV)

TV = quad(a1_init_error_kde,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from KDE of Beta = ', TV)

TV = quad(a1_dens_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ update from Beta = ', TV)

TV = quad(a1_dens_estimate_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ update from KDE of Beta = ', TV)

TV = quad(a1_sampling_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ Beta from KDE of Beta = ', TV)


# Now redo using only QoI #2
print('------------ Using second QoI --------------')
QoI_num = [1]
qs_init_xform = DA.QoI_xform(X_init_std,PC_vecs,PC_vals,QoI_num)
X_actual_std = DA.standard_scaling_xform(eigs_actual,scaler)
qs_actual_xform = DA.QoI_xform(X_actual_std,PC_vecs,PC_vals,QoI_num)

# Compute the predicted and observed densities on the QoI
pi_Q_kde = DA.kernel_density_estimate(qs_init_xform)
pi_obs_kde = DA.kernel_density_estimate(qs_actual_xform)

# Construct the ratio of observed to predicted densities and print diagnostic
r = DA.compute_r(pi_obs_kde,pi_Q_kde,qs_init_xform)

# Visualize the ratio over parameter space
plt.figure(figsize=(10,10))
a0_init = init_param_set['a0'][0]
a1_init = init_param_set['a1'][0]
scatter = plt.scatter(a0_init, a1_init, c=r, cmap='hot')
plt.colorbar(scatter,fraction=0.05, pad=0.04)
plt.xlabel('$a_0$')
plt.ylabel('$a_1$')
plt.xlim([0.8,1.2])
plt.ylim([0.1,0.2])
plt.title('$\mathcal{R}(a(r))$')
plt.tight_layout()
plt.savefig('Example-1-ratios-QoI-second.png')

# Compute and visualize the various marginal densities
a0_update_dens = DA.kernel_density_estimate(a0_init, r)
a1_update_dens = DA.kernel_density_estimate(a1_init, r)

plt.figure(figsize=(10,10))
x = np.linspace(0.8,1.2,100)
plt.plot(x,a0_true_beta.pdf(x), linestyle='solid', linewidth=4, label='Exact pdf')
plt.plot(x,a0_update_dens.pdf(x), linestyle='dashed', linewidth=4, label='Estimated pdf')
plt.plot(x,a0_true_kde.pdf(x), linestyle='dashdot', linewidth=4, label='Exact KDE')
plt.plot(x,1/0.4*np.ones(len(x)), linestyle='dotted', linewidth=4, label='Initial')
plt.legend()
plt.title('Estimated and Exact Variation in $a_0$')
plt.tight_layout()
plt.savefig('Example-1-second-QoI-a0-pdfs.png')

plt.figure(figsize=(10,10))
x = np.linspace(0.1,0.2,100)
plt.plot(x,a1_true_beta.pdf(x), linestyle='solid', linewidth=4, label='Exact pdf')
plt.plot(x,a1_update_dens.pdf(x), linestyle='dashed', linewidth=4, label='Estimated pdf')
plt.plot(x,a1_true_kde.pdf(x), linestyle='dashdot', linewidth=4, label='Exact KDE')
plt.plot(x,1/0.1*np.ones(len(x)), linestyle='dotted', linewidth=4, label='Initial')
plt.legend()
plt.title('Estimated and Exact Variation in $a_1$')
plt.tight_layout()
plt.savefig('Example-1-second-QoI-a1-pdfs.png')

print('------- Computing TV metrics for $a_0$ --------')
print('-- Using adaptive quadrature (may take a few minutes) --')

TV = quad(a0_init_soln,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from update = ', TV)

TV = quad(a0_init_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from Beta = ', TV)

TV = quad(a0_init_error_kde,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ init from KDE of Beta = ', TV)

TV = quad(a0_dens_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ update from Beta = ', TV)

TV = quad(a0_dens_estimate_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ update from KDE of Beta = ', TV)

TV = quad(a0_sampling_error,0.8,1.2,maxiter=maxiter)[0]
print('d_{TV} of $a_0$ Beta from KDE of Beta = ', TV)

print('------- Computing TV metrics for $a_1$ --------')
print('-- Using adaptive quadrature (may take a few minutes) --')

TV = quad(a1_init_soln,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from update = ', TV)

TV = quad(a1_init_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from Beta = ', TV)

TV = quad(a1_init_error_kde,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ init from KDE of Beta = ', TV)

TV = quad(a1_dens_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ update from Beta = ', TV)

TV = quad(a1_dens_estimate_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ update from KDE of Beta = ', TV)

TV = quad(a1_sampling_error,0.1,0.2,maxiter=maxiter)[0]
print('d_{TV} of $a_1$ Beta from KDE of Beta = ', TV)
