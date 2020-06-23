# Creates all figures and table data in Section 5.1

# Import modules and libraries
import DrumAnalysis as DA
import numpy as np
from scipy.stats import beta 
import scipy.integrate.quadrature as quad
import matplotlib.pyplot as plt


# Set plotting parameters
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'axes.linewidth': 2})

# Load observed data sets
actual_param_set = DA.load_data_set('poly_2_observed_input_set.mat')
actual_obs_set = DA.load_data_set('poly_2_observed_data_set_20.mat')

cf_obs = actual_param_set['cf'] # Used for plotting

# Do an un-mixing of observed data before learning # of QoI in data 
(ms_actual,ns_actual,eigs_actual) = DA.un_mix(actual_obs_set['ms'],
                                            actual_obs_set['ns'],
                                            actual_obs_set['eig'])

# Learn how many QoI (and thus # of parameters) are present in observed data
X_actual_std_how_many_QoI, _ = DA.standard_scaling_xform(eigs_actual[:,:])
PC_vals_actual, _ = DA.PCA(X_actual_std_how_many_QoI)
DA.plot_gap(2, PC_vals_actual, filename='Example-2-learn-QoI-number-from-observed')

# Load an initial design set
init_param_set = DA.load_data_set('cpl_3_prior_input_set.mat')
init_obs_set = DA.load_data_set('cpl_3_prior_data_set_20.mat')

yp_init = init_param_set['yp'] # the a(r) values of the initial set
x_pl = init_param_set['xp'][0,:] # the r-values from center to edge of drum

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

# Learn the QoI through feature extraction (using PCA) after
# first performing a standard scaling of data
X_init_std, scaler = DA.standard_scaling_xform(eigs_init)
PC_vals, PC_vecs = DA.PCA(X_init_std)

# Visualize spectral gaps
DA.plot_gap(2, PC_vals, filename='Example-2-QoI-Gap-design-set-1')

# Transform observed eigenvalue data into learned QoI
QoI_num = 3
qs_init_xform = DA.QoI_xform(X_init_std,PC_vecs,PC_vals,QoI_num)
X_actual_std = DA.standard_scaling_xform(eigs_actual,scaler)
qs_actual_xform = DA.QoI_xform(X_actual_std,PC_vecs,PC_vals,QoI_num)

# Visualize the improved correlation between QoI (both predicted and observed)
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 0, 1, filename='Example-2-QoI-pair1-design-set-1')
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 0, 2, filename='Example-2-QoI-pair2-design-set-1')
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 1, 2, filename='Example-2-QoI-pair3-design-set-1')

# Compute the predicted and observed densities on the QoI
pi_Q_kde = DA.kernel_density_estimate(qs_init_xform)
pi_obs_kde = DA.kernel_density_estimate(qs_actual_xform)

# Construct the ratio of observed to predicted densities and print diagnostic
r = DA.compute_r(pi_obs_kde,pi_Q_kde,qs_init_xform)

############### Create functions for computing distances and generating plots #############

# Create upper and lower bounds on parameter region plots
yu_pt1 = np.max(cf_obs[:,0])
yu_pt2 = np.max(cf_obs[:,0]+cf_obs[:,1]*0.5 + cf_obs[:,2]*0.5**2)
yu_pt3 = np.max(cf_obs[:,0]+cf_obs[:,1] + cf_obs[:,2])
yu = np.poly1d(np.polyfit(np.array([0,0.5,1]), np.array([yu_pt1,yu_pt2,yu_pt3]), deg=2))
yl_pt1 = np.min(cf_obs[:,0])
yl_pt2 = np.min(cf_obs[:,0]+cf_obs[:,1]*0.5 + cf_obs[:,2]*0.5**2)
yl_pt3 = np.min(cf_obs[:,0]+cf_obs[:,1] + cf_obs[:,2])
yl = np.poly1d(np.polyfit(np.array([0,0.5,1]), np.array([yl_pt1,yl_pt2,yl_pt3]), deg=2))

def distance(ys,yu,yl):
    
    dist = 0
    
    for i in range(x_pl.size-1):
    
        x = np.linspace(x_pl[i],x_pl[i+1],51)
        dx = x[1] - x[0]
        x_mid = x[0:-1] + dx/2
    
        ys_mid = ys[i]*(x_pl[i+1]-x_mid)/(x_pl[i+1]-x_pl[i]) \
                 + ys[i+1]*(x_mid-x_pl[i])/(x_pl[i+1]-x_pl[i])
    
        yu_mid = yu(x_mid)
    
        yl_mid = yl(x_mid)

        idx_above = np.where(ys_mid >= yu_mid)
        dist += np.sum(( ys_mid[idx_above] - yu_mid[idx_above] ) * dx)
    
        idx_below = np.where(ys_mid <= yl_mid)
        dist += np.sum( -( ys_mid[idx_below] - yl_mid[idx_below] ) * dx)
    
    return dist

def plot_r_greater(percent, filename=None):
    plt.figure()
    M = np.max(r)
    idx = np.where(r/M>=percent)[0]
    plt.title('Samples with r $\geq$ ' + str(percent))
    plt.plot(x_pl, yp_init[idx,:].transpose(),'r--')
    x = np.linspace(0,1,100)
    plt.plot(x,yu(x),'b')
    plt.plot(x,yl(x),'b')
    plt.ylim([1.1,2.5])
    if filename == None:
        print('Showing but not saving image')
        plt.show()
    else:
        print('Image saved.')
        plt.savefig(filename + '.png')
    return   
    
def plot_r_greater_distances(percent, filename=None):
    
    M = np.max(r)
    idx = np.where(r/M>=percent)[0]
    
    percent_distances = np.zeros(np.size(idx))

    for i in range(np.size(idx)):
        percent_distances[i] = distance(yp_init[idx[i],:],yu,yl)
        
    fig, ax = plt.subplots()
    plt.hist(percent_distances, bins=15, range=[0,0.03])
    plt.title('Distribution of distances')
    textstr = '\n'.join((
        r'Mean$=%2.3e$' % (np.mean(percent_distances), ),
        r'Median$=%2.3e$' % (np.median(percent_distances), ),
        r'St.dev.$=%2.3e$' % (np.std(percent_distances), )))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.45, 0.65, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    if filename == None:
        print('Showing but not saving image')
        plt.show()
    else:
        print('Image saved.')
        plt.savefig(filename + '.png')
    return   

# Visualize parameter space info
plot_r_greater(0.8, filename='Example-2-r-greater-0p8-design-set-1')
plot_r_greater(0.01, filename='Example-2-r-greater-0p01-design-set-1')
plot_r_greater_distances(0.01, filename='Example-2-distances-r-greater-0p01-design-set-1')
    

######## Now re-do with different design set
# Load an initial design set
init_param_set = DA.load_data_set('cpl_3_right_prior_input_set.mat')
init_obs_set = DA.load_data_set('cpl_3_right_prior_data_set_20.mat')

yp_init = init_param_set['yp'] # the a(r) values of the initial set
x_pl = init_param_set['xp'][0,:] # the r-values from center to edge of drum

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

# Learn the QoI through feature extraction (using PCA) after
# first performing a standard scaling of data
X_init_std, scaler = DA.standard_scaling_xform(eigs_init)
PC_vals, PC_vecs = DA.PCA(X_init_std)

# Visualize spectral gaps
DA.plot_gap(2, PC_vals, filename='Example-2-QoI-Gap-design-set-2')

# Transform observed eigenvalue data into learned QoI
QoI_num = 3
qs_init_xform = DA.QoI_xform(X_init_std,PC_vecs,PC_vals,QoI_num)
X_actual_std = DA.standard_scaling_xform(eigs_actual,scaler)
qs_actual_xform = DA.QoI_xform(X_actual_std,PC_vecs,PC_vals,QoI_num)

# Visualize the improved correlation between QoI (both predicted and observed)
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 0, 1, filename='Example-2-QoI-pair1-design-set-2')
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 0, 2, filename='Example-2-QoI-pair2-design-set-2')
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 1, 2, filename='Example-2-QoI-pair3-design-set-2')

# Compute the predicted and observed densities on the QoI
pi_Q_kde = DA.kernel_density_estimate(qs_init_xform)
pi_obs_kde = DA.kernel_density_estimate(qs_actual_xform)

# Construct the ratio of observed to predicted densities and print diagnostic
r = DA.compute_r(pi_obs_kde,pi_Q_kde,qs_init_xform)

# Visualize parameter space info
plot_r_greater(0.8, filename='Example-2-r-greater-0p8-design-set-2')
plot_r_greater(0.01, filename='Example-2-r-greater-0p01-design-set-2')
plot_r_greater_distances(0.01, filename='Example-2-distances-r-greater-0p01-design-set-2')



######## Now re-do with different design set not used in paper
# Load an initial design set
init_param_set = DA.load_data_set('cpl_3_left_prior_input_set.mat')
init_obs_set = DA.load_data_set('cpl_3_left_prior_data_set_20.mat')

yp_init = init_param_set['yp'] # the a(r) values of the initial set
x_pl = init_param_set['xp'][0,:] # the r-values from center to edge of drum

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

# Learn the QoI through feature extraction (using PCA) after
# first performing a standard scaling of data
X_init_std, scaler = DA.standard_scaling_xform(eigs_init)
PC_vals, PC_vecs = DA.PCA(X_init_std)

# Visualize spectral gaps
DA.plot_gap(2, PC_vals, filename='Example-2-QoI-Gap-design-set-3-not-in-paper')

# Transform observed eigenvalue data into learned QoI
QoI_num = 3
qs_init_xform = DA.QoI_xform(X_init_std,PC_vecs,PC_vals,QoI_num)
X_actual_std = DA.standard_scaling_xform(eigs_actual,scaler)
qs_actual_xform = DA.QoI_xform(X_actual_std,PC_vecs,PC_vals,QoI_num)

# Visualize the improved correlation between QoI (both predicted and observed)
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 0, 1, filename='Example-2-QoI-pair1-design-set-3-not-in-paper')
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 0, 2, filename='Example-2-QoI-pair2-design-set-3-not-in-paper')
DA.plot_QoI_pairs(qs_init_xform, qs_actual_xform, 1, 2, filename='Example-2-QoI-pair3-design-set-3-not-in-paper')

# Compute the predicted and observed densities on the QoI
pi_Q_kde = DA.kernel_density_estimate(qs_init_xform)
pi_obs_kde = DA.kernel_density_estimate(qs_actual_xform)

# Construct the ratio of observed to predicted densities and print diagnostic
r = DA.compute_r(pi_obs_kde,pi_Q_kde,qs_init_xform)

# Visualize parameter space info
plot_r_greater(0.8, filename='Example-2-r-greater-0p8-design-set-3-not-in-paper')
plot_r_greater(0.01, filename='Example-2-r-greater-0p01-design-set-3-not-in-paper')
plot_r_greater_distances(0.01, filename='Example-2-distances-r-greater-0p01-design-set-3-not-in-paper')
