import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import norm, beta 
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
'''
This module contains all the necessary functions along with some useful
plotting tools to re-generate the results shown in the two examples in the
paper.
'''

def compute_r(obs_dens,predict_dens,q_vals):
    '''
    This computes the ratio of observed to predicted density values at the predicted data
    '''
    # Compute the ratio
    r = np.divide(obs_dens(q_vals.T),predict_dens(q_vals.T))
    
    # Compute the diagnostic defined by the sample mean of the ratio
    diagnostic = r.mean()

    # Report the diagnostic results
    if np.abs(diagnostic - 1) > 0.1:
        print('Predictability assumption may be violated, mean(r) = %4.2f' %(diagnostic))
    else:
        print('mean(r) = %4.2f, everything looking good.' %(diagnostic))
        
    return(r)

def rejection_sampling(r):
    '''
    Perform accept/reject sampling on a set of proposal samples using
    the weights r associated with the set of samples and return
    the indices idx of the proposal sample set that are accepted.
    This is not used in the analysis of the paper, but is a useful function.
    '''
    N = r.size # size of proposal sample set
    check = np.random.uniform(low=0,high=1,size=N) # create random uniform weights to check r against
    M = np.max(r)
    new_r = r/M # normalize weights 
    idx_accept = np.where(new_r>=check)[0] # acceptance criterion
    idx_reject = np.where(new_r<check)[0] # rejection criterion
    return(idx_accept,idx_reject)


def load_data_set(filename):
    data_set = sio.loadmat(filename)
    return(data_set)


def un_mix(ms_mixed,ns_mixed,qs_mixed,ms_unmixed_array=np.array([]),ns_unmixed_array=np.array([])):
    '''
    The *_mixed are inputs of ms, ns, and qs for which eigenmodes (defined by ms and ns) are potentially
    mixed.
    
    The optional *_unmixed_arrays come from a previous "unmixing" of data (e.g., from a prior set that is done
    before the observed set). If these are passed to the function, then we make sure to "unmix" the eigenmodes
    in exactly the same order in the new data set. 
    '''
    num_samples = qs_mixed[:,0].size
    
    if ms_unmixed_array.size: 
        # If a non-empty array of previously un-mixed data is provided, then the mixed data sent to this
        # function is un-mixed in the same order
        
        # initialize un-mixed arrays
        num_modes = len(ms_unmixed_array[0,:])
        num_unmixed_modes = qs_mixed[0,:].size
        qs_unmixed = np.zeros((num_samples, num_modes))
        ms_unmixed = np.zeros((num_samples, num_modes))
        ns_unmixed = np.zeros((num_samples, num_modes))
        unmixed_idx = np.zeros((num_samples, num_modes))
        
        # Find and remove where mixing occurred within the set
        for i in range(num_modes):
            m_to_find = ms_unmixed_array[0,i]
            n_to_find = ns_unmixed_array[0,i]
            for j in range(num_unmixed_modes):
                idx_temp = np.where( (ms_mixed[:,j] == m_to_find) & (ns_mixed[:,j] == n_to_find) )
                if idx_temp[0].size is None:
                    print('----------------------------WARNING-----------------------------------')
                    print('The eigenmode given by (m,n)=(%i,%i) is not present in this mixed data' 
                          %(m_to_find,n_to_find))
                    print('-------------------------ACTION NEEDED----------------------------------')
                    print('Remove eigenmode (%i,%i) from other unmixed data set in order to compare' 
                          %(m_to_find,n_to_find))
                else:
                    unmixed_idx[idx_temp,i] = idx_temp
                    qs_unmixed[idx_temp,i] = qs_mixed[idx_temp,j]
                    ms_unmixed[idx_temp,i] = ms_mixed[idx_temp,j]
                    ns_unmixed[idx_temp,i] = ns_mixed[idx_temp,j]

        # Remove any mixing that occurs outside of the set
        bad_modes = []
        for i in range(num_modes):
            ms_set = set(ms_unmixed[:,i])
            ns_set = set(ns_unmixed[:,i])
            if len(ns_set)>1:
                if len(ms_set) != len(ns_set):
                    raise ValueError('The ms and ns do not match in length')
                for (m,n) in zip(ms_set,ns_set):
                    print('Mixing out of set occurs at eigenmode %i with (%i,%i)' 
                          %(int(num_modes-i),m,n))
                    bad_modes.append(i)

        if bad_modes:
            print('-----------------------------')
            print('Removing bad mixing from data')
            print('-----------------------------')
            qs_unmixed = np.delete(qs_unmixed,bad_modes, axis=1)
            ms_unmixed = np.delete(ms_unmixed,bad_modes, axis=1)
            ns_unmixed = np.delete(ns_unmixed,bad_modes, axis=1)

    else:
        # initialize un-mixed arrays
        num_modes = qs_mixed[0,:].size
        qs_unmixed = np.zeros((num_samples, num_modes))
        ms_unmixed = np.zeros((num_samples, num_modes))
        ns_unmixed = np.zeros((num_samples, num_modes))
        unmixed_idx = np.zeros((num_samples, num_modes))
        
        # Find and remove where mixing occurred within the set
        for i in range(num_modes):
            m_to_find = ms_mixed[0,i]
            n_to_find = ns_mixed[0,i]
            for j in range(num_modes):
                idx_temp = np.where( (ms_mixed[:,j] == m_to_find) 
                                    & (ns_mixed[:,j] == n_to_find) )
                unmixed_idx[idx_temp,i] = idx_temp
                qs_unmixed[idx_temp,i] = qs_mixed[idx_temp,j]
                ms_unmixed[idx_temp,i] = ms_mixed[idx_temp,j]
                ns_unmixed[idx_temp,i] = ns_mixed[idx_temp,j]

        # Remove any mixing that occurs outside of the set
        bad_modes = []
        for i in range(num_modes):
            ms_set = set(ms_unmixed[:,i])
            ns_set = set(ns_unmixed[:,i])
            if len(ns_set)>1:
                if len(ms_set) != len(ns_set):
                    raise ValueError('The ms and ns do not match in length')
                for (m,n) in zip(ms_set,ns_set):
                    print('Mixing out of set occurs at eigenmode %i with (%i,%i)' 
                          %(int(num_modes-i),m,n))
                    bad_modes.append(i)

        if bad_modes:
            print('-----------------------------')
            print('Removing bad mixing from data')
            print('-----------------------------')
            qs_unmixed = np.delete(qs_unmixed,bad_modes, axis=1)
            ms_unmixed = np.delete(ms_unmixed,bad_modes, axis=1)
            ns_unmixed = np.delete(ns_unmixed,bad_modes, axis=1)
    
    return(ms_unmixed,ns_unmixed,qs_unmixed)        



def print_modes(ms,ns):
    '''
    This is useful for checking that the unmixed data from both sets (initial and observed) are
    in the same order (i.e., that the eigen-modes appear in the same order).
    
    This is primarily useful for debugging but was not used in the scripts after debugging occurred. 
    '''
    num_modes = ms[0,:].size
    for i in range(num_modes):
        ms_set = set(ms[:,i])
        ns_set = set(ns[:,i])
        print('Eigenmode %i is given by (m,n) pair(s) (plural if there is mixing): ' 
              %(num_modes-i) ,(ms_set,ns_set))
        print('--------------------------------------------------------------------'
              + '---------')
    
    
    
def standard_scaling_xform(qs, scaler = None):
    '''
    Transforms given data (qs) using a standard scalar from sklearn, which helps
    remove the impact of outliers on analysis. 
    '''
    if scaler is None: # Scaling from the predicted data is not yet learned
        
        scaler = StandardScaler()

        X_std = scaler.fit_transform(qs)
        
        return(X_std, scaler)
    
    else: # scaler is previously learned, this is used when applying the scalar to observations
        
        X_std = scaler.transform(qs)
        
        return(X_std)


    
def PCA(X_std):
    '''
    A standard principal component analysis (PCA)
    '''
    cov_mat = np.cov(X_std.T)
    
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    
    return(eig_vals, eig_vecs)

def plot_gap(n, eig_vals, filename=None):
    '''
    This plots the percent variation explained by n of the principal components.
    
    Here, eig_vals refers to eigenvalues from principal component analysis
    '''
    fig = plt.figure(figsize=(10,10))
    fig.clear()
    plt.semilogy(np.arange(np.size(eig_vals)),eig_vals/np.sum(eig_vals)*100, Marker='.', MarkerSize=25, linestyle='')
    plt.semilogy(np.arange(np.size(eig_vals)),eig_vals[n]/np.sum(eig_vals)*100*np.ones(np.size(eig_vals)), 'r--')
    plt.semilogy(np.arange(np.size(eig_vals)),eig_vals[n+1]/np.sum(eig_vals)*100*np.ones(np.size(eig_vals)), 'r--')
    s = 'Order of magnitude of gap is %4.2f.' %(np.log10(eig_vals[n])-np.log10(eig_vals[n+1]))
    s += '\n%2.3f' %(np.sum(eig_vals[0:n+1])/np.sum(eig_vals)*100) + '% of variation explained.'
    plt.title(s)
    plt.xlabel('Principal Component #')
    plt.ylabel('% of Variation')
    plt.tight_layout()
    if filename == None:
        print('Showing but not saving image')
        plt.show()
    else:
        print('Image saved.')
        plt.savefig(filename + '.png')
    return

def plot_PC_vecs(n, eig_vals, eig_vecs, filename=None):
    '''
    This plots the n:th principal vector.
    
    Here, eig_vals and eig_vecs refer to the eigenvalues and eigenvectors from a PCA
    '''
    fig = plt.figure(figsize=(10,10))
    fig.clear()
    plt.plot(np.arange(np.size(eig_vals)),eig_vecs[:,n], Marker='.', MarkerSize=25, linestyle='', color='k')
    plt.title('Weighting vector ' + str(n+1))
    plt.xlabel('Eigenmode #')
    plt.ylabel('Weights')
    plt.ylim([-1,1])
    plt.tight_layout() 
    if filename == None:
        print('Showing but not saving image')
        plt.show()
    else:
        print('Image saved.')
        plt.savefig(filename + '.png')
    return

def plot_eigenvalue_pairs(eigs_predict, eigs_obs, eig1, eig2, filename=None):
    '''
    This is used to visualize the correlation between various eigenvalue data.
    
    Here, eigenvalues are associated with the drum problem.
    '''
    fig = plt.figure(figsize=(10,10))
    fig.clear()
    plt.scatter(eigs_predict[:,eig1], eigs_predict[:,eig2],5,marker='.')
    plt.scatter(eigs_obs[:,eig1], eigs_obs[:,eig2],25,marker='s')
    plt.xlabel('Eigenmode #%d' %(20-eig1))
    plt.ylabel('Eigenmode #%d' %(20-eig2))
    plt.tight_layout()
    if filename == None:
        print('Showing but not saving image')
        plt.show()
    else:
        print('Image saved.')
        plt.savefig(filename + '.png')
    return

def plot_QoI_pairs(q_predict, q_obs, q1, q2, filename=None):
    '''
    This visualizes the correlation between the various PCs used to define the relavant QoI.
    '''
    fig = plt.figure(figsize=(10,10))
    fig.clear()
    plt.scatter(q_predict[:,q1], q_predict[:,q2],marker='.')
    plt.scatter(q_obs[:,q1], q_obs[:,q2],25,marker='s')
    plt.xlabel('QoI #%d' %(q1+1))
    plt.ylabel('QoI #%d' %(q2+1))
    if filename == None:
        print('Showing but not saving image')
        plt.show()
    else:
        print('Image saved.')
        plt.savefig(filename + '.png')
    return    

def QoI_xform(X_std,eig_vecs,eig_vals,Select_QoI=2):
    '''
    Use the PCA to transform the scaled eigen-data associated with the drum problem to define the QoI.
    '''
    if isinstance(Select_QoI, int):
        QoI_num = Select_QoI
        qs_xform = np.dot(X_std,eig_vecs[:,0:QoI_num])/eig_vals[0:QoI_num]
    elif isinstance(Select_QoI, list):
        QoI_list = Select_QoI
        qs_xform = np.dot(X_std,eig_vecs[:,QoI_list])/eig_vals[QoI_list]
    else:
        print('Improper Select_QoI specified, give an int or a list')
        return
        
    return(qs_xform)



def kernel_density_estimate(qs, rs=None):
    '''
    Use a weighted Gaussian kernel density estimator from scipy to estimate density
    on an array of data (qs).
    '''
    dens_est = GKDE(qs.T, 'silverman', weights=rs)
    
    return(dens_est)


def beta_density_estimate(data_array):
    '''
    Can construct beta density estimates on a given array of data for the interested reader
    who would like to see how good results can be using this in place of the non-parametric
    KDE from above.
    '''
    dim = data_array[0,:].size
    
    beta_params = np.zeros((dim,4)) #col0=a,col1=b,col2=loc,col3=scale
    
    beta_dens = []
    
    for i in range(dim):
        
        beta_params[i,:] = beta.fit(data_array[:,i])
        
        beta_dens.append(beta(a=beta_params[i,0], b=beta_params[i,1],
                              loc=beta_params[i,2],scale=beta_params[i,3]))
    
    return(beta_dens,beta_params)

                         
def mean_beta_distr(r, param_samples, N=25):
    '''
    Perform rejection sampling N times to fit beta distributions on parameters.
    The parameter samples may be any set of a(s) values evaluated at some spatial positions
    \{s_i\} (where s_i is the radial distance from center s=0 to edge s=1 of drum). 
    
    Inputs:
    r = obs_dens/predict_dens evaluated at predicted points
    param_samples = array of parameter samples of size [sample_num,param_dim]
    N = number of times we re-do computations to determine mean hyperparameters for Beta distr fits. 
    
    Outputs:
    (beta_dens,beta_params_means)
    '''

    param_dim = param_samples[0,:].size
    beta_params = np.zeros((N,param_dim,4))
    beta_params_means = np.zeros((param_dim,4))
    beta_dens = []
    
    for i in range(N):
        
        (samples_to_keep,_) = rejection_sampling(r)

        accepted_samples = param_samples[samples_to_keep,:]

        for j in range(param_dim):
            
            beta_params[i,j,:] = beta.fit(accepted_samples[:,j])
    
    beta_params_means = np.mean(beta_params, axis=0)
    
    for i in range(param_dim):
        
        beta_dens.append(beta(a=beta_params_means[i,0], b=beta_params_means[i,1],
                              loc=beta_params_means[i,2],scale=beta_params_means[i,3]))
    
    return(beta_dens, beta_params_means)
    