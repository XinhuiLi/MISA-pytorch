import numpy as np

def run_mgpca(X, comps, rec_type='WT'):
    """
    Run multimodal group PCA on data X.
    :param X: list of numpy arrays, each of shape (n_features, n_samples)
    :param comps: int, number of components to extract
    :param rec_type: str, reconstruction type ('WT' or 'PINV')
    :return: S: list of numpy arrays, each of shape (n_samples, n_components)
    :return: whtM: list of numpy arrays, each of shape (n_components, n_features)
    :return: H: numpy array, shape (n_components, n_samples)
    """
    M = range(len(X))
    N = X[0].shape[1]
    
    V = [x.shape[0] for x in X]
    
    if N <= min(V):
        # Xm.T * Xm, weighted avg per modality --> cov(X)
        cvx = np.zeros((N, N))
        
        for mm in M:
            cvx_ = np.cov(X[mm], rowvar=False)
            cvx += cvx_ / (len(M) * np.trace(cvx_) / N)
        
        # Subject-level PCA reduction...
        lambda_, H = np.linalg.eig(cvx)
        lambda_ = lambda_[:comps]
        H = H[:, :comps]
    
    else:
        # Scale data and concatenate
        X_cat = np.concatenate([np.sqrt(N / (len(M) * np.sum(x**2))) * x for x in X], axis=0)
        cvx = X_cat @ X_cat.T
        
        lambda_, U = np.linalg.eig(cvx)
        lambda_ = lambda_[:comps]
        U = U[:, :comps]
        H = ((np.diag(1.0 / np.sqrt(lambda_)) @ U.T) @ X_cat).T
    
    init_A_list = [np.sqrt(N / (len(M) * np.sum(x**2))) * (x @ H) for x in X]
    norm_A_list = [np.sum(a**2,axis=0) for a in init_A_list]
    
    if len(M) == 1:
        norm_A = np.sqrt(norm_A_list)
    else:
        norm_A = np.sqrt(np.sum(norm_A_list,axis=0))
    
    A = [a / norm_A for a in init_A_list]
    
    if rec_type.upper() == 'WT':
        whtM = [a.T for a in A]
    elif rec_type.upper() == 'PINV':
        whtM = [np.linalg.pinv(a) for a in A]
    
    whtM = [np.sqrt(N - 1) * np.sqrt(N / (len(M) * np.sum(x**2))) * (1.0 / norm_A).reshape(-1, 1) * w for x, w in zip(X, whtM)]
    H = np.sqrt(N - 1) * H.T
    S = [(w @ x).T for x, w in zip(X, whtM)]
    
    return S, whtM, H