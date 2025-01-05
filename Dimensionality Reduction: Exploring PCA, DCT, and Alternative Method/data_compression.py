import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import dct, idct

def load_data(file_path):
    """Load data from a text file."""
    return np.loadtxt(file_path)

def plot_eigenvalues(eigenvalues):
    """Plot the eigenvalues."""
    plt.figure()
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues)
    plt.title('Eigenvalues')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.get_current_fig_manager().window.state('zoomed')  # Display in full screen

def plot_pca_images(original_data, compress_data, transformed_data):
    """Plot original data, image after compressing, and transformed data."""
    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(original_data)
    plt.title('Original Data')

    plt.subplot(1, 3, 2)
    plt.imshow(compress_data)
    plt.title('Compressed Image')

    plt.subplot(1, 3, 3)
    plt.imshow(transformed_data)
    plt.title('Transformed Data')
    plt.tight_layout()
    plt.get_current_fig_manager().window.state('zoomed')  # Display in full screen

def plot_dct_images(original_data, compress_data):#, transformed_data):
    """Plot original data, image after compressing, and transformed data."""
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(original_data)
    plt.title('Original Data')

    plt.subplot(1, 2, 2)
    plt.imshow(compress_data)
    plt.title('Compressed Image')
    plt.tight_layout()
    plt.get_current_fig_manager().window.state('zoomed')  # Display in full screen

def plot_dct_coefficients(dct_transformed):
    """Plot the DCT coefficients."""
    plt.figure()
    for i in range(dct_transformed.shape[1]):
        plt.plot(dct_transformed[:, i], label=f'DCT Coefficient {i+1}') 
    plt.title('DCT Coefficients')
    plt.xlabel('Sample Index')
    plt.ylabel('Coefficient Value')
    plt.tight_layout()
    plt.get_current_fig_manager().window.state('zoomed')  # Display in full screen

def apply_threshold(dct_transformed, threshold):
    """Apply thresholding to DCT coefficients."""
    thresholded_dct = dct_transformed.copy()  
    thresholded_dct[np.abs(thresholded_dct) < threshold] = 0  
    return thresholded_dct

def calculate_threshold(dct_transformed, percentage):
    """Calculate the threshold based on the percentage of coefficients to retain."""
    sorted_coeffs = np.sort(np.abs(dct_transformed).ravel())[::-1]  
    threshold_index = int(len(sorted_coeffs) * percentage)  
    threshold = sorted_coeffs[threshold_index]  
    return threshold

def select_top_eigenvectors(eigenvectors, k):
    """Select the top k eigenvectors based on the number of components to keep."""
    return eigenvectors[:, :k]

def pca_transform(data, n_components):
    """Perform Principal Component Analysis."""
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    cov_matrix = np.cov(data_standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Plot eigenvalues
    plot_eigenvalues(eigenvalues)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    top_eigenvectors = select_top_eigenvectors(eigenvectors, n_components)
    X_pca = np.dot(data_standardized, top_eigenvectors)
    inverse_pca_data = np.dot(X_pca, top_eigenvectors.T)
    inverse_pca_data = scaler.inverse_transform(inverse_pca_data)
    return X_pca, inverse_pca_data, top_eigenvectors

def find_k_values(dct_data, energy_percent):
    """Find the value of k based on the specified energy percentage."""
    total_energy = np.sum(np.abs(dct_data) ** 2)  # Total energy in DCT coefficients
    sorted_coeffs = np.sort(np.abs(dct_data).ravel())[::-1]  # Sorted DCT coefficients

    # Calculate cumulative energy and find the minimum k
    cumulative_energy = np.cumsum(sorted_coeffs ** 2)
    k = np.searchsorted(cumulative_energy, energy_percent * total_energy)

    return k

def dct_transform(data):
    """Perform Discrete Cosine Transform."""
    return dct(data, type=2, axis=0)

def idct_transform(dct_transformed):
    """Perform Inverse Discrete Cosine Transform."""
    return np.real(idct(dct_transformed[:, :], type=2, axis=0))

def plot_svd_components(X_svd):
    """Plot pairs of components from the SVD-reduced data."""
    n_components = X_svd.shape[1]
    fig, axs = plt.subplots(n_components, n_components, figsize=(12, 12))

    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                axs[i, j].scatter(X_svd[:, i], X_svd[:, j], marker='.')
                axs[i, j].set_xlabel(f'Component {i+1}')
                axs[i, j].set_ylabel(f'Component {j+1}')
            else:
                axs[i, j].hist(X_svd[:, i], bins=20)
                axs[i, j].set_xlabel(f'Component {i+1}')

    plt.tight_layout()

    plt.get_current_fig_manager().window.state('zoomed')  # Display in full screen

def svd_transform(data, n_components):
    """Perform Singular Value Decomposition (SVD) based dimensionality reduction."""
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    U, S, Vt = np.linalg.svd(data_standardized, full_matrices=False)

    
    # Plot the singular values
    plt.figure()
    plt.plot(S, marker='o', linestyle='-')
    plt.title('Singular Values')
    plt.xlabel('Component Index')
    plt.ylabel('Singular Value')
    plt.get_current_fig_manager().window.state('zoomed')  # Display in full screen
    
    # Select the first n_components from U, S, and Vt
    U_reduced = U[:, :n_components]
    S_reduced = np.diag(S[:n_components])
    Vt_reduced = Vt[:n_components, :]

    print("k-value of SVD: ", S_reduced.shape[0])
    
    # Compute the reduced representation of the data
    X_svd = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))
    
    # Inverse construction algorithm for SVD
    X_reconstructed = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    
    return X_svd, X_reconstructed, U_reduced, S_reduced, Vt_reduced

def input1(data, components_pca, components_dct, components_svd):

    n_samples, n_features = data.shape

    # Perform PCA
    X_pca, inverse_pca_data, top_eigenvectors = pca_transform(data, components_pca)

    print("k value of PCA: ", X_pca.shape)

    # Plot PCA images
    plot_pca_images(data, inverse_pca_data, X_pca)

    # Perform DCT
    dct_transformed = dct_transform(data)

    k_dct = find_k_values(dct_transformed,components_dct)

    # Calculate threshold
    percentage_threshold = components_dct  

    threshold = calculate_threshold(dct_transformed, percentage_threshold)

    # print("k-value of DCT: ", k_dct / 5)

    # Apply thresholding to the DCT coefficients
    thresholded_dct = apply_threshold(dct_transformed, threshold)

    # Apply the inverse Discrete Cosine Transform (iDCT)
    idct_transformed = idct_transform(thresholded_dct)

    # Plot DCT images
    plot_dct_images(data, idct_transformed)

    # Plot DCT coefficients
    plot_dct_coefficients(thresholded_dct)

    # Perform SVD based dimensionality reduction
    n_components_svd = components_svd  # Choose the number of components

    """Perform Singular Value Decomposition (SVD) based dimensionality reduction."""
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    
    # Plot the singular values
    plt.figure()
    plt.plot(S, marker='o', linestyle='-')
    plt.title('Singular Values')
    plt.xlabel('Component Index')
    plt.ylabel('Singular Value')
    plt.get_current_fig_manager().window.state('zoomed')  # Display in full screen
    
    # Select the first n_components from U, S, and Vt
    U_reduced = U[:, :components_svd]
    S_reduced = np.diag(S[:components_svd])
    Vt_reduced = Vt[:components_svd, :]

    print("k-value of SVD: ", S_reduced.shape[0])
    
    # Compute the reduced representation of the data
    X_svd = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))
    
    # Inverse construction algorithm for SVD
    X_reconstructed = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))

    X_svd, X_reconstructed_svd, U_svd, S_svd, Vt_svd = X_svd, X_reconstructed, U_reduced, S_reduced, Vt_reduced

    # Plot original and reconstructed data after SVD
    # plot_reconstructed_data(data, X_reconstructed_svd, n_components_svd)

    plot_pca_images(data, X_reconstructed_svd, X_svd)
   

    plt.show()   

def input(data, components_pca, components_dct, components_svd):

    n_samples, n_features = data.shape

    # Perform PCA
    X_pca, inverse_pca_data, top_eigenvectors = pca_transform(data, components_pca)

    print("k value of PCA: ", X_pca.shape)

    # Plot PCA images
    plot_pca_images(data, inverse_pca_data, X_pca)

    # Perform DCT
    dct_transformed = dct_transform(data)

    k_dct = find_k_values(dct_transformed,components_dct)

    # Calculate threshold
    percentage_threshold = k_dct / dct_transformed.shape[0]  

    threshold = calculate_threshold(dct_transformed, percentage_threshold)

    print("k-value of DCT: ", k_dct / 5)

    # Apply thresholding to the DCT coefficients
    thresholded_dct = apply_threshold(dct_transformed, threshold)

    # Apply the inverse Discrete Cosine Transform (iDCT)
    idct_transformed = idct_transform(thresholded_dct)

    # Plot DCT images
    plot_dct_images(data, idct_transformed)

    # Plot DCT coefficients
    plot_dct_coefficients(thresholded_dct)

    # Perform SVD based dimensionality reduction
    n_components_svd = components_svd  # Choose the number of components

    X_svd, X_reconstructed_svd, U_svd, S_svd, Vt_svd = svd_transform(data, n_components_svd)

    # Plot original and reconstructed data after SVD
    # plot_reconstructed_data(data, X_reconstructed_svd, n_components_svd)

    plot_pca_images(data, X_reconstructed_svd, X_svd)
   

    plt.show()   

def main():
    # Load data
    print('Table 1')
    file_path = 'Table1.txt'
    data1 = load_data(file_path)
    input(data1,13,0.80,12)

    print('\nTable 2')
    file_path1 = 'Table2.txt'
    data2 = load_data(file_path1)
    input(data2,17,0.83,16)

    print('\nTable 3')
    file_path2 = 'Table3.txt'
    data3 = load_data(file_path2)
    input1(data3,35,0.9,40)
    
    # plt.show()


if __name__ == "__main__":
    main()