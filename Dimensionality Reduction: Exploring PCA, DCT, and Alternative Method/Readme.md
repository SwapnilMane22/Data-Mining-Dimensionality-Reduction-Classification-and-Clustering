# Dimensionality Reduction: Exploring PCA, DCT, and Alternative Method

## Overview
This project implements and evaluates dimensionality reduction techniques (PCA, DCT, and an additional method) on three different datasets. The goal is to analyze the effectiveness of each method in reducing dimensionality while preserving essential data characteristics.

### Datasets
The project uses three datasets:
1. **Table 1**: A dense dataset describing a specific image.
2. **Table 2**: A randomly generated dense noise dataset.
3. **Table 3**: A sparse dataset.

### Objectives
1. **PCA and DCT Implementation**  
   - Implement Principal Component Analysis (PCA) and Discrete Cosine Transform (DCT).
   - Apply them to the three datasets and report the reduced dimensionalities.
2. **Additional Dimensionality Reduction Method**  
   - Identify and implement (or reuse) an additional dimensionality reduction method.
   - Apply this method to the datasets and report the reduced dimensionalities.
3. **Comparison and Discussion**  
   - Compare the results across the three methods.
   - Discuss consistencies or differences in the reduced dimensionalities for each dataset.

---

## Features
### 1. PCA (Principal Component Analysis)
- Standardizes the dataset.
- Computes eigenvalues and eigenvectors of the covariance matrix.
- Reduces dimensionality based on the number of components retaining the most variance.

### 2. DCT (Discrete Cosine Transform)
- Applies DCT to transform the data into the frequency domain.
- Applies thresholding to retain significant coefficients.
- Inverse transforms the reduced data back to the original domain for comparison.

### 3. Additional Method: Singular Value Decomposition (SVD)
- Computes singular values and vectors to represent the data in lower dimensions.
- Retains a specified number of components with the highest singular values.

---

## Code Details

### **Functions**
1. **`load_data(file_path)`**  
   Loads the dataset from a given file path.
2. **`pca_transform(data, n_components)`**  
   Performs PCA and returns reduced and reconstructed data.
3. **`dct_transform(data)` and `idct_transform(data)`**  
   Applies DCT and its inverse to the data.
4. **`svd_transform(data, n_components)`**  
   Reduces dimensionality using SVD.
5. **Visualization Functions**  
   - Eigenvalues, DCT coefficients, PCA-reconstructed images, and SVD-reduced data are plotted for analysis.
6. **`apply_threshold(dct_transformed, threshold)`**  
   Filters DCT coefficients below a specified threshold.

### **Main Logic**
The `main` function:
- Loads the datasets from files.
- Applies PCA, DCT, and SVD to each dataset with specified parameters.
- Prints the reduced dimensionalities and visualizes results.

---

## Usage

### Prerequisites
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `sklearn`, `scipy`

### Running the Code
1. Place the data files (`Table1.txt`, `Table2.txt`, `Table3.txt`) in the working directory.
2. Execute the script:
   ```bash
   python data_compression.py
   ```
3. Observe the console output and plots for dimensionality reduction results.

---

## Results and Evaluation

### Expected Outputs
1. **Reduced Dimensionalities**
   - The resulting number of dimensions for each dataset and method.
2. **Visualizations**
   - Plots of eigenvalues, reconstructed data, and DCT coefficients.

### Key Points for Discussion
- Are the dimensionality reductions consistent across datasets and methods?
- How does the sparsity or noise in datasets affect the performance of PCA, DCT, and SVD?
- Which method performs best for each type of dataset, and why?

---

## File Structure
- **`data_compression.py`**: Main implementation file.
- **`Table1.txt`**, **`Table2.txt`**, **`Table3.txt`**: Input datasets.
