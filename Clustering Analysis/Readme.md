# Clustering Analysis with Modified K-Means Algorithm

## Overview
This project explores clustering techniques with a focus on the **K-means algorithm** and its applications to **time-series data** and **relational data**. The implementation includes a modified K-means clustering algorithm enhanced with the **elbow method** to determine the optimal number of clusters without prior knowledge of K.

## Key Features
1. **Modified K-Means Algorithm** - Automatically determines the number of clusters (K) using the elbow method.
2. **Evaluation Metrics** - Utilizes **Adjusted Rand Index (ARI)** and **Normalized Mutual Information (NMI)** for assessing clustering performance.
3. **Application to Time-Series Data** - Demonstrates clustering of synthetic time-series data using the modified K-means algorithm.
4. **Application to Relational Data** - Proposes a representation scheme for clustering relational data and evaluates clustering results.

## Dataset Sources
- **Time-Series Data:** Downloaded from the [UCI Synthetic Control Dataset](http://kdd.ics.uci.edu/databases/synthetic_control/synthetic_control.html).
- **Relational Data:** Provided as a simulated dataset, representing financial transaction data over time.

## Implementation Steps
1. **Time-Series Clustering:**
   - Apply the modified K-means algorithm to the synthetic control dataset.
   - Evaluate clustering performance against ground truth using ARI and NMI.
   - Analyze the relationship between clustering parameters (e.g., cluster count) and results.

2. **Relational Data Clustering:**
   - Design a representation scheme to convert relational data into a format suitable for clustering.
   - Apply the modified K-means algorithm to the relational dataset.
   - Evaluate clustering performance against known groups using ARI and NMI.
   - Analyze the suitability of the modified K-means algorithm for this dataset.

## Results and Analysis
- **Time-Series Data:**
  - Demonstrates effective clustering with high ARI and NMI scores.
  - Shows correlation between cluster count and performance.

- **Relational Data:**
  - Evaluates the ability of the modified K-means algorithm to handle relational data.
  - Discusses algorithm performance and potential challenges due to data representation.

## Dependencies
- Python 3.x
- NumPy
- Scikit-learn
- Pandas
- Matplotlib

## Usage
1. Clone this repository.
2. Install the required dependencies.
3. Run the scripts for time-series and relational data clustering:
   ```bash
   python clustering.py
   ```
4. View the clustering results and performance metrics in the output logs.

## Future Work
- Extend the algorithm to support hierarchical and density-based clustering methods.
- Improve preprocessing techniques for relational datasets.
- Test scalability with larger datasets.
