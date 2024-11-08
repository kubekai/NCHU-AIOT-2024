import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC

# Streamlit App
st.title('3D Scatter Plot with Separating Hyperplane')

# Sidebar for adjusting distance threshold
distance_threshold = st.slider('Distance Threshold', min_value=0.1, max_value=10.0, value=4.0, step=0.1)

# Generate data and perform calculations
np.random.seed(0)
num_points = 600
mean = 0
variance = 10

# Generate data for x1 and x2 from normal distribution
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Calculate distances from the origin (0, 0)
distances = np.sqrt(x1**2 + x2**2)

# Assign labels based on the distance threshold
Y = np.where(distances < distance_threshold, 0, 1)

# Gaussian function for third dimension (x3)
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# Calculate x3 values using the Gaussian function
x3 = gaussian_function(x1, x2)

# Stack x1, x2, x3 to form the feature matrix X
X = np.column_stack((x1, x2, x3))

# Train a Linear SVM model
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)

# Extract model coefficients and intercept
coef = clf.coef_[0]
intercept = clf.intercept_

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points for class Y=0 and Y=1
ax.scatter(x1[Y==0], x2[Y==0], x3[Y==0], c='blue', marker='o', label='Y=0')
ax.scatter(x1[Y==1], x2[Y==1], x3[Y==1], c='red', marker='s', label='Y=1')

# Labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('3D Scatter Plot with Y Color and Separating Hyperplane')

# Plot the separating hyperplane using meshgrid
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10), np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

# Plot the hyperplane surface
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# Display the plot using Streamlit
ax.legend()
st.pyplot(fig)
