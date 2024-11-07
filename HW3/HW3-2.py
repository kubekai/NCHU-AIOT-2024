import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles
from sklearn.svm import SVC
import streamlit as st
from sklearn.preprocessing import StandardScaler
from matplotlib import cm

# Step 1: Generate Circular Dataset
X, y = make_circles(n_samples=300, factor=0.5, noise=0.05)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 2: Train SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(X, y)

# Step 3: Prepare Grid for Decision Boundary Visualization
xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
decision_scores = svm_rbf.decision_function(grid_points)
zz = decision_scores.reshape(xx.shape)

# Step 4: Streamlit App
st.title("2D SVM with RBF Kernel - 3D Decision Boundary Visualization")
st.write("This app shows a 3D visualization of an SVM classification decision boundary on a circularly distributed 2D dataset using an RBF kernel.")

# Plot 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')  

# Plot decision surface
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, color='blue', alpha=0.3, edgecolor='none', cmap=cm.coolwarm)

# Plot the data points in 3D
ax.scatter(X[y == 0, 0], X[y == 0, 1], -0.5, color='red', label="Class 0", s=50)
ax.scatter(X[y == 1, 0], X[y == 1, 1], -0.5, color='green', label="Class 1", s=50)

# Highlight support vectors
support_vectors = svm_rbf.support_vectors_
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], -0.5, s=100, facecolors='none', edgecolors='k', label="Support Vectors")

# Set labels
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Decision Function Value")
ax.set_title("SVM with RBF Kernel (3D Decision Boundary)")

# Add legend
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)
