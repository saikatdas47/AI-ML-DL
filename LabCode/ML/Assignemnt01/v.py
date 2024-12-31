import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to compute the multivariate normal distribution
def multivariate_normal(x, mean, cov):
    d = len(mean)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    x_minus_mean = x - mean
    exponent = -0.5 * np.dot(np.dot(x_minus_mean.T, cov_inv), x_minus_mean)
    normalization = 1 / np.sqrt(((2 * np.pi) ** d) * cov_det)
    return normalization * np.exp(exponent)

# Given parameters for two classes
mean1 = np.array([0, 0])  # Mean for Class 1
cov1 = np.array([[0.25, 0.3], [0.3, 1]])  # Covariance for Class 1
prior1 = 0.5  # Prior for Class 1

mean2 = np.array([2, 2])  # Mean for Class 2
cov2 = np.array([[0.5, 0], [0, 0.5]])  # Covariance for Class 2
prior2 = 0.5  # Prior for Class 2

# Load test data from the text file (2D points)
test_data = np.loadtxt("test.txt", delimiter=',')  # Assuming test.txt has two columns (x, y)

# Classify the test data points
class_labels = []
for point in test_data:
    # Compute probability for each class
    px_w1 = multivariate_normal(point, mean1, cov1) * prior1
    px_w2 = multivariate_normal(point, mean2, cov2) * prior2
    if px_w1 > px_w2:
        class_labels.append(1)  # Assign to Class 1
    else:
        class_labels.append(2)  # Assign to Class 2

# Convert class labels to numpy array
class_labels = np.array(class_labels)

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the test points with class-based coloring and markers
for i, point in enumerate(test_data):
    if class_labels[i] == 1:
        ax.scatter(point[0], point[1], 0, c='red', marker='o', label='Class 1' if i == 0 else "")
    else:
        ax.scatter(point[0], point[1], 0, c='blue', marker='^', label='Class 2' if i == 0 else "")

# Plotting the probability distributions (contour) in 3D
x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 4, 100)
X, Y = np.meshgrid(x, y)
pdf1 = np.zeros_like(X)
pdf2 = np.zeros_like(X)

# Calculate probability density for each point on the grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        pdf1[i, j] = multivariate_normal(point, mean1, cov1)
        pdf2[i, j] = multivariate_normal(point, mean2, cov2)

# Plot the probability contours in 3D for both classes
ax.contour3D(X, Y, pdf1, 10, cmap='Reds', alpha=0.5)
ax.contour3D(X, Y, pdf2, 10, cmap='Blues', alpha=0.5)

# Decision boundary in 3D (where pdf1 == pdf2)
decision_boundary = np.abs(pdf1 - pdf2)
ax.contour3D(X, Y, decision_boundary, levels=[0.01], colors='black', linewidths=2)

# Axis labels and legend
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Probability Density")
ax.legend()

# Show the plot
plt.show()



# f(x) = \frac{1} sqrt{{(2\pi)^{d} }\sqrt{|Cov|}} \exp \left( -\frac{1}{2} (x - \mu)^T \cdot Cov^{-1} \cdot (x - \mu) \right)
