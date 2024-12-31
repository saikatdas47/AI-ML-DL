import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def multivariate_normal(x, mean, cov):
    d = len(mean)
    cov_det = np.linalg.det(cov)  
    cov_inv = np.linalg.inv(cov)  
    x_minus_mean = x - mean
    exponent = -0.5 * np.dot(np.dot(x_minus_mean.T, cov_inv), x_minus_mean)
    normalization = 1 / np.sqrt(((2 * np.pi) ** d) * cov_det)     
    return normalization * np.exp(exponent)



mean1 = np.array([0, 0])
cov1 = np.array([[0.25, 0.3], [0.3, 1]])
prior1 = 0.5

mean2 = np.array([2, 2])
cov2 = np.array([[0.5, 0], [0, 0.5]])
prior2 = 0.5

test_data = np.loadtxt("/Users/saikatdas/Desktop/4.1/Pattern Recog/Lab/Assignemnt01/test.txt", delimiter=',')

class_labels = [] #empty list 
for point in test_data:
   
    px_w1 = multivariate_normal(point, mean1, cov1) * prior1
 
    px_w2 = multivariate_normal(point, mean2, cov2) * prior2

    print(point)
    print(px_w1, px_w2)

    if px_w1 > px_w2:
        class_labels.append(1)
    else:
        class_labels.append(2)






class_labels = np.array(class_labels)
fig = plt.figure(figsize=(12, 8))   #12*8 inch
ax = fig.add_subplot(111, projection='3d') #1st row, 1st column, 1st plot

for i, point in enumerate(test_data):
    if class_labels[i] == 1:
        ax.scatter(point[0], point[1], 0, c='red', marker='o', label='Class 1' if i == 0 else "")
    else:
        ax.scatter(point[0], point[1], 0, c='blue', marker='^', label='Class 2' if i == 0 else "")

x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 4, 100)
X, Y = np.meshgrid(x, y)
pdf1 = np.zeros_like(X)
pdf2 = np.zeros_like(X)

for i in range(X.shape[0]):   #[X[i, j], Y[i, j]]।
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        pdf1[i, j] = multivariate_normal(point, mean1, cov1)
        pdf2[i, j] = multivariate_normal(point, mean2, cov2)

ax.contour3D(X, Y, pdf1, 10, cmap='Reds', alpha=0.5)
ax.contour3D(X, Y, pdf2, 10, cmap='Blues', alpha=0.5)

decision_boundary = np.abs(pdf1 - pdf2)
ax.contour3D(X, Y, decision_boundary, levels=[0.01], colors='black', linewidths=2)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Probability Density")
ax.legend()

plt.show()




