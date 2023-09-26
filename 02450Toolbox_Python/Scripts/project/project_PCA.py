import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.colors as mcolors
from itertools import cycle


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

import matplotlib.pyplot as plt

import statsmodels.api as sm

from project_ETL import * 


print('######################### PCA #########################')

data = raw_data_clean
data = pd.get_dummies(data, columns = ['ocean_proximity']) 
attributeNames = data.columns

X = data.copy()#.drop(['median_house_value'], axis =1 )
y_cont = data['median_house_value']


# X_scale = StandardScaler().fit_transform(X)
# pca = PCA(0.9) # Use enough PC to capture 90% of the variability
# pca.fit(X_scale) 
# X_trans = pca.transform(X_scale)
# print(X_trans.shape[1], ' principal components are needed to cover 90% variability for this data') 
# 7 principal components are needed to cover 90% variability for this data

#Scree Plot 
# plt.plot(range(1, 8), np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Proportion of Variance Explained')
# plt.show()



#Doing PCA using methods from exercises: 


# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y = X - 1*X.mean(0)
Y = Y*(1/np.std(Y,0))
# Here we're utilizing the broadcasting of a row vector to fit the dimensions 
# of Y

threshold = 0.9
# Choose two PCs to plot (the projection)
i = 0
j = 1

# Make the plot
plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
plt.title('Effect of standardization')
nrows=3
ncols=2

# Obtain the PCA solution by calculate the SVD of either Y1 or Y
U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T # For the direction of V to fit the convention in the course we transpose
# For visualization purposes, we flip the directionality of the
# principal directions such that the directions match for Y1 and Y.

# Compute variance explained
rho = (S*S) / (S*S).sum() 

# Compute the projection onto the principal components
Z = U*S;

# Plot projection
plt.subplot(nrows, ncols, 1)
C = len(classNames)
for c in range(C):
    plt.plot(Z[y==c,i], Z[y==c,j], '.', alpha=.5)
plt.xlabel('PC'+str(i+1))
plt.xlabel('PC'+str(j+1))
plt.title('Standardized Dataset' + '\n' + 'Projection' )
plt.legend(classNames)
plt.axis('equal')


        
# Plot cumulative variance explained
plt.subplot(nrows, ncols, 2);
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.title('Standardized Dataset'+'\n'+'Variance explained')




# Plot attribute coefficients in principal component space on 4 different plots to make sure 
# labels are visible
# Principal directions
intervals = [
    [1,5,9,13],
    [2,6,10,0],
    [3,7,11],
    [4,8,12]
]
plot_loc = 3
for interval in intervals:
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols, plot_loc)
    
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01)))
    plt.grid()
    plt.title('Standardized Dataset' + '\n' + 'Attribute coefficients')
    plt.axis('equal')

    # Generate a list of distinct colors
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_cycle = cycle(colors)

    for att in interval:
        color = next(color_cycle)
        plt.arrow(0, 0, V[att, i], V[att, j], color=color, label=attributeNames[att])

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))

    # Add a legend inside the plot with numbers
    plt.legend(loc='upper right', title="Legend")


    plot_loc = plot_loc + 1















plt.show()


print("")