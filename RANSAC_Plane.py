import numpy as np
import matplotlib.pyplot as plt 

def EoP(p1,p2,p3):
    """for a given three points, finds out the equation of plane"""
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    A,B,C = cp
    D = np.dot(cp, p3)
    return A,B,C,D

def RANSAC(X,itr=5000,threshold=500):
    """
    Inputs:
    X- sample coordinates
    
    itr - number of iterations 
    threshold - error threshold 
    
    output:
    Three  coordinates of a plane if RANSAC found the best fit 
    """
    N = X.shape[0] #size of the smaple 
    n=0 
    for i in range(itr):
        idx = np.random.randint(N,size=(1,3))[0]  #generating three random indices  
        Xsample = X[idx]  #gather the coordinates with those indices 
        p1,p2,p3 = Xsample# unpacking three points 

        A,B,C,D = EoP(p1,p2,p3)  #Equation of plane 
        # print(A,B,C,D)
        inliers= 0
        #Now for every point in input sample, find if it is a inlier 
        for p in X:
            xi,yi,zi = p

            d = np.abs((A*xi+B*yi+C*zi+D)/np.sqrt(xi**2+yi**2+zi**2))
            if d<threshold:
                inliers+=1 # finding out the inliers that satisfy the distance condition
        if inliers > n:
            n = inliers
            Abest = A
            Bbest = B
            Cbest = C
            Dbest = D

    if inliers > 0:
        return (Abest , Bbest, Cbest, Dbest) 
    return None


if __name__ == "__main__":
    rdm = np.random.RandomState(10)  # to repeat the random state 
    X = rdm.randint(100,size=(100,3))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    Out = RANSAC(X)
    if Out:
        A,B,C,D = Out
        ax.scatter3D(X[:,0],X[:,1],X[:,2],c="g",label="Samples")
        xx, yy = np.meshgrid(range(100), range(100))
        z = -(A*xx+B*yy+D)/C
        ax.plot_surface(xx, yy, z, alpha=0.5) #,label = "Best Fit Plane ")
        plt.legend()
        plt.show()

    else:
        print("No best RANSAC")
