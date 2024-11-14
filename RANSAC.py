import numpy as np
import matplotlib.pyplot as plt 

def RANSAC(X,itr=200,threshold=20):
    """
    Inputs:
    X- sample coordinates
    
    itr - number of iterations 
    threshold - error threshold 
    
    output:
    Two coordinates of a line if RANSAC found the best fit 
    """
    N = X.shape[0] #size of the smaple 
    n=0 
    for i in range(itr):
        idx = np.random.randint(N,size=(1,2))[0]  #generating two random indices  
        Xsample = X[idx]
        p1 = Xsample[0]
        p2 = Xsample[1]
        inliers= 0
        for p3 in X:
            #let's find the error corresponding to this random selection 
            d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1) #distance between point (p3), and a line formed by p1 & p2 . This should be as small as possible 
            if d<threshold:
                inliers+=1 # finding out the inliers that satisfy the distance condition
        if inliers > n:
            n = inliers
            p1_best = p1
            p2_best  = p2 
    if inliers > 0:
        return p1_best , p2_best 
    return None ,None  #No RANSAC found 


if __name__ == "__main__":
    rdm = np.random.RandomState(9)  # to repeat the random state 
    X = rdm.randint(100,size=(100,2))

    plt.scatter(X[:,0],X[:,1],c="g",label="Samples")
    # plt.show()
    p1,p2 = RANSAC(X)
    # print(p1,p2)
    if p1 is not None:
        plt.scatter(p1[0],p1[1],c="r")
        plt.scatter(p2[0],p2[1],c="r")
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],c="r",label="RANSAC")

    plt.legend()
    plt.show()
