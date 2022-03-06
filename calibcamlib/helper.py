import numpy as np

def intersect(P,V):
    """This function returns the least squares intersection of the N
    lines from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    Implementation from https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python
    """
    # generate all line direction vectors 
    P = P[~np.isnan(V[:,1])]
    V = V[~np.isnan(V[:,1])]

    if len(V)<2:
        ret = np.empty(shape=(1,3))
        ret[:] = np.NaN
        return ret
    
    n = V/np.linalg.norm(V,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors 
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T
    # see fig. 1 

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    q = (projs @ P[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the 
    # intersection point p: Rp = q
    p = np.linalg.lstsq(R,q,rcond=None)[0]

    return p

def calc_3derr(X,P,V):
    dists = np.zeros(shape=(P.shape[0]))
    
    for i,p in enumerate(P):
        dists[i] = calc_min_line_point_dist(X,p,V[i])
    
    if np.all(np.isnan(dists)):
        return (np.nansum(dists**2),dists)
    else:
        return (np.NaN,dists)
        
def calc_min_line_point_dist(x,p,v):
    #print(x.shape)
    #print(p.shape)
    #print(v.shape)
    d = x-p;
    dist = np.sqrt(np.sum((d-np.sum(d*v,axis=1)[:,np.newaxis]@v)**2))
    return dist

