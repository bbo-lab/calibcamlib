import numpy as np
from . import Camera
from .helper import intersect

# R,t are world->cam

class Camerasystem:
    cameras = list();
    
    def __init__(self):
        pass
    
    def add_camera(self, A, k, R, t):
        self.cameras.append({'camera': Camera(A,k),'R': R,'t': t})
        
        
    def project(self,X):
        x = np.zeros(shape=(len(self.cameras),X.shape[0],2))

        for i,c in enumerate(self.cameras):
            x[i] = (c['camera'].space_to_sensor((c['R']@X.T).T + c['t']).T).T

        return x
            
            
    def triangulate_3derr(self,x):
        #TODO support more than one point!
        V = np.empty(shape=(x.shape[0],x.shape[1],3))
        P = np.empty(shape=(x.shape[0],x.shape[1],3))
        
        for i,c in enumerate(self.cameras):
            V[i,:] = c['camera'].sensor_to_space(x[i]) @ c['R']
            P[i,:] = np.tile(-c['R'].T@c['t'],(x.shape[1],1))

        X = np.empty(V.shape[1:])
        for i,Xp in enumerate(X):
            X[i] = intersect(P[:,i,:],V[:,i,:]).T

        return (X,P,V)

    
    def from_calibcam_file(filename):
        cs = Camerasystem()
        calib = np.load(filename, allow_pickle=True).item()

        for i in range(len(calib['RX1_fit'])):
            A = np.array([
                    [calib['A_fit'][i][0], 0,                    calib['A_fit'][i][1]],
                    [0,                    calib['A_fit'][i][2], calib['A_fit'][i][3]],
                    [0,                    0,                    1]
                ])

            cs.add_camera(A,
                          calib['k_fit'][i],
                          calib['RX1_fit'][i],
                          calib['tX1_fit'][i]*calib['square_size_real']
                          )
            
        return cs

    
    
    
