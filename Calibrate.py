import numpy as np
import math

###
#
#Input: X serie ordinata di punti nel mondo 3D del piano target 
#e U serie di punti ordinata ed in relazione con X sul piano 2D.
#
#Output: Paramentri intrinseci stimati della camera
#
###

def Calibrate(X,U):
    Hinit = getHomographies(X, U)
    Ainit = getCameraIntrinsics(Hinit)
    Winit = getExtrinsic(Ainit, Hinit)
    Kinit = EstLensDistrortion(Ainit, Winit, X, U)
    A, k, W = refineAll(Ainit,Kinit,Winit,X,U)
    return A,k,W

####
#
# Accetta in input i punti X del modello 3D e i relativi punti nel sensore per ritornare
# una stima dell'homografia
#
####
def getHomographies(X,U):
    M     = len(U)
    H     = []
    for i in range (0,M):
        Hinit = estimateHomography(X,U[i])
        Href = refineHomography(Hinit, X, U[i])
        H.append(Href)
    return H

###
#
# Accetta in input P e Q serie di punti 2D e ritorna l'omografia stimata
#
###

def hom(v):
    v.append(1)

def dehom(v):
    newV = [v[1]/v[3], v[2]/v[3]]
    return newV

def estimateHomography(P,Q):
    N = len(P)
    Np = getNormalisationMatrix(P)
    Nq = getNormalisationMatrix(Q)

    M = np.empty([(2*N), 9])

    for i in range (0,N):
        k = 2* i

        pfirst = dehom( Np * hom(P[i]))
        qfirst = dehom( Nq * hom(Q[i]))

        M[k,:]   = pfirst[1],pfirst[2],1,0,0,0,-(pfirst[1]*qfirst[1]),-(pfirst[2]*qfirst[1]),-qfirst[1]
        M[k+1,:] = 0,0,0,pfirst[1],pfirst[2],1,-(pfirst[1]*qfirst[2]),-(pfirst[2]*qfirst[1]),-qfirst[1]
    V = np.linalg.svd(M)
    h = V[-1]

    Hfirst = np.array([[h[0:2]][h[3:5]][h[6:8]]])

    H = np.matmul(np.transpose(Nq), np.matmul(Hfirst,Np))
    return H

def getNormalisationMatrix(x):
    N = len(x)
    
    
    xsign, sigmaSqX = sigmaSq(x[:,0])
    Ysign, sigmaSqY = sigmaSq(x[:,1])

    sx = math.sqrt(2/sigmaSqX)
    sy = math.sqrt(2/sigmaSqY)

    Nx = np.array([[sx,0,-(sx*xsign)][0,sy,-(sy*Ysign)][0,0,1]])
    return Nx


def sigmaSq(x):
    sigma = 0
    xsign = (1/N) * sum(x)
    for i in range (0,len(x)):
        sigma = sigma + (x[i]-xsign)**2
    return xsign, sigma


