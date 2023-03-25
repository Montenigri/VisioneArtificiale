import glob
import numpy as np
import cv2
from scipy import optimize as opt

PATTERN_SIZE = (9, 6)
SQUARE_SIZE = 1.0 

def get_camera_images(dir):
    images = glob.glob(dir + "*.jpeg")
    images = sorted(images)
    for each in images:
        yield (each, cv2.imread(each, 0))


def getChessboardCorners(dir):
    objp = np.zeros((PATTERN_SIZE[1]*PATTERN_SIZE[0], 3), dtype=np.float64)
    objp[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    chessboard_corners = []
    image_points = []
    object_points = []
    correspondences = []
    getImage = get_camera_images(dir)
    for (path, each) in getImage:  #images:
       # print("Processing Image : ", path)
        ret, corners = cv2.findChessboardCorners(each, patternSize=PATTERN_SIZE)
        if ret:
           # print ("Chessboard Detected ")
            corners = corners.reshape(-1, 2)
            if corners.shape[0] == objp.shape[0] :
                image_points.append(corners)
                object_points.append(objp[:,:-1]) #Togliamo Z perché ci interessano i punti nel piano XY
                correspondences.append([corners.astype(int), objp[:, :-1].astype(int)])

    return correspondences

def get_normalization_matrix(pts, name="A"):
        
    pts = pts.astype(np.float64)
    x_mean, y_mean = np.mean(pts, axis=0)
    var_x, var_y = np.var(pts, axis=0)
    s_x , s_y = np.sqrt(2/var_x), np.sqrt(2/var_y)
    n = np.array([[s_x, 0, -s_x*x_mean], [0, s_y, -s_y*y_mean], [0, 0, 1]])
    n_inv = np.array([ [1./s_x ,  0 , x_mean], [0, 1./s_y, y_mean] , [0, 0, 1] ])

    return n.astype(np.float64), n_inv.astype(np.float64)


def normalize_points(chessboard_correspondences):
    views = len(chessboard_correspondences)
    ret_correspondences = [] 
    for i in range(views):
        imp, objp = chessboard_correspondences[i]
        N_x, N_x_inv = get_normalization_matrix(objp, "A")
        N_u, N_u_inv = get_normalization_matrix(imp, "B")
        
        # converto imp ed objp a omogenei
        hom_imp = np.array([ [[each[0]], [each[1]], [1.0]] for each in imp])
        hom_objp = np.array([ [[each[0]], [each[1]], [1.0]] for each in objp])

        normalized_hom_imp = hom_imp
        normalized_hom_objp = hom_objp

        for i in range(normalized_hom_objp.shape[0]):
           
            n_o = np.matmul(N_x,normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o/n_o[-1]
            
            n_u = np.matmul(N_u,normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u/n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:,:-1]        
        normalized_imp = normalized_imp[:,:-1]

        ret_correspondences.append((imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv))

    return ret_correspondences

def compute_view_based_homography(corrispondenze):
    
    image_points = corrispondenze[0]
    object_points = corrispondenze[1]
    normalized_image_points = corrispondenze[2]
    normalized_object_points = corrispondenze[3]
    N_u = corrispondenze[4]
    N_x = corrispondenze[5]
    N_u_inv = corrispondenze[6]
    N_x_inv = corrispondenze[7]

    N = len(image_points)

    M = np.zeros((2*N, 9), dtype=np.float64)
     
    for i in range(N):
        X, Y = normalized_object_points[i] #A
        u, v = normalized_image_points[i] #B
        row_1 = np.array([ -X, -Y, -1, 0, 0, 0, X*u, Y*u, u])
        row_2 = np.array([ 0, 0, 0, -X, -Y, -1, X*v, Y*v, v])
        M[2*i] = row_1
        M[(2*i) + 1] = row_2

    u, s, vh = np.linalg.svd(M)
    h_norm = vh[np.argmin(s)]
    h_norm = h_norm.reshape(3, 3)
    h = np.matmul(np.matmul(N_u_inv,h_norm), N_x)
    h = h[:,:]/h[2, 2]    

    return h




        



    

def v_pq(p, q, H):
    v = np.array([
            H[0, p]*H[0, q],
            H[0, p]*H[1, q] + H[1, p]*H[0, q],
            H[1, p]*H[1, q],
            H[2, p]*H[0, q] + H[0, p]*H[2, q],
            H[2, p]*H[1, q] + H[1, p]*H[2, q],
            H[2, p]*H[2, q]
        ])
    return v

def get_intrinsic_parameters(H_r):
    
    M = len(H_r)
    V = np.zeros((2*M, 6), np.float64)

    for i in range(M):
        H = H_r[i]
        V[2*i] = v_pq(p=0, q=1, H=H)
        V[2*i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))

    # risoluzione V.b = 0
    u, s, vh = np.linalg.svd(V)
    b = vh[np.argmin(s)]

    vc = (b[1]*b[3] - b[0]*b[4])/(b[0]*b[2] - b[1]**2)
    l = b[5] - (b[3]**2 + vc*(b[1]*b[2] - b[0]*b[4]))/b[0]
    alpha = np.sqrt((l/b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2] - b[1]**2)))
    gamma = -1*((b[1])*(alpha**2) *(beta/l))
    uc = (gamma*vc/beta) - (b[3]*(alpha**2)/l)

    A = np.array([
            [alpha, gamma, uc],
            [0, beta, vc],
            [0, 0, 1.0],
        ])
    
    return A

def getCameraCal(dir):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = [] 
    
    objp = np.zeros((1, PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    getImage = get_camera_images(dir)
    for (path, each) in getImage:
        gray = each
        ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx

def calibra(dir):
    dir = f"datiLaboratorio/checkboard/{str(dir)}/"
    chessboard_correspondences = getChessboardCorners(dir)
        
    chessboard_correspondences_normalized = normalize_points(chessboard_correspondences)

    H = []
    for correspondence in chessboard_correspondences_normalized:
        H.append(compute_view_based_homography(correspondence))

    CameraIntrinsic = get_intrinsic_parameters(H)
    CalcolataDaCV2 = getCameraCal(dir)
    Errore = np.absolute(CameraIntrinsic  - CalcolataDaCV2)
    print("<----------------->")
    print("<----------------->")
    print("<----------------->")
    print("Camera Intrinsic")
    print(CameraIntrinsic)
    print("<----------------->")
    print("<----------------->")
    print("<----------------->")
    print ("Calcolata da CV2")
    print(CalcolataDaCV2)
    print("<----------------->")
    print("<----------------->")
    print("<----------------->")
    print("Errore")
    print(Errore)


