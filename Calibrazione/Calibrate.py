import numpy as np
import math



def Calibrate(corrispondenze):
    print("sono dentro calibrate")
    H = []
    for corrispondenza in corrispondenze:
        H.append(compute_view_based_homography(corrispondenza))
    A = get_intrinsic_parameters(H)
    return A


def compute_view_based_homography(corrispondenze):
    print("calcolo omografia")
    image_points = corrispondenze[0]
    object_points = corrispondenze[1]
    normalized_image_points = corrispondenze[2]
    normalized_object_points = corrispondenze[3]
    N_u = corrispondenze[4]
    N_x = corrispondenze[5]
    N_u_inv = corrispondenze[6]
    N_x_inv = corrispondenze[7]

    N = len(image_points)

    M = np.zeros((2 * N, 9), dtype=np.float64)
 
    for i in range(N):
        X, Y = normalized_object_points[i]  # A
        u, v = normalized_image_points[i]  # B

        row_1 = np.array([-X, -Y, -1, 0, 0, 0, X * u, Y * u, u])
        row_2 = np.array([0, 0, 0, -X, -Y, -1, X * v, Y * v, v])
        M[2 * i] = row_1
        M[(2 * i) + 1] = row_2


    u, s, vh = np.linalg.svd(M)

    h_norm = vh[np.argmin(s)]
    h_norm = h_norm.reshape(3, 3)
    print(N_u_inv)
    print(N_x)
    h = np.matmul(np.matmul(N_u_inv, h_norm), N_x)

    h = h[:, :] / h[2, 2]

    print("Homography for View : \n", h)
    return h




def estimateHomography(corrispondenze):
    N = len(corrispondenze)
    ret_correspondences = []
    for i in range(N):
        imp, objp = corrispondenze[i]
        N_x, N_x_inv = getNormalisationMatrix(objp, "A")
        N_u, N_u_inv = getNormalisationMatrix(imp, "B")
        
        hom_imp = np.array([[[each[0]], [each[1]], [1.0]] for each in imp])
        hom_objp = np.array([[[each[0]], [each[1]], [1.0]] for each in objp])

        normalized_hom_imp = hom_imp
        normalized_hom_objp = hom_objp

        for i in range(normalized_hom_objp.shape[0]):
            
            n_o = np.matmul(N_x, normalized_hom_objp[i])
            normalized_hom_objp[i] = n_o / n_o[-1]

            n_u = np.matmul(N_u, normalized_hom_imp[i])
            normalized_hom_imp[i] = n_u / n_u[-1]

        normalized_objp = normalized_hom_objp.reshape(normalized_hom_objp.shape[0], normalized_hom_objp.shape[1])
        normalized_imp = normalized_hom_imp.reshape(normalized_hom_imp.shape[0], normalized_hom_imp.shape[1])

        normalized_objp = normalized_objp[:, :-1]
        normalized_imp = normalized_imp[:, :-1]

        ret_correspondences.append((imp, objp, normalized_imp, normalized_objp, N_u, N_x, N_u_inv, N_x_inv))

    return ret_correspondences
    



def getNormalisationMatrix(punti, nome = "A"):
    punti = punti.astype(np.float64)
    
    x_mean, y_mean = np.mean(punti, axis=0)
    var_x, var_y = np.var(punti, axis=0)    
    

    s_x, s_y = np.sqrt(2 / var_x), np.sqrt(2 / var_y)


    Nx = np.array([[s_x, 0, -s_x * x_mean], [0, s_y, -s_y * y_mean], [0, 0, 1]])
    Nx_inv = np.array([[1. / s_x, 0, x_mean], [0, 1. / s_y, y_mean], [0, 0, 1]])
    return Nx, Nx_inv


def v_pq(p, q, H):
    v = np.array([
            H[0, p] * H[0, q],
            H[0, p] * H[1, q] + H[1, p] * H[0, q],
            H[1, p] * H[1, q],
            H[2, p] * H[0, q] + H[0, p] * H[2, q],
            H[2, p] * H[1, q] + H[1, p] * H[2, q],
            H[2, p] * H[2, q]
    ])
    return v

def get_intrinsic_parameters(H_r):
    M = len(H_r)
    V = np.zeros((2 * M, 6), np.float64)
    for i in range(M):
        H = H_r[i]
        V[2 * i] = v_pq(p=0, q=1, H=H)
        V[2 * i + 1] = np.subtract(v_pq(p=0, q=0, H=H), v_pq(p=1, q=1, H=H))

    u, s, vh = np.linalg.svd(V)
    b = vh[np.argmin(s)]
    print("V.b = 0: ", b.shape)

    vc = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] ** 2)
    l = b[5] - (b[3] ** 2 + vc * (b[1] * b[2] - b[0] * b[4])) / b[0]
    alpha = np.sqrt((l / b[0]))
    beta = np.sqrt(((l * b[0]) / (b[0] * b[2] - b[1] ** 2)))
    gamma = -1 * ((b[1]) * (alpha ** 2) * (beta / l))
    uc = (gamma * vc / beta) - (b[3] * (alpha ** 2) / l)

    print([vc,
           l,
           alpha,
           beta,
           gamma,
           uc])

    A = np.array([
        [alpha, gamma, uc],
        [0, beta, vc],
        [0, 0, 1.0],
    ])
    print("Valori intrinseci della matrice :")
    print(A)
    return A