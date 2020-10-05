
"""
@author: simen og kristoffer
"""
import numpy as np

def plante(ex,ey,ep,D,eq=None):
    
    Dshape = D.shape
    if Dshape[0] != 3:
        raise NameError('Wrong constitutive dimension in plante')
        
    if ep[0] == 1 :
        return tri3e(ex,ey,D,ep[1],eq)
    else:
        Dinv = np.inv(D)
        return tri3e(ex,ey,Dinv,ep[1],eq)


def tri3e(ex,ey,D,th,eq=None):
    """
    Compute the stiffness matrix for a two dimensional beam element.
    
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return mat Ke: element stiffness matrix [6 x 6]
    :return mat fe: consistent load vector [6 x 1] (if eq!=None)
    """
    
    tmp = np.matrix([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])
    
    A2 = np.linalg.det(tmp)  # Double of triangle area

    A  = A2 / 2.0
       
    zeta_px, zeta_py = zeta_partials_x_and_y(ex,ey) #Partial derivatives of zeta with respect to x and y 
    
    B  = np.array([[zeta_px[0], 0, zeta_px[1], 0, zeta_px[2], 0], #B matrix is strain displacement matrix for x and y
                            [0, zeta_py[0], 0, zeta_py[1], 0, zeta_py[2]],
                            [zeta_py[0], zeta_px[0], zeta_py[1], zeta_px[1], zeta_py[2], zeta_px[2]]])
    
    Ke = np.mat(np.zeros((6, 6)))
    Ke = (B.T * D * B) * A * th #K matrix, element stiffness matrix


    if eq is None:
        return Ke
    else:
        fe = np.mat(np.zeros((6,1)))
        fx = A * th * eq[0]/ 3.0
        fy = A * th * eq[1]/ 3.0
        fe = np.mat([[fx],[fy],[fx],[fy],[fx],[fy]])
        return Ke, fe
        
def zeta_partials_x_and_y(ex,ey):
    """
    Compute partials of area coordinates with respect to x and y.
    
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    """

    tmp = np.matrix([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])
    
    A2 = np.linalg.det(tmp)  # Double of triangle area
       
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k
    
    zeta_px = np.zeros(3)           # Partial derivative with respect to x
    zeta_py = np.zeros(3)           # Partial derivative with respect to y

   
    for i in range(3): 
        j = cyclic_ijk[i+1]
        k = cyclic_ijk[i+2]
        zeta_px[i] = (ey[j] - ey[k]) / A2
        zeta_py[i] = (ex[k] - ex[j]) / A2

    return zeta_px, zeta_py


# Functions for 6 node triangle
    
def tri6_area(ex,ey): #Function that calculates area of a triangle
        
    tmp = np.matrix([[1,ex[0],ey[0]],
                     [1,ex[1],ey[1]],
                     [1,ex[2],ey[2]]])
    
    A = np.linalg.det(tmp) / 2
    
 
    return A


def tri6_shape_functions(zeta):
    '''
    Compution shape functions of a 6 node element 
    :param list zeta: Area coordinates
    :return list N6: shape-functions/interpolation-functions 
    '''

    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    N6 = np.zeros(6)

    #Shape functions for nodes 1-3, separated because they are calculated differently
    for i in range(3):
        N6[i] = zeta[i] * (zeta[i] - 0.5) * 2.0
        
    #Shape functions for nodes 4-6
    for i in range(3):
        j = cyclic_ijk[i+1]
        k = cyclic_ijk[i+2]  
        N6[i+3] = zeta[i] * zeta[j] * 4.0

    return N6


def tri6_shape_function_partials_x_and_y(zeta,ex,ey):
    '''
    Computing partial derivatives of shape functions with respect to x and y
    :param list zeta: Area coordinates
    :param list ex: element x coordinates [x1, x2, x3, x4, x5, x6]
    :param list ey: element y coordinates [y1, y2, y3, y4, y5, y6]
    :return list N6_px: partial derivatives with respect to x of shape functions 
    :return list N6_py: partial derivatives with respect to y of shape functions 
    '''

    zeta_px, zeta_py = zeta_partials_x_and_y(ex,ey)
    
    N6_px = np.zeros(6)
    N6_py = np.zeros(6)
    
    cyclic_ijk = [0,1,2,0,1]      # Cyclic permutation of the nodes i,j,k

    #Computing partial derivatives from formula 
    for i in range(3):
        j = cyclic_ijk[i+1]
        N6_px[i] = (4 * zeta[i] - 1) * zeta_px[i]
        N6_py[i] = (4 * zeta[i] - 1) * zeta_py[i]
        N6_px[i+3] = 4 * zeta[j] * zeta_px[i] + 4 * zeta[i] * zeta_px[j]
        N6_py[i+3] = 4 * zeta[j] * zeta_py[i] + 4 * zeta[i] * zeta_py[j]

    return N6_px, N6_py


def tri6_Bmatrix(zeta,ex,ey):
    """
    Calculating the strain displacement matrix
    :param list zeta: Area coordinates for the respective nodes in the 6-node element, e.g [0.5 0.5 0.]
    :param list ex: element x coordinates [x1, x2, x3, x4, x5, x6]
    :param list ey: element y coordinates [y1, y2, y3, y4, y5, y6]
    :return matrix B: Strain displacement matrix
    """
    
    nx,ny = tri6_shape_function_partials_x_and_y(zeta, ex, ey)

    Bmatrix = np.matrix([
        [nx[0], 0, nx[1], 0, nx[2], 0, nx[3], 0, nx[4], 0, nx[5], 0],
        [0, ny[0], 0, ny[1], 0, ny[2], 0, ny[3], 0, ny[4], 0, ny[5]],
        [ny[0], nx[0], ny[1], nx[1], ny[2], nx[2], ny[3], nx[3], ny[4], nx[4], ny[5], nx[5]]])


    return Bmatrix


def tri6_Kmatrix(ex,ey,D,th,eq=None):
    '''
    Calculating element stiffness matrix for 6-node triangle
    :param list ex: element x coordinates [x1, x2, x3, x4, x5, x6]
    :param list ey: element y coordinates [y1, y2, y3, y4, y5, y6]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]
    :return matrix K: element stiffness matrix [12x12]
    :return matrix fe: consistent load vector [12 x 1] (if eq!=None)

    '''
    
    zetaInt = np.array([[0.5,0.5,0.0], #Areacoordinates for node 4,5 and 6
                        [0.0,0.5,0.5],
                        [0.5,0.0,0.5]])
    
    wInt = np.array([1.0/3.0,1.0/3.0,1.0/3.0]) #Weights for the gauss points used in the integration 

    A    = tri6_area(ex,ey) #Area of triangles 
    
    Ke = np.zeros((12, 12)) #Creating empty array of zeroes with 12x12 dimention, will be filled by values to become the stiffness matrix
    fe = np.zeros((12, 1)) #Creating empty array of zeroes that will be the consistent load vector
    

    for iGauss in range(3): #Numerical integration of the elemental stiffness matrix and load vector 
        zeta = zetaInt[iGauss]

        B = tri6_Bmatrix(zeta, ex, ey) 
        Ke += (B.T @ D @ B) * A * th * wInt[iGauss] #Adding stiffness matrix of every iteration

        if eq is not None:
            f = np.array([[eq[0]], [eq[1]]]) #load vector 
            N6 = tri6_shape_functions(zeta)
            N2 = np.zeros((2, 12)) #displacement interpolation matrix

        for i in range(6):
            N2[0, i * 2] = N6[i]
            N2[1, 1 + i * 2] = N6[i]

        fe += N2.T @ f * A * wInt[iGauss] #numerical integration of load vector 

    if eq is None:
        return Ke
    else:
        return Ke, fe

def tri6e(ex,ey,D,th,eq=None):
    return tri6_Kmatrix(ex,ey,D,th,eq)


