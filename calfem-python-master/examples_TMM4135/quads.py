# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 08:15:51 2018
@author: bjohau
"""
import numpy as np
import sys

def gauss_points(iRule):
    """
    Returns gauss coordinates and weight given integration number

    Parameters:

        iRule = number of integration points

    Returns:

        gp : row-vector containing gauss coordinates
        gw : row-vector containing gauss weight for integration point
    """

    gauss_position = [[ 0.000000000],
                      [-0.577350269,  0.577350269],
                      [-0.774596669,  0.000000000,  0.774596669],
                      [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116],
                      [-0.9061798459, -0.5384693101, 0.0000000000, 0.5384693101, 0.9061798459]]
    gauss_weight   = [[2.000000000],
                      [1.000000000,   1.000000000],
                      [0.555555556,   0.888888889,  0.555555556],
                      [0.3478548451,  0.6521451549, 0.6521451549, 0.3478548451],
                      [0.2369268850,  0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850]]


    if iRule < 1 and iRule > 5:
        sys.exit("Invalid number of integration points.")

    idx = iRule - 1
    return gauss_position[idx], gauss_weight[idx]


def quad4_shapefuncs(xsi, eta):
    
    """
    Calculates shape functions evaluated at xsi and eta
    Uses zero-lines to find the shape 

    Parameters:
        xsi = Coordinate along the xsi-axis
        eta = Coordinate along the eta-axis
        
    Returns:
        N   = Array of shapefunctions for the given quad
    """
    
    # ----- Shape functions -----
    N    = np.zeros(4)
    N[0] = (1/4) * (1 + xsi) * (1 + eta)
    N[1] = (1/4) * (1 - xsi) * (1 + eta)
    N[2] = (1/4) * (1 - xsi) * (1 - eta)
    N[3] = (1/4) * (1 + xsi) * (1 - eta)
    return N


def quad4_shapefuncs_grad_xsi(xsi, eta):

    """
    Calculates derivatives of shape functions wrt. xsi from quad4_shapefuncs_grad
    Uses the chain rule to find the derivation of N with respect to eta

    Parameters:
        xsi  = Coordinate along the xsi-axis
        eta  = Coordinate along the eta-axis
        
    Returns:
        Ndxi = Array of the derivated shapefunctions with respect to xsi for the given quad
    """
    
    # ----- Derivatives of shape functions with respect to xsi -----
    Ndxi    = np.zeros(4)
    Ndxi[0] = (1/4) * (1 + eta)
    Ndxi[1] = (1/4) * (1 + eta) * (-1)
    Ndxi[2] = (1/4) * (1 - eta) * (-1)
    Ndxi[3] = (1/4) * (1 - eta)
    return Ndxi


def quad4_shapefuncs_grad_eta(xsi, eta):

    """
    Calculates derivatives of shape functions wrt. eta from quad4_shapefuncs_grad
    Uses the chain rule to find the derivation of N with respect to eta

    Parameters:
        xsi  = Coordinate along the xsi-axis
        eta  = Coordinate along the eta-axis
        
    Returns:
        Ndxi = Array of the derivated shapefunctions with respect to eta for the given quad
    """
    
    # ----- Derivatives of shape functions with respect to eta -----
    Ndeta    = np.zeros(4)
    Ndeta[0] = (1/4) * (1 + xsi)
    Ndeta[1] = (1/4) * (1 - xsi)
    Ndeta[2] = (1/4) * (1 - xsi) * (-1)
    Ndeta[3] = (1/4) * (1 + xsi) * (-1)
    return Ndeta


def quad4e(ex, ey, D, thickness, eq=None):

    """
    Compute the stiffness matrix for a four node membrane element.

    Parameters:
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]

    Returns:
    :return mat Ke: element stiffness matrix [8 x 8]
    :return mat fe: consistent load vector [8 x 1] (if eq!=None)
    """

    t = thickness

    if eq is 0:
        f = np.zeros((2,1))   # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T  # Convert load to 2x1 matrix

    Ke = np.zeros((8,8))        # Create zero-matrix for stiffness-matrix
    fe = np.zeros((8,1))        # Create zero-matrix for distributed load

    numGaussPoints = 2                     # Number of integration points
    gp, gw = gauss_points(numGaussPoints)  # Get integration points and -weight

    for iGauss in range(numGaussPoints):   # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            # Collect shape functions evaluated at xi and eta
            Ndxsi = quad4_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad4_shapefuncs_grad_eta(xsi, eta)
            N1    = quad4_shapefuncs(xsi, eta)

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])    # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])  # Collect gradients of shape function evaluated at xi and eta

            # Calculates the Jacobian-matrix by taking the matrix multiplication of G and H
            J = G @ H
            # Defines the inverse Jacobian-matrix
            invJ = np.linalg.inv(J)  # Inverse of Jacobian
            # Defines the determinant of the Jacobian-matrix
            detJ = np.linalg.det(J)  # Determinant of Jacobian

            # Derivatives of shape functions with respect to x and y
            dN = invJ @ G
            dNdx = dN[0]
            dNdy = dN[1]

            # Strain displacement matrix calculated at position xsi, eta

            # Sets the correct values for strain displacement matrix at current xsi and eta
            B  = np.array([[dNdx[0], 0, dNdx[1], 0, dNdx[2], 0, dNdx[3], 0],
                            [0, dNdy[0], 0, dNdy[1], 0, dNdy[2], 0, dNdy[3]],
                            [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2], dNdy[3], dNdx[3]]])

            # Sets the correct values for displacement interpolation xsi and eta
            zeroMatrix = np.zeros((4))
            N2 = np.zeros((2, 8))
            N2[0] = np.concatenate([N1, zeroMatrix])
            N2[1] = np.concatenate([zeroMatrix, N1])

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T) @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f    * detJ * t * gw[iGauss] * gw[jGauss]
    
    return Ke, fe  # Returns stiffness matrix and nodal force vector

def quad9_shapefuncs(xsi, eta):

    """
    Calculates shape functions evaluated at xsi and eta
    Uses zero-lines to find the shape functions

    Parameters:
        xsi  = Coordinate along the xsi-axis
        eta  = Coordinate along the eta-axis
        
    Returns:
        Ndxi = Array of shapefunctions for the given quad
    """
    
    # ----- Shape functions -----
    N    = np.zeros(9)
    N[0] =   (1/4) * (xsi+xsi**2) * (eta+eta**2)
    N[1] = - (1/4) * (xsi-xsi**2) * (eta+eta**2)
    N[2] =   (1/4) * (xsi-xsi**2) * (eta-eta**2)
    N[3] = - (1/4) * (xsi+xsi**2) * (eta-eta**2)
    N[4] =   (1/2) * (eta+eta**2) * (1-xsi**2)
    N[5] = - (1/2) * (xsi-xsi**2) * (1-eta**2)
    N[6] = - (1/2) * (eta-eta**2) * (1-xsi**2)
    N[7] =   (1/2) * (xsi+xsi**2) * (1+eta**2)
    N[8] =   (1-eta**2) * (1-xsi**2)
    return N


def quad9_shapefuncs_grad_xsi(xsi, eta):

    """
    Calculates derivatives of shape functions wrt. xsi from quad4_shapefuncs_grad
    Uses the chain rule to find the derivation of N with respect to xsi

    Parameters:
        xsi  = Coordinate along the xsi-axis
        eta  = Coordinate along the eta-axis
        
    Returns:
        Ndxi = Array of the derivated shapefunctions with respect to xsi for the given quad
    """
    
    # ----- Derivatives of shape functions with respect to xsi -----
    Ndxi    = np.zeros(9)
    Ndxi[0] =   (1/4) * (eta+eta**2) * (1+2*xsi)
    Ndxi[1] = - (1/4) * (eta+eta**2) * (1-2*xsi)
    Ndxi[2] =   (1/4) * (eta-eta**2) * (1-2*xsi)
    Ndxi[3] = - (1/4) * (eta-eta**2) * (1+2*xsi)
    Ndxi[4] = -  xsi  * (eta+eta**2)
    Ndxi[5] = - (1/2) *  (1-eta**2)  * (1-2*xsi)
    Ndxi[6] =    xsi  * (eta-eta**2)
    Ndxi[7] =   (1/2) *  (1-eta**2)  * (1+2*xsi)
    Ndxi[8] = -  xsi  *  (1-eta**2)  *     2
    return Ndxi


def quad9_shapefuncs_grad_eta(xsi, eta):
    
    """
    Calculates derivatives of shape functions wrt. eta from quad4_shapefuncs_grad
    Uses the chain rule to find the derivation of N with respect to eta

    Parameters:
        xsi  = Coordinate along the xsi-axis
        eta  = Coordinate along the eta-axis
        
    Returns:
        Ndxi = Array of the derivated shapefunctions with respect to eta for the given quad
    """

    # ----- Derivatives of shape functions with respect to eta -----
    Ndeta    = np.zeros(9)
    Ndeta[0] =   (1/4) * (xsi+xsi**2) * (1+2*eta)
    Ndeta[1] = - (1/4) * (xsi-xsi**2) * (1+2*eta)
    Ndeta[2] =   (1/4) * (xsi-xsi**2) * (1-2*eta)
    Ndeta[3] = - (1/4) * (xsi+xsi**2) * (1-2*eta)
    Ndeta[4] =   (1/2) *  (1-xsi**2)  * (1+2*eta)
    Ndeta[5] =    eta  * (xsi-xsi**2)
    Ndeta[6] = - (1/2) *  (1-xsi**2)  * (1-2*eta)
    Ndeta[7] = -  eta  * (xsi+xsi**2)
    Ndeta[8] = -  eta  *  (1-xsi**2)  *     2
    return Ndeta

def quad9e(ex,ey,D,th,eq=None):

    """
    Compute the stiffness matrix for a nine node membrane element.

    Parameters:
    :param list ex: element x coordinates [x1, x2, x3]
    :param list ey: element y coordinates [y1, y2, y3]
    :param list D : 2D constitutive matrix
    :param list th: element thickness
    :param list eq: distributed loads, local directions [bx, by]

    Returns:
    :return mat Ke: element stiffness matrix [18 x 18]
    :return mat fe: consistent load vector [18 x 1] (if eq!=None)
    """

    t = th

    if eq is 0:
        f = np.zeros((2,1))                     # Create zero matrix for load if load is zero
    else:
        f = np.array([eq]).T                    # Convert load to 2x1 matrix

    Ke = np.matrix(np.zeros((18,18)))           # Create zero-matrix for stiffness-matrix
    fe = np.matrix(np.zeros((18,1)))            # Create zero-matrix for distributed load

    numGaussPoints = 2                          # Number of integration points
    gp, gw = gauss_points(numGaussPoints)       # Get integration points and -weight

    for iGauss in range(numGaussPoints):        # Solves for K and fe at all integration points
        for jGauss in range(numGaussPoints):

            xsi = gp[iGauss]
            eta = gp[jGauss]

            # Collect shape functions evaluated at xi and eta
            Ndxsi = quad9_shapefuncs_grad_xsi(xsi, eta)
            Ndeta = quad9_shapefuncs_grad_eta(xsi, eta)
            N1    = quad9_shapefuncs(xsi, eta)

            # Matrix H and G defined according to page 52 of Waløens notes
            H = np.transpose([ex, ey])          # Collect global x- and y coordinates in one matrix
            G = np.array([Ndxsi, Ndeta])        # Collect gradients of shape function evaluated at xi and eta

            # Calculates the Jacobian-matrix by taking the matrix multiplication of G and H
            J = G @ H
            # Defines the inverse Jacobian-matrix
            invJ = np.linalg.inv(J)             # Inverse of Jacobian
            # Defines the determinant of the Jacobian-matrix
            detJ = np.linalg.det(J)             # Determinant of Jacobian

            # Derivatives of shape functions with respect to x and y
            dN = invJ @ G
            dNdx = dN[0]
            dNdy = dN[1]

            # Strain displacement matrix calculated at position xsi, eta

            # Sets the correct values for strain displacement matrix at current xsi and eta
            B  = np.zeros((3, 18))
            j = 0
            g = 0
            h = 0
            for i in range(0, 18):
                if (i % 2) == 0:
                    B[0, i] = dNdx[j]
                    B[1, i] = 0
                    B[2, i] = dNdy[h]
                    j += 1
                else:
                    B[0, i] = 0
                    B[1, i] = dNdy[g]
                    B[2, i] = dNdx[h]
                    g += 1
                    h += 1

            # Sets the correct values for displacement interpolation xsi and eta
            zeroMatrix = np.zeros((9))
            N2 = np.zeros((2, 18))
            N2[0] = np.concatenate([N1, zeroMatrix])
            N2[1] = np.concatenate([zeroMatrix, N1])

            # Evaluates integrand at current integration points and adds to final solution
            Ke += (B.T)  @ D @ B * detJ * t * gw[iGauss] * gw[jGauss]
            fe += (N2.T) @ f     * detJ * t * gw[iGauss] * gw[jGauss]

    # Returns Ke if load is zero
    if eq is None:
        return Ke
    # Returns Ke and Fe
    else:
        return Ke, fe  