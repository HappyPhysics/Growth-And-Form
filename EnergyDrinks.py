# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:24:52 2017

@author: Salem  

This script has a series of methods for calculating the elastic energy of a given mesh. 
"""

import numpy as np

#===========================================================================================================================================
# returns the Edge Matrix given the edge array
#=========================================================================================================================================== 
def makeEdgeMatrix1(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)):
    
    """
    makeEdgeMatrix(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)): 
        gives the edge matrix, which has dimenstions (numOfEdges, numOfVerts).
        For each edge there is a row in the matrix, the row is only nonzero at the positions 
        corresponding to the points connected by that edge, one of them will be 1 the other will be -1.
        When useSpringK is True, each edge will be multiplied by the spring constant which is a convenient thing
        
        Example: verts, edges = squareLattice(2)
            EdgeMat1 = makeEdgeMatrix(edges); EdgeMat1
       Out:  array([[ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
       [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.]])
    
    """
    
    if useSpringK:
        if (springK < 0).all():
            springK = np.ones(numOfEdges)
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
        
    edgeMat = np.zeros((2*numOfEdges, 2*numOfVerts)) #TODO dtype=np.dtype('int32')
    
    for edgeNum, edge in enumerate(edgeArray):
        if not useSpringK:
            edgeMat[2*edgeNum, 2*edge[0]] = 1 
            edgeMat[2*edgeNum + 1, 2*edge[0] + 1] = 1 
            edgeMat[2*edgeNum, 2*edge[1]] = -1 
            edgeMat[2*edgeNum + 1, 2*edge[1] + 1] = -1
        else:
            edgeMat[2*edgeNum, 2*edge[0]] = 1 *springK[edgeNum]
            edgeMat[2*edgeNum + 1, 2*edge[0] + 1] = 1 *springK[edgeNum]
            edgeMat[2*edgeNum, 2*edge[1]] = -1 *springK[edgeNum]
            edgeMat[2*edgeNum + 1, 2*edge[1] + 1] = -1 *springK[edgeNum]
        
    return edgeMat
#===========================================================================================================================================  
    
#===========================================================================================================================================
# returns the Edge Matrix given the edge array
#=========================================================================================================================================== 
def makeEdgeMatrix2(edgeArray, numOfVerts=-1, numOfEdges=-1):
    """
    makeEdgeMatrix2(edgeArray, numOfVerts=-1, numOfEdges=-1, useSpringK = False, springK = -np.ones(1)): 
        gives the edge matrix, which has dimenstions (numOfEdges, 2*numOfEdges).
        For each edge there is a row in the matrix, the row is only nonzero at 2 positions in which 
        it is equal to 1, this is used for adding together the two rows corresponding to the different
        x and y componenets that resulted from multiplying edgeMatrix1 with the vertices. 
        
        Example: verts, edges = squareLattice(2)
            EdgeMat2 = makeEdgeMatrix2(edges); EdgeMat2
            array([[ 1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  1.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
       [ 1.,  0.,  0.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0., -1.]])
    """

    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
        
    edgeMat = np.zeros((numOfEdges, 2*numOfEdges)) #CONSIDER MODIFICATION: dtype=np.dtype('int32')
    
    for edgeNum, edge in enumerate(edgeArray):
        edgeMat[edgeNum, 2*edgeNum] = 1 
        edgeMat[edgeNum , 2*edgeNum + 1] = 1 
       
        
    return edgeMat
#===========================================================================================================================================  
    
#===========================================================================================================================================
# returns the Rigidity Matrix as an array
#===========================================================================================================================================  
def makeRigidityMat(verts, edgeArray=np.array([0]), numOfVerts=-1, numOfEdges=-1, edgeMat1 = np.zeros(1), edgeMat2 = np.zeros(1)):
    """
    makeRigidityMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1,method):
        Takes in the edgeArray then finds Rigidity matrix. The rigidity matrix helps
        to find the bond stretching to linear order in displacement u which has 
        size = 2 numOfVerts. Bond stretchings are equal to 
        dl_e = R_ei * u_i, where i is summed over.
        
        The method parameter desides how the rigidity matrix will be computed. When method = 1
        the edgeMatrices will be used, which is useful when the vertex positions are minimized over. 
        verts should be flattened when this method is used
        
    Example1: 
            sq = squareLattice(2, randomize=False); 
            edgeMat1= makeEdgeMatrix1(sq[1])
            edgeMat2 = makeEdgeMatrix2(sq[1])
            R = makeRigidityMat(sq[0].flatten(), edgeMat1=edgeMat1, edgeMat2=edgeMat2)
            R 
        Out: array([[ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  1.,  1.]])
    
    Example2:
        (verts, edges) = squareLattice(2, randomize=False); 
        edgeMat1 = 
            R = makeRigidityMat(verts, edges) ;R
      array([[ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  1.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.],
       [-1., -1.,  0.,  0.,  0.,  0.,  1.,  1.]])       
    """
    
    if not (edgeMat1==0).all():
        RMat = np.dot(edgeMat1, verts)
        RMat = np.multiply(edgeMat1.transpose(), RMat).transpose()
        return np.dot(edgeMat2, RMat)
    if numOfVerts < 1:
        numOfVerts = len(set(list(edgeArray.flatten())))
    if numOfEdges < 0:
        numOfEdges = edgeArray.size//2
      
    RigidityMat = np.zeros((numOfEdges, 2 * numOfVerts))
    
    for edgeNum, edge in enumerate(edgeArray):
        t = np.zeros((numOfVerts, 2))
        t[edge[1]] = verts[edge[1]] - verts[edge[0]]
        t[edge[0]] = verts[edge[0]] - verts[edge[1]]
        RigidityMat[edgeNum] = t.flatten()
    
    return RigidityMat
#===========================================================================================================================================
   

    
#===========================================================================================================================================
# returns the Rigidity Matrix as an array
#===========================================================================================================================================  
def makeDynamicalMat(edgeArray = np.zeros(1), verts = np.zeros(1), RigidityMat= np.zeros(1), springK= np.zeros(1),  
                     numOfVerts=-1, numOfEdges=-1, negativeK=False):
    """
    makeDynamicalMat(verts, edgeArray, numOfVerts=-1, numOfEdges=-1):
        Takes in the edgeArray then finds dynamical matrix. The dynamical matrix
        help in calculating the potential energy of a displacement u which has 
        size = 2 numOfVerts. The energy is given by E[u] = u.T D u.
        
    Example: 
            (verts, edges) = squareLattice(2, randomize=False); 
             makeDynamicalMat(edgeArray=edges, RigidityMat=R)
        Out: array([[ 2.,  1.,  0.,  0., -1.,  0., -1., -1.],
       [ 1.,  2.,  0., -1.,  0.,  0., -1., -1.],
       [ 0.,  0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0., -1.,  0.,  1.,  0.,  0.,  0.,  0.],
       [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0., -1.],
       [-1., -1., -1.,  0.,  0.,  0.,  2.,  1.],
       [-1., -1.,  0.,  0.,  0., -1.,  1.,  2.]])
    """
    if numOfEdges < 0:
            if(not edgeArray.any()):
                raise NameError("Please either provide the the number of edges or the edge array")
            numOfEdges = edgeArray.size//2
            
            
    if(not RigidityMat.any()):
        print("This is not supposed to be true during minimization because we would be using a rigidity matrix")
        if not verts.any():
            raise NameError("Please either provide the rigidity matrix or the vertices for calculating the dynamical matrix")
        if numOfVerts < 1:
            numOfVerts = len(set(list(edgeArray.flatten())))

        RigidityMat = makeRigidityMat(verts, edgeArray, numOfVerts, numOfEdges) 
    
    if(not springK.any()):
        springK = np.ones(numOfEdges)

    if not negativeK:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK**2)), RigidityMat)
    else:
        dynMat = np.dot(np.dot(RigidityMat.transpose(), np.diag(springK)), RigidityMat)
    return dynMat
#================================================================================================================================================