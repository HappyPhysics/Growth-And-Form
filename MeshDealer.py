# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:53:53 2017

@author: Salem

This script has methods to define and generate lattices amd edges. It also has methods to operate on lattice data. 

Methods: 
    makeCylVerts(num_theta_verts , num_z_verts, radius = 0.20, length = 1.0, capSize = 0.1, is_capped = True):
        returns the vertices of a cylinder. with respect to the origin.
        
    make_cyl_edges(num_z_verts, num_theta_verts, is_capped = True):
        returns an array of pairs of indices of the cylinder.
     
    neibs_from_edges(edge_list, num_of_verts = -1):
        Returns a Neighbor list and a map between the edges list and Neighbor list.
        
    DihedralVertices(neib_list, edge_list):
        finds the two vertices oppisite to each edge 
        

"""
import numpy as np


# make lattice. 
# make edges corresponding to stretching and bending separatly.
# make the rigidity and dynamical matrix. 
# minimize energy subject to constraints.  
# test on know lattice results



#=========================================================================================================================================================
#returns the vertices of a cylinder. with respect to the origin
#=========================================================================================================================================================
def make_cyl_verts(num_theta_verts , num_z_verts, radius = 0.20, length = 1.0, capSize = 0.1, is_capped = True):
    """
    Returns a vertex array of 3D positions on a cylinder (hexagonally arranged triangulation). 
    
    num_theta_verts: the number of vertices in the theta direction
    num_z_verts: number of vertices in the z direction
    length: length of the cylinder.
    radius: radius of cylinder.
    is_capped: whether we will add  caps at the end of the cylinder to close it. These are just two points added to the top.
    capSize: the relative size of the cap region to the length
    
    Example:  makeCylVerts(2, 2)
    Out[16]: 
        array([[  0.00000000e+00,   0.00000000e+00,  -6.00000000e-01],
       [  2.00000000e-01,   0.00000000e+00,  -5.00000000e-01],
       [ -2.00000000e-01,   2.44929360e-17,  -5.00000000e-01],
       [  1.22464680e-17,   2.00000000e-01,   5.00000000e-01],
       [ -3.67394040e-17,  -2.00000000e-01,   5.00000000e-01],
       [  0.00000000e+00,   0.00000000e+00,   6.00000000e-01]])
    """
    
    zStep = 1.0/(num_z_verts - 1.0) # z goes from 0 to 1
    
    thetaStep = (2 * np.pi)/ (num_theta_verts) # theta goes periodically from 0 to 2 pi
    
    # make the mesh grid. z here will run the z-value for all mesh points. 
    # each row in z has the same element because that's what the height would be for a row of points
    [z, theta]= np.mgrid[0:1 + zStep/2:zStep, 0:2*np.pi:thetaStep]
    
    [zIndx, thetaIndx]= np.mgrid[0:num_z_verts:1, 0:num_theta_verts:1] #mesh grid of indices
    
    zIndx = zIndx.reshape(z.size, 1)
    z = z.reshape(z.size, 1)
    
    theta = theta.reshape(theta.size, 1)
    
    mask = [np.mod(zIndx, 2) != 0] # picks out odd rows
    
    theta[mask] += 0.5*thetaStep   # shift the odd rows, this helps us make a hexagonal matrix
    
    vertices = np.hstack((radius*np.cos(theta), radius*np.sin(theta), length * (z - 0.5))) # stack x, y, z components
    
    if (is_capped):
    # capSize gives the length of the cap area relative to the legth of the cylinder  
        cap1 = np.array([0, 0, -length * (capSize + 0.5)])
        cap2 = np.array([0, 0, length * (capSize + 0.5) ])
        vertices = np.vstack((cap1, vertices, cap2))
        
        
    return vertices
#=========================================================================================================================================================


#=========================================================================================================================================================
# returns an array of pairs of indices of the cylinder
#=========================================================================================================================================================
def make_cyl_edges(num_z_verts, num_theta_verts, is_capped = True):
    """
    returns an array with each edge listed as a pair of indices. The indices correspond to the positions of the points in the vertex array. 
    So the position in a vertex array of a point is assumed and must be preserved.
    
    num_z_verts: The number of points in the z-direction (roughly speaking at a fixed theta).
    
    num_theta_verts: number of point in the theta direction at fixed z. This is assumed to be bigger than two. Otherwise, we will have repeated edges.
    For example 0-1, 1-0 (because of periodicity).
    
    is_capped: whether we will add caps to the cylinder
    
    
   Example 2: make_cyl_edges(2, 3, is_capped=False) 
    array([[0, 2],
       [0, 3],
       [0, 5],
       [1, 0],
       [1, 4],
       [1, 3],
       [2, 1],
       [2, 5],
       [2, 4],
       [3, 5],
       [4, 3],
       [5, 4]])
    
    """
    
    if(num_theta_verts < 3):
        raise NameError("The number of vertices in the theta direction should be greater than 2.")
    
    edge_list = [] #list of pairs of vertex indices
    
 
    #number of indices assuming is_capped. 
    num_of_verts = num_z_verts*num_theta_verts + 2  
    #give the indices of all the vertices excluding the first and last cap vertices
    indices = np.arange(num_of_verts - 2, dtype=int) + 1
    #if is_capped is False these indices will be modified at the end of the method to give the correct values 

    
    for indx in np.nditer(indices):   
        
    #nZ, nT are the theta and z indices of the vertex indx
        nZ = (indx - 1) // num_theta_verts
        nT = np.mod(indx - 1, num_theta_verts)
        
    # if nT != 0, because nT = 0 are adjacent to nT = num_theta_verts - 1 (periodicity)
        if(nT):
            
      #Horizontal edge between indices except the nT = 0 one      
            edge_list.append([indx, indx - 1])
            
    # if nZ is even, these are shifted in theta compared to odd nZ
            if(not np.mod(nZ, 2)):
                
                if (nZ != 0):
               # the two edge pointing below from even rows
                    edge_list.append([indx, indx - num_theta_verts])        
                    edge_list.append([indx, indx - num_theta_verts - 1])
                    
                if(nZ != num_z_verts - 1):
                # the two edge pointing above from even rows   
                    edge_list.append([indx, indx + num_theta_verts]) 
                    edge_list.append([indx, indx + num_theta_verts - 1])
                    
                    
     # special case because of the periodicity
        elif(nT == 0):
            
    #Horizontal edge, indices are not 1 apart because of periodicity  
            edge_list.append([indx, indx + num_theta_verts - 1]) # assumes that num_theta_verts > 2, other wise this is a repeat
       
     #even case with nT = 0   
            if(not np.mod(nZ, 2)):
                 if (nZ != 0):
                     
     #the two edge pointing below from even rows and nT = 0  
                     edge_list.append([indx, indx - num_theta_verts])
                     edge_list.append([indx, indx - 1])
                     
                 if(nZ < num_z_verts - 1):
    #the two edge pointing above from even rows and nT = 0                       
                     edge_list.append([indx, indx + num_theta_verts])
                     edge_list.append([indx, indx +2* num_theta_verts - 1])
                     
    if(is_capped):              
        
        #add the top cap edges to the edge list
        for i in np.nditer(np.arange(num_theta_verts)):   
            edge_list.append([num_of_verts - 1, num_of_verts - (2 + i)])
            
            #add the bottom cap edges to the edge list
        for j in np.nditer(np.arange(num_theta_verts)):
            edge_list.append([0, j + 1])
                
    else:
        edge_list = np.array(edge_list) - 1 #brings the indices back to what they should be if the first cap is not there
    
    
    return (np.array(edge_list)) 
#=========================================================================================================================================================


#=================================================================================
#flatten a list 
#=================================================================================
def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if type(item) == type([]):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis
#=================================================================================


#===============================================================================================================================================
# Returns a Neighbor list and a map between the edges list and Neighbor list.
#===============================================================================================================================================
def neibs_from_edges(edge_list, num_of_verts = -1):
    """
    a neighbor list is a list of (num_of_verts) lists each contain the indices of the neighbors of the corresponding verts.
    The vertex being refered to is assumed to be implied by the position of it's neighbors in the neib_list.
    
    We also return a map between the edge_list and the neighbor list.
    
             edges = make_cyl_edges(2,3,is_capped=False)
    Example: neibs_from_edges(edges)
    [[2, 3, 5, 1],
  [0, 4, 3, 2],
  [0, 1, 5, 4],
  [0, 1, 5, 4],
  [1, 2, 3, 5],
  [0, 2, 3, 4]],
    
"""

    if (num_of_verts < 1):
        num_of_verts = len(set(list(edge_list.flatten())))
    #num_of_edges = edge_list[:, 0].size
    
    #for each row neib_list gives the neighbor indices of the vertex correspoding to the row index                       
    neib_list = [[]]*num_of_verts
    
    #For each index in a row, neibs-to-edges will point to the correct edge index  in the edge array               
    neibs_to_edges = [[]]*num_of_verts 
     
    
    #loop over the vertices      
    for Vindx in np.nditer(np.arange(num_of_verts)):
        
        #for each vertex list it's neighbors by finding the edges it appears in
        for Eindx, edge in enumerate(edge_list): 
            
            #when you find it in one index of the edge, 
             if edge[0] == Vindx:
            #add the second index to the neib_list     
                neib_list[Vindx] = [neib_list[Vindx], edge[1]]  #doing it this way is important for the nesting to come out right
            #make the map too
                neibs_to_edges[Vindx] = [neibs_to_edges[Vindx], Eindx]
                
                
            #when you find it in one index of the edge, 
             elif edge[1] == Vindx:
            #add the second index to the neib_list     
                neib_list[Vindx] = [neib_list[Vindx], edge[0]]
            #make the map too
                neibs_to_edges[Vindx] = [neibs_to_edges[Vindx], Eindx]
                
        neib_list[Vindx] = flatten(neib_list[Vindx]) #flatten the rows to get rid of extra nesting
        neibs_to_edges[Vindx] = flatten(neibs_to_edges[Vindx])
     
                
    return (neib_list, neibs_to_edges)
#===============================================================================================================================================


#=================================================================================
#find the two vertices oppisite to each edge
#=================================================================================
def DihedralVertices(neib_list, edge_list):
    ''' Calculates the dihedral vertices for the triangulation from the neighbor list.
        return (numEdges, 2) array containing the indices of the two 
        vertices corresponding to the triangles that include the edge. In other words,
        return the two vertices opposite to the edge. This is useful for implementing 
        bending rigidity
        
        Neibs can be and array or list (I think)
        '''
        
    num_of_edges = edge_list[:, 0].size
    
    
    dihedral_verts = np.zeros((num_of_edges, 2), dtype=int)
    
    #loop over all the edges
    for edge_index in np.nditer(np.arange(num_of_edges)):
        
        #find the two triangles intersecting at the edge by finding common neighbors of the two edge vertices
        dihedral_verts[edge_index] = np.intersect1d(neib_list[edge_list[edge_index,0]], neib_list[edge_list[edge_index,1]]) 
    
    return dihedral_verts
#=================================================================================    


















