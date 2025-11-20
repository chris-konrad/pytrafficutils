# -*- coding: utf-8 -*-
"""
Created on Fr Jul 24, 2025

pytrafficutils.footprint

Footprints for different traffic agents. 

@author: Christoph M. Konrad
"""

import numpy as np

from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.plotting import plot_polygon


class Footprint():

    def __init__(self, xy):

        xy = self._check_xy(xy)

        self.max_radial_extent = np.max(np.linalg.norm(xy, axis=1))
        self._poly_prototype = Polygon(xy)


    def _check_xy(self, xy):

        xy = np.asarray(xy)

        if xy.ndim != 2:
            raise ValueError(f"'xy' must be shape (n_nodes, 2)! Instead it was {xy.shape}!")
        if xy.shape[1] != 2:
            raise ValueError(f"'xy' must be shape (n_nodes, 2)! Instead it was {xy.shape}!")
        
        return xy
    

    def _check_anchor(self, anchor):
        
        anchor = np.asarray(anchor).reshape([1,2])

        if anchor.size != 2:
            raise ValueError(f"'anchor' must be [[x, y]]. Instead it was {anchor}")
        
        return anchor
    

    def get_polygons_at(self, states):
        """ Return a Polygon or list of Polygons representing the footprint at the requested state(s).

        Parameters
        ----------

        states : array-like
            Array of states shaped (n_samples, 3) or (3,), where each state states[i,:] = [x_i, y_i, psi_i]

        Returns
        -------
        footprints : shapely.geometry.Polygon or list of shapely.geometry.Polygon
            Returns the polygon at the requested state(s). states.ndim > 1, a list is returned.

        """
        states = np.array(states)
        if states.ndim == 1:
            return_single = True
            states = states.reshape(1,3)
        else:
            return_single = False

        states[:,2] = np.rad2deg(states[:,2])
        
        footprints = []
        for i in range(states.shape[0]):
            center = states[i,:2]
            orient = states[i,2]

            poly = rotate(self._poly_prototype, orient, origin=(0, 0))
            poly = translate(poly, xoff=center[0], yoff=center[1])

            footprints.append(poly)

        if return_single:
            return footprints[0]
        else:
            return footprints

    def plot_at(self, states, ax, **kwargs_polygon):
        """ Plot the footprint(s) at the requested state(s).

        Parameters
        ----------

        states : array-like
            Array of states shaped (n_samples, 3) or (3,), where each state states[i,:] = [x_i, y_i, psi_i]
        """

        kwargs_polygon['ax'] = ax

        footprints = self.get_polygons_at(states)
        if not isinstance(footprints, list):
            footprints = [footprints]
        
        for fp in footprints:
            plot_polygon(fp, **kwargs_polygon)


class RectangularFootprint(Footprint):

    def __init__(self, width, height, anchor=[0,0]):
        """ Make a rectangular footprint, where width height and anchor are defined as follows:
                            
                                       ^ y
                                       |
            (-w/2, h/2)----------------|----------------(w/2, h/2)
                 | anchor=(x0, y0)     |                     |
         -------------x--------------------------------------------> x                       
                 |                     |                     |
            (-w/2,-h/2)----------------|----------------(w/2,-h/2)       
                                       |        

        Parameters
        ----------
        width : float
            Width of the rectangle (x-direction)
        height : float
            Height of the rectangle (y-direction)
        anchor : array-like
            Anchor location relative to the center of the rectangle. 

        """

        xy = self._make_xy(width, height, anchor)
        super().__init__(xy)
    
    
    def _make_xy(self, width, height, anchor):
        """ Make the polygon nodes"""

        anchor = self._check_anchor(anchor)
    
        xy = np.array([[width/2, height/2],
                       [-width/2, height/2],
                       [-width/2, -height/2],
                       [width/2, -height/2]])

        xy = xy - anchor

        return xy
    


class DiamondFootprint(Footprint):
    
    def __init__(self, width, height, anchor=[0,0]):
        """ Make a diamond-shaped footprint, where width height and anchor are defined as follows:

        Parameters
        ----------
        width : float
            Width of the diamond (x-direction)
        height : float
            Height of the diamond (y-direction)
        anchor : array-like
            Anchor location relative to the center of the diamond. 

        """

        xy = self._make_xy(width, height, anchor)
        super().__init__(xy)
    
    
    def _make_xy(self, width, height, anchor):
        """ Make the polygon nodes"""

        anchor = self._check_anchor(anchor)
    
        xy = np.array([[width/2, 0],
                       [0, height/2],
                       [-width/2, 0],
                       [0, -height/2]])

        xy = xy - anchor

        return xy