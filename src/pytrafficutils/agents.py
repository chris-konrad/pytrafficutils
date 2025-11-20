# -*- coding: utf-8 -*-
"""
Created on Fr Jul 24, 2025

pytrafficutils.agents

Traffic Agents for surrogate safety assessment.

@author: Christoph M. Konrad
"""

import numpy as np

from pytrafficutils.footprints import RectangularFootprint, DiamondFootprint



class Agent:

    STATE_MAP = {}
    N_STATES = 0  


    def __init__(self, t, name="agent"):

        self.name = name
        self.footprint = None
        self._traj = None

        self.t = self._check_t(t)


    def _check_t(self, t):

        t = np.asarray(t, dtype='float')
        
        if t.size != 3:
            raise ValueError(f"t must be size 3 with t = [t_begin, t_end, t_s]! Instead it was size {t.size}")
        
        if t[1] <= t[0]:
            raise ValueError(f"t_end must be larger then t_begin! Instead t = [t_begin, t_end, t_s] was {t}")
        
        if t[2] >= t[1] - t[0]:
            raise ValueError(f"The sample t_s in [t_begin, t_end, t_s] must be smaller then t_end - t_begin")
        
        return t


    def get_state(self, t):
        """ Return the state of the agent at time t.

        PROTOTYPE. NOT IMPLEMENTED FOR BASE CLASS
        """
        raise NotImplementedError(f'get_state() not implemented for class {type(self).__name__()}')


    def extrapolate_from_t(self, t, T, t_s = 0.01):
        """ Given an index of a state s = [x, y, psi, v], extrapolate over T seconds using sample time t_s. 

        PROTOTYPE. NOT IMPLEMENTED FOR BASE CLASS
        """
        raise NotImplementedError(f'extrapolate_from_t() not implemented for class {type(self).__name__()}')



class StationaryAgent(Agent):
    """ An agent fixed in space
    """

    STATE_MAP = dict(p_x=0, p_y=1, psi=2)
    N_STATES = 3


    def __init__(self, t, s, footprint, name='stationary_agent'):

        super().__init__(t, name=name)

        self.footprint = footprint
        self._traj = self._check_traj(s)
        self.n_samples = 1


    def _check_traj(self, traj):
        """ Verify that the trajectory is shaped (1, n_states)"""

        traj = np.asarray(traj)
        traj = traj.reshape(1,-1)

        if not (traj.shape[1] == self.N_STATES):
            raise ValueError(f"The trajectory must be shaped (1, {self.N_STATES}) or ({self.N_STATES},). Instead it was {traj.shape}.")
        
        return traj


    def get_state(self,t):
        """ Return the state of the agent at time t.

        Returns the fixed state of the stationary agent

        Parameters
        ----------
        t : float
            Time of the requested state.

        Returns
        -------
        s : np.ndarray
            State shaped [x, y, psi].
        """

        return self._traj.flatten()


    def extrapolate_from_t(self, t, T, t_s = 0.01):
        """ Given a time t, extrapolate the state at t over T seconds using sample time t_s. 

        Repeats the fixed state of the stationary agent n_extrapolation_samples times. 

        Parameters 
        ----------

        t : float
            Ignored for the stationary agent. Kept for API compatibility.
        T : float
            Extrapolation time T in s.
        t_s : float, optional
            Sample time t_s in s. Default is 0.01 s

        Returns
        -------

        extrapolated_traj : np.ndarray
            The extrapolated trajectory shaped (n_extrapolation_samples, 2), where each extrapolation sample s[i,:] is [x_i, y_i, psi_i].
        """

        n_ext = int(round(T/t_s))

        extrapolated_traj = np.tile(self._traj[:,:3], (n_ext, 1))

        return extrapolated_traj



class MovingAgent(Agent):
    """ An agent moving along a given trajectory.
    """

    STATE_MAP = dict(p_x=0, p_y=1, psi=2, v=3)
    N_STATES = 4

    def __init__(self, t, traj, footprint, name='moving_agent'):

        super().__init__(t, name=name)

        self.footprint = footprint
        self._traj = self._check_traj(traj)
        self.n_samples = self._traj.shape[0]


    def _check_traj(self, traj):
        """ Verify that the trajectory is shaped (n_samples, n_states)"""

        traj = np.asarray(traj)

        if not ((traj.shape[1] == self.N_STATES) and (traj.ndim == 2)):
            raise ValueError(f"The trajectory must be shaped (n_samples, {self.N_STATES}) with states {list(self.STATE_MAP.keys())}. "
                             f"Instead it was {traj.shape}.")
        
        return traj


    def _find_state_index_at_time(self, t):
        """ Return the state index closest to time t"""

        if t >= self.t[0] and t < self.t[1]:
            idx = int(round((t-self.t[0])/self.t[2]))
        else:
            idx = None

        return idx


    def get_state(self, t):
        """ Return the state sample of the agent closest to time t.

        Parameters
        ----------
        t : float
            Time of the requested state.

        Returns
        -------
        s : np.ndarray
            State shaped [x, y, psi, v].

        Warning
        -------

        Returns the state sample closes to the requested time. 
        
        TODO: Implement interpolation between state samples.
        """

        idx = self._find_state_index_at_time(t)
        s = self._traj[idx, :].copy()

        return s


    def extrapolate_from_t(self, t, T, t_s = 0.01):
        """ Given a time t, extrapolate the state at t over T seconds using sample time t_s. 

        Assumes constant heading and constant speed. 

        Parameters 
        ----------

        index : array-like
            Index of the state in the trajectory of this agent such that s = agent.traj[index, :]
        T : float
            Extrapolation time T in s.
        t_s : float, optional
            Sample time t_s in s. Default is 0.01 s

        Returns
        -------

        extrapolated_traj : np.ndarray
            The extrapolated trajectory from state s shaped (n_extrapolation_samples, 2), where each extrapolation sample s[i,:] is [x_i, y_i, psi_i]. 
        """
        s = self.get_state(t)

        # extrapolation times
        n_ext = int(round(T/t_s))
        t = np.arange(n_ext) * t_s
        t = t.reshape(n_ext, 1)

        v = s[3] * np.array([[np.cos(s[2]), np.sin(s[2])]])
        
        extrapolated_traj = s[0:2].reshape(1,2) + t @ v
        extrapolated_traj = np.hstack((extrapolated_traj, s[2] * np.ones((n_ext, 1))))

        return extrapolated_traj


    def extrapolate_all_states(self, T, t_s=0.01, range=None):
        """
        Extrapolate all states in the trajectory over time T using sample time t_s.

        Assumes constant heading and constant speed for each state.

        Parameters
        ----------
        T : float
            Extrapolation time in seconds.
        t_s : float, optional
            Sample time in seconds. Default is 0.01 s.
        range : array-like, optional
            Index range of this agent's state trajctory to be extrapolated given as [idx_begin, idx_end], 
            [idx_begin, idx_end, step] or None. If None, all states of the trajectory are extrapolated. 
            Default is None.

        Returns
        -------
        extrapolated_trajectories : np.ndarray
            Extrapolated trajectories for all states, shaped (n_samples, n_ext, 2),
            where each trajectory extrapolated_trajectories[i, :, :] is a sequence of [x, y] positions.
        """

        # select trajectory range to be extrapolated
        if range is None:
            traj = self._traj
        else:
            range = np.asarray(range, dtype=int).flatten()
            
            if range.size < 2 or range.size > 3:
                raise ValueError(f"'range' must be [idx_begin, idx_end] or [idx_begin, idx_end, step] but it was {range}")
            if range.size == 2:
                range = [range[0], range[1], 1]

            traj = self._traj[range[0]:range[1]:range[2],:]

        # extrapolation times
        n_ext = int(round(T/t_s))
        t = np.arange(n_ext) * t_s
        t = t.reshape(n_ext, 1)

        v = traj[:, 3] * np.stack([np.sin(traj[:, 2]), np.cos(traj[:, 2])], axis=1)

        extrapolated_trajectories = traj[:, 0:2][:, np.newaxis, :] + t[np.newaxis, :, :] * v[:, np.newaxis, :]

        return extrapolated_trajectories


class RectObstacle(StationaryAgent):
    """ A stationary rectangular obstacle."""

    def __init__(self, t, s, length=1, width=1, anchor=[0,0], name='obstacle'):
        """
        Parameters
        ----------

        t : array-like
            Active time range of the obstacle given as [t_begin, t_end, t_s]. Note that t_end is assumed to be after the time of the last sample, not including. 
        s : array_like
            The fixed state of the obstale given as [x, y, psi].
        length : float, optional
            The length of the obstacle in m. The default is 4.4 m.
        width : float
            The width of the obstacle in m. The default is 1.8 m.
        """

        footprint = RectangularFootprint(length, width, anchor=anchor)

        super().__init__(t, s, footprint, name=name)


class Bicycle(MovingAgent):
    """ A moving agent with a diamond footprint representing a bicycle.
    """

    def __init__(self, t, traj, length=1.8, width=0.8, wheeldiameter=0.72, name='cyclist'):
        """
        Parameters
        ----------

        t : array-like
            Active time range of the bicycle given as [t_begin, t_end, t_s]. Note that t_end is assumed to be after the time of the last sample, not including. 
        traj : array_like
            The trajectory array of the bicycle shaped (n_samples, n_states) where each state is [x, y, psi, v].
        length : float, optional
            The length of the bike in m.. The default is 1.8.
        widht : float
            The width of the bike in m. The default is 0.8.
        wheeldiameter : float
            The diameter of the wheel of the bike in m. The default is 0.72
        """
        footprint = DiamondFootprint(length, width, anchor = [- length/2 + wheeldiameter/2, 0])

        super().__init__(t, traj, footprint, name='cyclist')


class Car(MovingAgent):
    """ A moving agent with a rectangular footprint representing a car."""

    def __init__(self, t, traj, length=4.4, width=1.8, name='car'):
        """
        Parameters
        ----------

        t : array-like
            Active time range of the car given as [t_begin, t_end, t_s]. Note that t_end is assumed to be after the time of the last sample, not including. 
        traj : array_like
            The trajectory array of the car shaped (n_samples, n_states) where each state is [x, y, psi, v].
        length : float, optional
            The length of the car in m. The default is 4.4 m.
        widht : float
            The width of the car in m. The default is 1.8 m.
        """

        footprint = RectangularFootprint(length, width)

        super().__init__(t, traj, footprint, name=name)