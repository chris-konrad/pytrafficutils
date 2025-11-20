# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:00:26 2023

pytrafficutils.ssm

Surrogate safety measures for traffic analysis 

@author: Christoph M. Konrad
"""

import numpy as np
import pandas as pd

from pytrafficutils.agents import Agent


def pet(t1, t2, ts):   
    '''Calculates the post encroachment time (PET) between the trajectories of
    two encroaching road users.
    
    The PET was introduced by Allen et al. (1978) and is a commonly used for
    surrogate safety assessment in traffic analysis and simulation. This 
    function measures the time between the first road user and the second user
    approximately occupying the same point in space. It does not perform 
    interpolation of the trajectories and does not consider the spatial extent
    of road users. 
    
    Assumes that t1 and t2 have at least one encroachment. If there is no 
    encroachmemt, the function returns the smallest distance in space.
    Finds the smallest encroachment time if there are multiple encroachments.

    Parameters
    ----------
    t1 : array
        Trajectory of the first road user.
    t2 : array
        Trajectory of the second road user.
    ts : float
        Sampling time.

    Returns
    -------
    float
        Post encroachment time.
    array
        Position of the encroachment in t1.
    TYPE
        Position of the encroachment in t2.

    References
    ----------
    Allen, B. L., Shin, B. T., & Cooper, P. J. (1978). Analysis of traffic 
    conflicts and collisions. Transportation Research Record, 667(1), 67–74.
    '''   
    dmin = 100000000.
    imint1 = 0
    imint2 = 0
    for i1 in range(t1.shape[1]):
        di = np.sqrt((t1[0,i1]-t2[0,:])**2+(t1[1,i1]-t2[1,:])**2)
        i2 = np.argmin(di)
        if di[i2] < dmin:
            dmin = di[i2]
            imint1 = i1
            imint2 = i2
    return abs((imint1-imint2)*ts), t1[:,imint1], t2[:,imint2]


class SurrogateSafetyMetricProcessor:

    METRICS = ['TTC']

    def __init__(self, agents, t_s=0.01, verbose=True):

        # agents
        self.agents = self._check_agents(agents)
        self.n_agents = len(self.agents)
        
        # time
        self.t_s = t_s
        self.t_ranges, self.t_begin, self.t_end, self.n_steps = self._get_time_ranges()

        # preferences
        self.verbose=verbose


    def _check_agents(self, agents):
        """ Verify that agents is an array-like of agents and cast to list."""

        agents = np.asarray(agents)

        for i, a in enumerate(agents):
            if not isinstance(a, Agent):
                raise TypeError(f"Agents must be a list of {Agent.__name__} objects. Found {type(a).__name__} at index {i} in 'agents'! ")
        return agents


    def _get_time_ranges(self):

        t_ranges = np.zeros((self.n_agents, 2))

        for i, a in enumerate(self.agents):
            t_ranges[i,:] = a.t[:2]

        t_begin = t_ranges.min()
        t_end = t_ranges.max()+self.t_s

        steps = int(round((t_end - t_begin)/self.t_s))

        return t_ranges, t_begin, t_end, steps     


    def _calc_distances(trajs):
        """ Given the trajectories (n_trajs, n_samples, 2), calculate the euclidian distances of each trajectory sample to each other.

        Parameters:
        -----------
        trajs : array-like
            Trajectory array shaped (n_trajs, n_samples, n_states)

        Returns:
        --------
        dist : np.ndarray
            Array of shape (n_trajs, n_trajs, n_samples), where dist[i,j,k] is the distance between rajectory i and trajectory j at sample k 
        """
        
        diff = trajs[:, np.newaxis, :, :2] - trajs[np.newaxis, :, :, :2]  # shape = (n_trajs, n_trajs, n_samples, 2)
        dist = np.linalg.norm(diff, axis=-1)  
        
        return dist


    def eval_TTC(self, T_extrapolation, t_s_extrapolation=0.01, dt_min_interaction = 1, conflict_thresholds=[0.5, 1.0, 1.5, 2.0]):

        times = self.t_begin + np.arange(self.n_steps) * self.t_s

        ttc = np.nan * np.zeros((self.n_agents, self.n_agents, self.n_steps))

        for i, t in enumerate(times):

            #find active agents at time t
            is_active = np.logical_and(t >= self.t_ranges[:,0], t < self.t_ranges[:,1])
            active_agents = self.agents[is_active]
            active_agent_indices = np.argwhere(is_active).flatten()

            #prepare ttc_i array
            ttc_i = np.nan * np.zeros((active_agents.size, active_agents.size, 1))

            n_ext = int(round(T_extrapolation/t_s_extrapolation))
            extrapolations = np.zeros((active_agents.size, n_ext, 3))
            minimum_distances = np.zeros(active_agents.size)

            for j, a in enumerate(active_agents):

                #get trajectory extrapolations shaped (n_agents, n_ext, 3)
                extrapolations[j,:,:] = a.extrapolate_from_t(t, T_extrapolation, t_s_extrapolation)

                #get the minimum distance based on the radius of the footprint of the agent
                minimum_distances[j] = a.footprint.max_radial_extent

            # get distance matrix shaped (n_agents, n_agents, n_ext)
            distances = SurrogateSafetyMetricProcessor._calc_distances(extrapolations)

            # broadcast minimum distance vector to matrix 
            minimum_distances = minimum_distances[np.newaxis,:] + minimum_distances[:,np.newaxis]
            minimum_distances = minimum_distances[:,:,np.newaxis]

            # find samples where agents are within collision range
            within_collision_range = distances < minimum_distances

            # get footprints of agents in collision range
            agent_collision_range_footprints = []

            for j, a in enumerate(active_agents):
                other = np.arange(within_collision_range.shape[1]) != j

                # index mask where a is in collision range with any other agent
                a_within_collision_range = within_collision_range[j , other, :].flatten()

                if np.any(a_within_collision_range):
                    # positions where a is in collision range with any other agent
                    collision_range_positions = extrapolations[j, a_within_collision_range, :]

                    # get the footprints at the collision range positions, make dict and store in vector
                    collision_range_footprints = a.footprint.get_polygons_at(collision_range_positions)
                    footprint_dict = dict(zip(np.argwhere(a_within_collision_range).flatten(), 
                                            collision_range_footprints))
                else:
                    footprint_dict = {}
                
                agent_collision_range_footprints.append(footprint_dict)

            # check footprints of agents in collision range for overlap
            for a_i in range(active_agents.size):
                for a_j in range(active_agents.size):
                    if a_j > a_i:
                        ij_within_collision_range = within_collision_range[a_i,a_j,:]

                        if np.any(ij_within_collision_range):
                            idx_collision_range = np.argwhere(ij_within_collision_range).flatten()

                            for idx in idx_collision_range:
                                footprint_i = agent_collision_range_footprints[a_i][idx]
                                footprint_j = agent_collision_range_footprints[a_j][idx]
                                collision = footprint_i.intersects(footprint_j)

                                # in case of collision, the a ttc for this pair of agents at this time is found! 
                                if collision:
                                    ttc_ai_aj = idx * t_s_extrapolation
                                    ttc_i[a_i, a_j] = ttc_ai_aj
                                    ttc_i[a_j, a_i] = ttc_ai_aj
                                    break

            # after potential collisions have been checked, store the ttc_i array for this timestep
            ttc[np.ix_(active_agent_indices, active_agent_indices, [i])] = ttc_i

        
        # convert ttc array to interaction table 
        partner_1_id = []
        partner_2_id = []
        partner_1_name = []
        partner_2_name = []
        t_begin = []
        t_end = []
        ttc_min = []
        ttc_max = []
        ttc_mean = []
        t_ttc_min = []
        t_ttc_max = []
        x1_ttc_min = []
        y1_ttc_min = []
        psi1_ttc_min = []
        x2_ttc_min = []
        y2_ttc_min = []  
        psi2_ttc_min = []      
        conflicts = [[] for c in conflict_thresholds]

        for a_i in range(self.n_agents):
            for a_j in range(self.n_agents):
                if a_j > a_i:

                    ttc_exists = np.isfinite(ttc[a_i, a_j,:])
                    if np.any(ttc_exists):
                        
                        #ttc_ij = ttc[a_i, a_j,:]

                        interaction_begin = np.argwhere(np.diff(ttc_exists.astype(int))>0).flatten()+1
                        interaction_end = np.argwhere(np.diff(ttc_exists.astype(int))<0).flatten()+1

                        # treat edge cases, where interaction begins/ends before/after the tested time period.
                        if interaction_begin.size < 1:
                            interaction_begin = np.array([0])
                        if interaction_end.size < 1:
                            interaction_end = np.array([ttc_exists.size])
                        if interaction_end[0] < interaction_begin[0]:
                            interaction_begin = np.r_[0, interaction_begin]
                        if interaction_begin[-1] > interaction_end[-1]:
                            interaction_end = np.r_[interaction_end, ttc_exists.size-1]

                        assert interaction_begin.size == interaction_end.size

                        # fuse interactions if they are close to each other
                        t_interaction_begin = self.t_begin + interaction_begin * self.t_s
                        t_interaction_end = self.t_begin + interaction_end * self.t_s

                        interaction_mask_begin = np.ones_like(interaction_begin, dtype=bool)
                        interaction_mask_end = np.ones_like(interaction_begin, dtype=bool)
                        for m in range(interaction_begin.size-1):
                            t0 = t_interaction_end[m]
                            t1 = t_interaction_begin[m+1]

                            gap_has_min_length = t1 - t0 >= dt_min_interaction
                            if not gap_has_min_length:
                                interaction_mask_begin[m+1] = gap_has_min_length
                                interaction_mask_end[m] = gap_has_min_length
                        
                        t_interaction_begin = t_interaction_begin[interaction_mask_begin]
                        t_interaction_end = t_interaction_end[interaction_mask_end]

                        interaction_begin = interaction_begin[interaction_mask_begin]
                        interaction_end = interaction_end[interaction_mask_end]

                        # append to interaction lists
                        for m in range(interaction_begin.size):
                            ttc_m = ttc[a_i, a_j,interaction_begin[m]:interaction_end[m]]

                            partner_1_id.append(a_i)
                            partner_2_id.append(a_j)
                            partner_1_name.append(self.agents[a_i].name)
                            partner_2_name.append(self.agents[a_j].name)
                            t_begin.append(t_interaction_begin[m])
                            t_end.append(t_interaction_end[m])
                            ttc_min.append(np.nanmin(ttc_m))
                            ttc_max.append(np.nanmax(ttc_m))
                            ttc_mean.append(np.nanmean(ttc_m))
                            t_ttc_min.append(np.argwhere(ttc_min[-1]==ttc_m).flatten()[0]*self.t_s)
                            t_ttc_max.append(np.argwhere(ttc_max[-1]==ttc_m).flatten()[0]*self.t_s)

                            s1 = self.agents[a_i].get_state(t_ttc_min[-1])
                            x1_ttc_min.append(s1[0])
                            y1_ttc_min.append(s1[1])
                            psi1_ttc_min.append(s1[2])

                            s2 = self.agents[a_j].get_state(t_ttc_min[-1])
                            x2_ttc_min.append(s2[0])
                            y2_ttc_min.append(s2[1])
                            psi2_ttc_min.append(s2[2])
                            for clist, th in zip(conflicts, conflict_thresholds):
                                clist.append(ttc_min[-1] < th)


        ttc_results = dict(
            id1=partner_1_id,
            name1=partner_1_name,
            id2=partner_2_id,
            name2=partner_2_name,
            t_begin=t_begin,
            t_end=t_end,
            ttc_min=ttc_min,
            ttc_max=ttc_max,
            ttc_mean=ttc_mean,
            t_ttc_min = t_ttc_min,
            t_ttc_max = t_ttc_max,
            x1_ttc_min = x1_ttc_min,
            y1_ttc_min = y1_ttc_min,
            psi1_ttc_min = psi1_ttc_min, 
            x2_ttc_min = x2_ttc_min,
            y2_ttc_min = y2_ttc_min,
            psi2_ttc_min = psi2_ttc_min,
        )

        for clist, th in zip(conflicts, conflict_thresholds):
            label = f"{int(th*1000):04d}ms-conflict"
            ttc_results[label] = clist
        ttc_results = pd.DataFrame(ttc_results)

        if self.verbose:
            print(ttc_results)

        return ttc_results, ttc
