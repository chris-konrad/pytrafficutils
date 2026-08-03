# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:00:26 2023

pytrafficutils.ssm

Surrogate safety measures for traffic analysis 

@author: Christoph M. Konrad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrafficutils.agents import Agent
from shapely import unary_union
from shapely.plotting import plot_polygon

from pypaperutils.design import TUDcolors

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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
            Trajectory array shaped (n_trajs, n_samples, n_states) with [:,:,0] the x coordinates and [:,:,1] the y coordinates.

        Returns:
        --------
        dist : np.ndarray
            Array of shape (n_trajs, n_trajs, n_samples), where dist[i,j,k] is the distance between rajectory i and trajectory j at sample k 
        """
        
        diff = trajs[:, np.newaxis, :, :2] - trajs[np.newaxis, :, :, :2]  # shape = (n_trajs, n_trajs, n_samples, 2)
        dist = np.linalg.norm(diff, axis=-1)  
        
        return dist


    def _calc_minpathdist(trajs):
        """ Given the trajectories (n_trajs, n_samples, 2), calculate the minimum culician distance between paths across all times.
        
        Parameters:
        ----------
        trajs: array-like
            Trajectory array shaped (n_trajs, n_samples, n_states)

        Returns:
        --------
        dmin: array-like
            Minimum distances between paths with dmin shaped (n_trajs, n_samples) and dmin[i, j, k] being the minimum distance
            of sample k in path i to path j. 
    
        """

        diff = trajs[:, np.newaxis, :, np.newaxis, :] - trajs[np.newaxis, :, np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        return np.nanmin(dist, axis=-1)


    def eval_PET(self, max_pet = 5, min_tgap_between_conflicts = 0.5, conflict_thresholds=[0.5, 1.0, 1.5, 2.0], ignore_edgecases=False, plot=False):

        times = self.t_begin + np.round(np.arange(self.n_steps) * self.t_s,2)
        n_grow = int(np.ceil(min_tgap_between_conflicts / self.t_s))

        # build array of all trajectories. 
        trajs =  np.nan * np.zeros((self.n_agents, self.n_steps, 2))
        max_extent = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            is_active = np.logical_and(times >= self.t_ranges[i,0], times <= np.round(self.t_ranges[i,1],2))
            trajs[i, is_active, :] = self.agents[i].get_state_trajectories(states=['p_x', 'p_y'])

            max_extent[i] = self.agents[i].footprint.max_radial_extent

        # calculate minimum distances.
        min_pathdists = SurrogateSafetyMetricProcessor._calc_minpathdist(trajs)

        max_extent = max_extent[np.newaxis,:] + max_extent[:, np.newaxis]   
        max_extent = max_extent[:,:,np.newaxis]

        is_close = min_pathdists < max_extent          #[i, j, k]: sample k of trajectory i is close to trajectory j
        is_close_anysample = np.any(is_close, axis=-1) #[i,j]: Trajectoy i is close to trajectory j at any sample. 
        is_close_anytraj = np.any(is_close, axis=-2)   #[i,k]: Sample k of trajectory i is close to any other sample

        traces = []
        footprints = []
        for i in range(self.n_agents):
            if np.any(is_close[i, :, :]):   #is close to another path

                footprints_t = []
                footprints_i = []
                close_indices = np.argwhere(is_close_anytraj[i]).flatten()
                for k in close_indices:
                    footprints_t.append(k)
                    footprints_i.append(self.agents[i].get_footprint_at(times[k]))
                footprints.append([footprints_t, footprints_i])
                traces.append(unary_union(footprints_i))
            else:
                footprints.append([[],[]])
                traces.append(None)

        pet_results = [] 
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if j <= i:
                    continue
                if is_close_anysample[i,j]:
                    trace_intersections = traces[i].intersection(traces[j])

                    def _calc_conflict_indices(ii, jj):
                        close_indices = np.argwhere(is_close[ii, jj, :]).flatten()
                        is_in_conflict = []
                        for l, k in enumerate(close_indices):
                            fp_k = footprints[ii][0].index(k)
                            if footprints[ii][1][fp_k].intersects(trace_intersections):
                                is_in_conflict.append(k)

                        mask = np.zeros((self.n_steps,), dtype=bool)
                        mask[is_in_conflict] = True
                        mask = np.convolve(mask, np.ones(2*n_grow+1), mode='same').astype(bool)
                        conflict_indices = np.flatnonzero(np.diff(np.r_[0, mask, 0])).reshape(-1,2)
                        conflict_indices[:,0] += n_grow
                        conflict_indices[:,1] -= n_grow
                        return conflict_indices

                    conflict_indices_i = _calc_conflict_indices(i, j)
                    conflict_indices_j = _calc_conflict_indices(j, i)

                    assert conflict_indices_i.shape == conflict_indices_j.shape

                    for l in range(conflict_indices_i.shape[0]):
                        if conflict_indices_i[l,0] < conflict_indices_j[l,0]:
                            id1 = i
                            i_pet_a1 = conflict_indices_i[0,0] + np.argmin(min_pathdists[i, j, conflict_indices_i[0,0]:conflict_indices_i[0,1]]).flatten()
                        
                            id2 = j
                            i_pet_a2 = conflict_indices_j[0,0] + np.argmin(min_pathdists[j, i, conflict_indices_j[0,0]:conflict_indices_j[0,1]]).flatten()
                        else:
                            id1 = j
                            i_pet_a1 = conflict_indices_j[0,0] + np.argmin(min_pathdists[j, i, conflict_indices_j[0,0]:conflict_indices_j[0,1]]).flatten()

                            id2 = i
                            i_pet_a2 = conflict_indices_i[0,0] + np.argmin(min_pathdists[i, j, conflict_indices_i[0,0]:conflict_indices_i[0,1]]).flatten()

                        a1 = self.agents[id1]
                        a2 = self.agents[id2]

                        i_entr_1st = min(conflict_indices_i[l,0], conflict_indices_j[l,0])
                        i_entr_2nd = max(conflict_indices_i[l,0], conflict_indices_j[l,0])
                        i_exit_1st = min(conflict_indices_i[l,1], conflict_indices_j[l,1])
                        i_exit_2nd = max(conflict_indices_i[l,1], conflict_indices_j[l,1])

                        t_entr_1st = times[i_entr_1st]
                        t_entr_2nd = times[i_entr_2nd]
                        t_exit_1st = times[i_exit_1st]
                        t_exit_2nd = times[i_exit_2nd]

                        pet = t_entr_2nd - t_exit_1st

                        is_conflict = np.array(conflict_thresholds) > pet

                        if pet > max_pet:
                            continue

                        s1_pet = a1.get_state(times[i_pet_a1][0])
                        s2_pet = a2.get_state(times[i_pet_a2][0])        

                        if pet < 0:
                            collision_ij = footprints[i][i_exit_1st].intersects(footprints[j][i_entr_2nd])
                            collision_ji = footprints[i][i_entr_2nd].intersects(footprints[j][i_exit_1st])
                            if collision_ij or collision_ji:
                                pet = 0
                            elif ignore_edgecases:
                                continue
                            else:
                                msg = (f"At {t_entr_2nd} agents {a1.name} and {a2.name} are simultaneously present"
                                       f" on the same conflict patch without collision! Automatic interpretation of"
                                       f" this edge case is currently not implemented. Run with 'ignore_edgecases=True'" 
                                       f" to suppress error.")
                                raise NotImplementedError(msg)

                        row = [id1, a1.name, id2, a2.name, 
                               t_entr_1st, t_entr_2nd, t_exit_1st, t_exit_2nd,
                               pet, 
                               s1_pet[0], s1_pet[1], s1_pet[2],
                               s2_pet[0], s2_pet[1], s2_pet[2]]
                        row += is_conflict.tolist()

                        pet_results.append(row)

        col_names = ["id1", "name1", "id2", "name2", 
                     "t_enter1_s", "t_enter2_s", "t_exit1_s", "t_exit2_s",
                     "pet_s",
                     "x1_pet_m", "y1_pet_m", "psi1_pet_rad", "x2_pet_m", "y2_pet_m", "psi2_pet_rad"]
        col_names += [f"{int(th*1000):04d}ms-conflict" for th in conflict_thresholds]

        pet_results = pd.DataFrame(pet_results, columns=col_names)

        if plot:
            figures = []
            cmp1 = TUDcolors().colormap('oranje')
            cmp2 = TUDcolors().colormap('lichtgroen')
            for i, row in pet_results.iterrows():

                fig, ax = plt.subplots(1,1, layout='constrained')
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_title(f"Post Encroachment Time (PET) - Conflict PET#{i}\n {row['name1']} vs. {row['name2']}")
                ax.set_aspect("equal")

                a1 = self.agents[row["id1"]]
                a2 = self.agents[row["id2"]]

                traj_a1 = a1.get_state_trajectories(states=['p_x', 'p_y'])
                traj_a2 = a2.get_state_trajectories(states=['p_x', 'p_y'])

                plot_polygon(traces[row["id1"]], edgecolor='none', facecolor='gray', alpha=0.4, add_points=False)
                plot_polygon(traces[row["id2"]], edgecolor='none', facecolor='gray', alpha=0.4, add_points=False)

                ax.plot(traj_a1[:,0], traj_a1[:,1], color='black', zorder=0)
                ax.plot(traj_a2[:,0], traj_a2[:,1], color='black', zorder=0)

                t_plot = times[np.argwhere(times==(row['t_exit1_s'])).flatten()[0]:np.argwhere(times==(row['t_enter2_s'])).flatten()[0]]
                norm = Normalize(vmin=t_plot[0]*0.8, vmax=t_plot[-1]*1.2)
                for t in t_plot:
                    if t <= a1.t[1]:
                        ta1 = t
                        a1.draw_footprint_at(ta1, ax, color=cmp1(norm(ta1)), alpha=1)
                    if t <= a2.t[1]:
                        ta2 = t
                        a2.draw_footprint_at(ta2, ax, color=cmp2(norm(ta2)), alpha=1)
                a1.draw_footprint_at(ta1, ax, facecolor=cmp1(norm(ta1)), edgecolor='black', alpha=1, label=a1.name)
                a2.draw_footprint_at(ta2, ax, facecolor=cmp2(norm(ta2)), edgecolor='black', alpha=1, label=a2.name)

                s1_0 = a1.get_state(t_plot[0])
                s1_1 = a1.get_state(ta1)
                s2_0 = a2.get_state(t_plot[0])
                s2_1 = a2.get_state(ta2)

                xmin = min(s1_0[0], s1_1[0], s2_0[0], s2_1[0])-4
                xmax = max(s1_0[0], s1_1[0], s2_0[0], s2_1[0])+4
                ymin = min(s1_0[1], s1_1[1], s2_0[1], s2_1[1])-4
                ymax = max(s1_0[1], s1_1[1], s2_0[1], s2_1[1])+4

                # adjust frame ratio
                dx = (xmax - xmin)
                dy = (ymax - ymin) 
                ratio = dx / dy
                if ratio > 4/3:
                    r0 = 4/3
                    a = r0 * dy / dx
                    dy_add = (dy / a - dy) /2
                    ymin -= dy_add
                    ymax += dy_add
                elif ratio < 3/4:
                    r0 = 3/4
                    a = r0 * dy / dx
                    dx_add = (dx*a - dx) /2
                    xmin -= dx_add
                    xmax += dx_add

                sm1 = ScalarMappable(norm=norm, cmap=cmp1)
                sm2 = ScalarMappable(norm=norm, cmap=cmp2)
                cbar1 = fig.colorbar(sm1, ax=ax)
                cbar2 = fig.colorbar(sm2, ax=ax)

                cbar1.set_label('time [s]')
                cbar2.set_ticks([])
                cbar1.ax.set_ylim([t_plot[0], t_plot[-1]])
                cbar2.ax.set_ylim([t_plot[0], t_plot[-1]])

                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.legend(loc='lower left')

                ax.annotate(f"PET = {row['pet_s']:.2f} s", (0.05, 0.95), xycoords='axes fraction', va='top', ha='left', fontsize=20)

                figures.append(fig)

            return pet_results, figures
        else:
            return pet_results
            

    def eval_TTC(self, T_extrapolation, t_s_extrapolation=0.01, min_tgap_between_conflicts = 1, conflict_thresholds=[0.5, 1.0, 1.5, 2.0], plot=False):
        """Evaluate the scenario for Time to Collision (TTC)

        Uses linear extrapolation of the current state assuming constant speed and constant yaw angle:
            dx = v * cos(psi)
            dy = v * sin(psi)

        Evaluates extrapolations for collisions based on footprint overlap.

        Parameters
        ----------
        T_extrapolation : float
            The maximum time [s] to extrapolate. This equals the maximum TTC that will be detected. 
        t_s_extrapolation : float, optional
            The step width (=sample time) [s] for extrapolation. By default 0.01 s
        min_tgap_between_conflicts : int, optional
            Minimum time gap [s] between separate conflicts. If the time between two conflicts is smaller 
            than this time, they will be treated as one conflict, by default 1 s
        conflict_thresholds : list, optional
            List of TTC thresholds [s] to classify an interaction as a conflict. An interaction is classified as a
            t-conflict if TTC < t. By default [0.5, 1.0, 1.5, 2.0]
        plot : bool, optional
            Visualize each conflict in a plot. By default False

        Returns
        -------
        ttc_results : pd.DataFrame
            The TTC results. Each row in the table corresponds to a conflict. All The columns are:
                id1 : int
                    Index of the first agent involved in a conflict.
                name1: str
                    Name of the first agent
                id2 : int
                    Index of the second agent involved in a conflict
                name2 : str
                    Name of the second agent
                t_begin : float
                    Time [s] of the beginning of the conflict
                t_end : float
                    Time [s] of the end of the conflict
                ttc_min : float
                    Minimum TTC [s] reached during the conflict
                ttc_max, float
                    Maximum TTC [s] reached during the conflict
                ttc_mean: float
                    Mean TTC [s] of the conflict
                t_ttc_min : float
                    Time of the minimum TTC.
                t_ttc_max : float
                    Time of the maximum TTC.
                x1_ttc_min : float
                    x-position [m] of the first agent at the minimum TTC 
                y1_ttc_min : float
                    y-position [m] of the first agent at the minimum TTC 
                psi1_ttc_min : psi1_ttc_min, 
                    yaw angle [rad] of the first agent at the minimum TTC
                x2_ttc_min : float
                    x-position [m] of the second agent at the minimum TTC 
                y2_ttc_min : float
                    y-position [m] of the second agent at the minimum TTC 
                psi2_ttc_min : psi2_ttc_min,
                    yaw angle [rad] of the second agent at the minimum TTC 
        ttc_trajectories : np.ndarray
            The trajectories of TTC values [s] between pairs of agents shaped
            (n_agents, n_agents, n_steps). Thus, [i, j, k] returns the TTC value 
            between agent i and agent j at time step k. NaN marks samples for which
            the TTC value does not exits. 

        """
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

                            gap_has_min_length = t1 - t0 >= min_tgap_between_conflicts
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
                            t_ttc_min.append(np.argwhere(ttc_min[-1]==ttc_m).flatten()[0]*self.t_s + interaction_begin[m]*self.t_s)
                            t_ttc_max.append(np.argwhere(ttc_max[-1]==ttc_m).flatten()[0]*self.t_s + interaction_begin[m]*self.t_s)

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

        if plot:

            col1 = TUDcolors().get("oranje")
            col2 = TUDcolors().get("lichtgroen")

            cmp1 = TUDcolors().colormap("oranje")
            cmp2 = TUDcolors().colormap("lichtgroen")

            red = TUDcolors().get("rood")

            figs = []

            for i, row in ttc_results.iterrows():
                fig, axes = plt.subplots(1,2, layout='constrained')
                fig.suptitle(f"Time to Collision (TTC) - Conflict TTC#{i}\n {row['name1']} vs. {row['name2']}")
                fig.set_size_inches(15,6.2)

                # xy-plot
                ax = axes[0]
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_title("conflict trajectories (XY)")
                ax.set_aspect("equal")

                a1 = self.agents[row["id1"]]
                a2 = self.agents[row["id2"]]

                traj_a1 = a1.get_state_trajectories(states=['p_x', 'p_y'])
                traj_a2 = a2.get_state_trajectories(states=['p_x', 'p_y'])

                i1_begin = a1._find_state_index_at_time(row['t_begin'])
                i1_end = a1._find_state_index_at_time(row['t_end'])
                i2_begin = a2._find_state_index_at_time(row['t_begin'])
                i2_end = a2._find_state_index_at_time(row['t_end'])

                extrpol1 = a1.extrapolate_from_t(row['t_ttc_min'], row['ttc_min'], t_s_extrapolation)
                extrpol2 = a2.extrapolate_from_t(row['t_ttc_min'], row['ttc_min'], t_s_extrapolation)
                ax.plot(extrpol1[:,0], extrpol1[:,1], color=col1, linestyle='dashed')
                ax.plot(extrpol2[:,0], extrpol2[:,1], color=col2, linestyle='dashed')

                ax.plot(traj_a1[i1_begin:i1_end,0], traj_a1[i1_begin:i1_end,1], color=col1, linewidth=5)
                ax.plot(traj_a2[i2_begin:i2_end,0], traj_a2[i2_begin:i2_end,1], color=col2, linewidth=5)
                ax.plot(traj_a1[:,0], traj_a1[:,1], color='black', linewidth=1)
                ax.plot(traj_a2[:,0], traj_a2[:,1], color='black', linewidth=1)

                a1.draw_footprint_at(row['t_ttc_min'], ax, facecolor=col1, edgecolor='black', alpha=1, label=a1.name)
                a2.draw_footprint_at(row['t_ttc_min'], ax, facecolor=col2, edgecolor='black', alpha=1, label=a2.name)
                a1.footprint.plot_at(extrpol1[-1,:], ax, edgecolor='none', facecolor=cmp1(0.3))
                a2.footprint.plot_at(extrpol2[-1,:], ax, edgecolor='none', facecolor=cmp2(0.3))

                s1_0 = a1.get_state(row['t_begin'])
                s1_1 = a1.get_state(row['t_end'])
                s2_0 = a2.get_state(row['t_begin'])
                s2_1 = a2.get_state(row['t_end'])

                ax.scatter([s1_0[0], s2_0[0]], [s1_0[1], s2_0[1]], marker='o', facecolors='none', edgecolors='black', label=f"t_begin: {row['t_begin']:.2f} s", zorder=100)
                ax.scatter([s1_1[0], s2_1[0]], [s1_1[1], s2_1[1]], marker='o', facecolors='black', edgecolors='black', label=f"t_end: {row['t_end']:.2f} s", zorder=100)
                ax.plot([row['x1_ttc_min'], row['x2_ttc_min']], [row['y1_ttc_min'], row['y2_ttc_min']], marker='o', linestyle='none', markerfacecolor='black', markerfacecoloralt='none', fillstyle='left', markeredgecolor='black', label=f't_ttcmin: {row['t_ttc_min']:.2f} s', zorder=100)
                ax.plot([extrpol1[-1,0], extrpol2[-1,0]], [extrpol1[-1,1], extrpol2[-1,1]], marker='o', linestyle='none', markerfacecolor='black', markerfacecoloralt='none', fillstyle='right', markeredgecolor='black', label=f'extrapolated collision: {row['t_ttc_min']+row['ttc_min']:.2f} s', zorder=100)

                xmin = min(s1_0[0], s1_1[0], s2_0[0], s2_1[0], np.min(extrpol1[:,0]), np.min(extrpol2[:,0]))-3
                xmax = max(s1_0[0], s1_1[0], s2_0[0], s2_1[0], np.max(extrpol1[:,0]), np.max(extrpol2[:,0]))+3
                ymin = min(s1_0[1], s1_1[1], s2_0[1], s2_1[1], np.min(extrpol1[:,1]), np.min(extrpol2[:,1]))-3
                ymax = max(s1_0[1], s1_1[1], s2_0[1], s2_1[1], np.max(extrpol1[:,1]), np.max(extrpol2[:,1]))+3

                # adjust frame ratio
                dx = (xmax - xmin)
                dy = (ymax - ymin) 
                ratio = dx / dy
                if ratio > 4/3:
                    r0 = 4/3
                    a = r0 * dy / dx
                    dy_add = (dy / a - dy) /2
                    ymin -= dy_add
                    ymax += dy_add
                elif ratio < 3/4:
                    r0 = 3/4
                    a = r0 * dy / dx
                    dx_add = (dx*a - dx) /2
                    xmin -= dx_add
                    xmax += dx_add
    
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.legend(loc='lower left')
                ax.grid()

                ax.annotate(f"TTC = {row['ttc_min']:.2f} s", (0.05, 0.95), xycoords='axes fraction', va='top', ha='left', fontsize=20)

                #t plot
                ax = axes[1]

                ax.set_title("TTC trajectory (t)")
                ax.set_xlabel("time [s]")
                ax.set_ylabel("TTC [s]")

                i_begin = np.argmin(np.abs(times-row['t_begin'])).flatten()[0]
                i_end = np.argmin(np.abs(times-row['t_end'])).flatten()[0]
                i_ttcmin = np.argmin(np.abs(times-row['t_ttc_min'])).flatten()[0]
                i_ttcmax = np.argmin(np.abs(times-row['t_ttc_max'])).flatten()[0]

                ax.plot(times[i_begin:i_end+1], ttc[row["id1"], row["id2"], i_begin:i_end+1], color='black', label='TTC(t)')
                ax.plot([times[i_ttcmin]], [row["ttc_min"]], color=red, linestyle='none', marker='^', label=f"min. TTC: {row["ttc_min"]:.2f} s")
                ax.plot([times[i_ttcmax]], [row["ttc_max"]], color=red, linestyle='none', marker='v', label=f"max. TTC: {row["ttc_max"]:.2f} s")
                ax.axhline(y=row["ttc_mean"], color=red, label=f"mean TTC: {row["ttc_mean"]:.2f} s")
                ax.grid()
                ax.legend()

                figs.append(fig)

            return ttc_results, ttc, figs

        return ttc_results, ttc
