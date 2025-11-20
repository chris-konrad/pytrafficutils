Pytrafficutils - Process traffic trajectories, extract surrogate safety metrics (SSMs), represent traffic agents.
==============================

A personal collection of tools frequently used for traffic analysis and simulation:
- Represent **traffic agents** (moving or fixed, road users or obstacles, cars or bicycles) and the trajectories of their kinematic states.
- Provide 2D polygons representing the **footprints** of different kinds of traffic agents (rectangls, diamonds).
- Calculate the Surrogate Safety Metrics (SSMs) Time-To-Collsision (TTC) and Post-Encroachment Time (PET).

## Disclaimer

The package is under development. It may contain bugs and sections of unused or insensible code. Major changes to this package are planned for the time to come. A proper API documentation is still missing. **Refer to the docstrings for API information.** 

## Installation

1. Clone this repository. 
   
   ```
   git clone  https://github.com/chris-konrad/pytrafficutils.git
   ```

2. Install the package and it's dependencies. Refer to `pyproject.toml` for an overview of the dependencies. 
   
   ```
   cd ./pytrafficutils
   pip install . 
   ```

## Modules

This package has several modules to define traffic agents, their trajectories, their footprints, and to process SSMs

### agents

A module defining traffic agents. A traffic agent can be moving or stationary and has a footprint defining it`s 2D shape. 
Agents hold their trajectory in time, can return their state at a requested time and extrapolate their states from a requested point in time. Several agents are available:

`pytrafficutils.agents.Agent()` 
   Agent base class. Not designed to create instances.

`pytrafficutils.agents.MovingAgent(Agent)` 
   Class for agents that move at least during a part of their time in the traffic environment. 

`pytrafficutils.agents.Bicycle(MovingAgent)` 
   Class for moving bicycles / cyclists with a diamond footprint. Moving agents can linearly extrapolate their state in time.

`pytrafficutils.agents.Car(MovingAgent)` 
   Class for moving cars with a rectangular footprint.

`pytrafficutils.agents.StationaryAgent(Agent)` 
   Class for agents that do not move during their entire time in the traffic environment. For example, a parked vehicle or fixed obstacle.

`pytrafficutils.agents.RectObstacle(StationaryAgent)` 
   Class for fixed rectangular obstacles.

### footprints

A module defining the footprints of traffic agents as 2D shapes. Footprints are defined as a polygon (set of nodes relative to the center/reference point/anchor). Footprints can be translated / rotated to different positions and support plotting. Several footprints are available:

`pytrafficutils.footprints.Footprint()`
   Footprint base class. Not designed to create instances.

`pytrafficutils.footprints.RectangularFootprint(Footprint)`
   A footprint shaped like a rectangle for use as cars / trucks / busses. 

`pytrafficutils.footprints.DiamondFootprint(Footprint)`
   A footprint shaped like a diamond / rhombus for use as bicycles. 

### ssm

A module to evaluate surrogate safety metrics (SSMs) using the `pytrafficutils.ssm.SurrogateSafetyProcessor()` class. This class takes a list of traffic agents `pytrafficutils.agents.Agent()` and evaluates SSMs on the trajectories of their kinematic states. Respects footprints where appropriate. Uses matrix operations where possible for computational efficiency. Currently supports the calculation of:
- Time-To-Collision (TTC): Calculates the TTC using linear extrapolation of the velocity (i.e., speed and heading). Reports the minimum TTC and classifies interactions as conflicts using (multiple) configurable TTC thresholds.

> [!warning]
> Currently, only TTC is supported by `pytrafficutils.ssm.SurrogateSafetyProcessor()`. The Post-Encroachment Time (PET) can be calculated using the function `pytrafficutils.ssm.pet()` for specific trajectory pairs, but this is not yet integrated with `SurrogateSafetyProcessor()` and does not consider footprints.

## Authors

- Christoph M. Konrad, c.m.konrad@tudelft.nl

License
--------------------

This package is licensed under the terms of the [MIT license](https://github.com/chrismo-schmidt/cyclistsocialforce/blob/main/LICENSE).

