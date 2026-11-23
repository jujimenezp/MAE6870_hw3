# Flocking_nav_less_boids — HW3 Part 2 (Robo-Boids flocking)

Reynolds' Boids (separation, alignment, cohesion) plus waypoint steering, run on the
Robotarium with 6 robots, the proximity graph from Part 1C (R = 0.75 m) and the Part 1B
waypoints.

| File | Question | Waypoint access | Switching rule |
|---|---|---|---|
| `flocking_boids.py` | Q1 (implementation) and Q2 (all robots informed) | b_i = 1 for all robots | flock centroid within `close_enough` |
| `flocking_boids_leader.py` | Q3 (leader only) | b_i = 1 for robot 0 only, 0 for the rest | leader within `close_enough` (0.15 m) |

Both scripts use the same gains (eps = 0.01, ks = 0.01, ka = 1, kc = 0.25, kp = 0.2,
kv = 1, R = 0.75), the same hard-coded initial poses, and the same speed limit
(0.75 x 0.2 m/s = 0.15 m/s).

## Discrete-time velocity update
Forward Euler with dt = 0.033 s (one Robotarium iteration):

    v_i[k+1] = sat( v_i[k] + dt * u_i[k] ),   ||v_i|| <= 0.15 m/s,   v_i[0] = 0

The saturated v_i[k+1] is sent as the single-integrator command (si_to_uni dynamics +
unicycle barrier certificate with boundary), and stored as the velocity used by the
controller in the next iteration.

## Running
Requires the Robotarium Python simulator (`pip install .` from
robotarium_python_simulator). Then:

    python flocking_boids.py           # Q1/Q2  -> flock_*.npy
    python flocking_boids_leader.py    # Q3     -> flock_leader_*.npy

Each run is 3000 iterations (~99 s) and saves poses, controller velocities, accelerations
and the active waypoint index as .npy files. For a hardware submission, upload only the
script as the main file, with 6 robots and an estimated duration of ~180 s.
