"""
MAE 6870 - HW3, Part 2, Question 1: Flocking with Robo-Boids (Robotarium, Python)

Continuous-time model (from the assignment):
    p_i' = v_i ,   v_i' = u_i
    N_i(t) = { j != i : ||p_j - p_i|| < R }

    u_i = ks * sum_{j in N_i} (p_i - p_j) / (||p_i - p_j||^2 + eps^2)      (separation)
        + ka/|N_i| * sum_{j in N_i} (v_j - v_i)                             (alignment)
        + kc/|N_i| * sum_{j in N_i} (p_j - p_i)                             (cohesion)
        + b_i * [ kp * (w_l - p_i) - kv * v_i ]                              (waypoint steering)

Discrete-time velocity update (forward Euler, dt = 0.033 s = 1 Robotarium iteration):
    u_i[k]   = f( p[k], v[k], A[k], w_l[k] )
    v_i[k+1] = sat_vmax( v_i[k] + dt * u_i[k] )
The saturated v_i[k+1] is sent as a single-integrator velocity command, converted to
unicycle (v, w) with si_to_uni and filtered by the barrier certificate.

Velocity initialization / source:
    v_i[0] = 0 for all i (robots start at rest).
    The v_i used in the controller is the controller's own velocity state, i.e. the
    saturated command from the previous iteration (no differentiation of noisy poses).
"""

import numpy as np
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary

# =========================================================
# PARAMETERS
# =========================================================
N = 6                       # flock size
iterations = 3000           # 3000 * 0.033 s ~= 99 s
dt = 0.033                  # 1 iteration = 0.033 s

# Boids gains (assignment starting values)
eps = 0.01
ks = 0.01                   # separation
ka = 1.0                    # alignment
kc = 0.25                   # cohesion
kp = 0.2                    # waypoint attraction
kv = 1.0                    # waypoint velocity damping
R = 0.75                    # communication / sensing radius [m]

b = np.ones(N)              # waypoint access flags b_i (all robots informed)

# Waypoints (same sequence as Part 1B)
waypoints = np.array([[-0.9, -0.9, 0.9, 0.9],
                      [ 0.4, -0.4, -0.4, 0.4]])
close_enough = 0.15         # [m] advance when the flock centroid is this close

SHOW_FIGURE = True
SAVE_DATA = True
DATA_PREFIX = "flock"

# Hard-coded initial poses [x; y; theta] (shared by simulation and hardware),
# on the right side of the arena, away from the first waypoint.
initial_conditions = np.array([
    [ 0.40,  0.90,  0.40,  0.90,  0.40,  0.90],
    [ 0.50,  0.55,  0.00,  0.05, -0.50, -0.45],
    [ 0.00,  np.pi/2, -np.pi/2, np.pi, np.pi/4, -3*np.pi/4],
])


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def proximity_adjacency(positions, R):
    """
    A_ij = 1 if i != j and ||p_j - p_i|| < R, else 0.   (Part 1C helper)
    positions : (2, N) or (3, N) array of robot poses (only x, y are used)
    R         : communication radius, R > 0
    returns   : (N, N) symmetric 0/1 integer array
    """
    if R <= 0:
        raise ValueError("Communication radius R must be positive.")
    p = positions[:2, :].T                          # (N, 2)
    diff = p[:, None, :] - p[None, :, :]            # (N, N, 2)
    dist = np.linalg.norm(diff, axis=2)             # (N, N)
    A = (dist < R).astype(int)
    np.fill_diagonal(A, 0)                          # enforce i != j
    return A


def boids_acceleration(p, v, A, w, b):
    """
    Robo-Boids control input u_i for every robot (vectorised).
    p : (2, N) positions, v : (2, N) velocities, A : (N, N) adjacency,
    w : (2,) active waypoint, b : (N,) waypoint-access flags
    returns u : (2, N)
    """
    w = np.asarray(w, dtype=float).reshape(2, 1)
    deg = A.sum(axis=1)                             # |N_i|
    inv_deg = np.zeros(p.shape[1])
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]           # isolated robot -> no align/cohesion

    rel = p[:, :, None] - p[:, None, :]             # rel[:, i, j] = p_i - p_j
    dist2 = np.sum(rel**2, axis=0)                  # (N, N)

    u_sep = ks * np.sum(rel * (A / (dist2 + eps**2))[None, :, :], axis=2)
    u_ali = ka * inv_deg * (v @ A.T - deg * v)      # ka/|N_i| sum_j (v_j - v_i)
    u_coh = kc * inv_deg * (p @ A.T - deg * p)      # kc/|N_i| sum_j (p_j - p_i)
    u_nav = b * (kp * (w - p) - kv * v)             # b_i [kp (w - p_i) - kv v_i]

    return u_sep + u_ali + u_coh + u_nav


def saturate(v, vmax):
    """Scale each column of v so that ||v_i|| <= vmax (keeps direction)."""
    norms = np.linalg.norm(v, axis=0)
    scale = np.ones_like(norms)
    big = norms > vmax
    scale[big] = vmax / norms[big]
    return v * scale


# =========================================================
# ROBOTARIUM SETUP
# =========================================================
r = robotarium.Robotarium(number_of_robots=N, show_figure=SHOW_FIGURE,
                          initial_conditions=initial_conditions, sim_in_real_time=True)

si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=0.8)
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()
v_max = 0.75 * r.MAX_LINEAR_VELOCITY                # speed limit (0.15 m/s)

if SHOW_FIGURE:
    ax = r._axes_handle
    wp_handles = []
    for k in range(waypoints.shape[1]):
        h, = ax.plot(waypoints[0, k], waypoints[1, k], 's', markersize=14,
                     markerfacecolor='none', markeredgecolor='k', markeredgewidth=3, zorder=2)
        ax.text(waypoints[0, k] + 0.06, waypoints[1, k] + 0.06, f'W{k+1}',
                fontsize=12, fontweight='bold', zorder=2)
        wp_handles.append(h)
    centroid_h, = ax.plot([], [], 'x', color='m', markersize=12, markeredgewidth=3, zorder=3)
    edge_h = []
    for i in range(N):
        for j in range(i + 1, N):
            ln, = ax.plot([], [], '-', color='0.6', linewidth=1.5, zorder=1)
            edge_h.append((i, j, ln))

# =========================================================
# STATE AND LOGS
# =========================================================
v = np.zeros((2, N))                         # v_i[0] = 0
wp_idx = 0

pose_log = np.zeros((3, N, iterations))
vel_log = np.zeros((2, N, iterations))       # v[k] used in the controller
u_log = np.zeros((2, N, iterations))
wp_log = np.zeros(iterations, dtype=int)

# =========================================================
# MAIN LOOP
# =========================================================
for k in range(iterations):
    x = r.get_poses()
    p = x[:2, :]

    # Proximity graph (Part 1C helper)
    A = proximity_adjacency(x, R)

    # Waypoint switching: flock centroid within close_enough of active waypoint
    centroid = p.mean(axis=1)
    if np.linalg.norm(centroid - waypoints[:, wp_idx]) < close_enough:
        wp_idx = (wp_idx + 1) % waypoints.shape[1]
    w = waypoints[:, wp_idx]

    # Boids acceleration and discrete velocity update
    u = boids_acceleration(p, v, A, w, b)
    v_next = saturate(v + dt * u, v_max)

    pose_log[:, :, k] = x
    vel_log[:, :, k] = v
    u_log[:, :, k] = u
    wp_log[k] = wp_idx

    # SI -> unicycle, collision avoidance, send
    dxu = si_to_uni_dyn(v_next, x)
    dxu = uni_barrier_cert(dxu, x)
    r.set_velocities(np.arange(N), dxu)

    v = v_next

    if SHOW_FIGURE:
        centroid_h.set_data([centroid[0]], [centroid[1]])
        for kk, h in enumerate(wp_handles):
            h.set_markeredgecolor('r' if kk == wp_idx else 'k')
        for (i, j, ln) in edge_h:
            if A[i, j]:
                ln.set_data([p[0, i], p[0, j]], [p[1, i], p[1, j]])
            else:
                ln.set_data([], [])

    r.step()

# =========================================================
# SAVE DATA
# =========================================================
if SAVE_DATA:
    np.save(DATA_PREFIX + "_poses.npy", pose_log)
    np.save(DATA_PREFIX + "_vel.npy", vel_log)
    np.save(DATA_PREFIX + "_u.npy", u_log)
    np.save(DATA_PREFIX + "_wp_idx.npy", wp_log)
    np.save(DATA_PREFIX + "_waypoints.npy", waypoints)

r.debug()
