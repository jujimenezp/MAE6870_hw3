import numpy as np
from matplotlib.animation import FFMpegWriter
import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.graph import completeGL, topological_neighbors
from rps.utilities.barrier_certificates import create_uni_barrier_certificate_with_boundary
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.misc import generate_random_poses

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def determine_marker_size(robotarium_instance, marker_size_meters):
    """Marker size in points matching a physical width in metres."""
    fig  = robotarium_instance._fig
    ax   = robotarium_instance._axes_handle
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    arena_w = robotarium_instance.BOUNDARIES[1] - robotarium_instance.BOUNDARIES[0]
    return (marker_size_meters / arena_w) * bbox.width

def determine_font_size(robotarium_instance, font_height_meters):
    """Font size in points matching a physical height in metres."""
    fig  = robotarium_instance._fig
    fig.canvas.draw()
    fig_h_px = fig.get_size_inches()[1] * fig.dpi
    arena_h  = robotarium_instance.BOUNDARIES[3] - robotarium_instance.BOUNDARIES[2]
    return fig_h_px * (font_height_meters / arena_h)

# =========================================================
# SIMULATION PARAMETERS
# =========================================================
N          = 12
iterations = 2500

# Flocking parameters
eps = 0.01
k_s = 0.01
k_a = 1.2
k_c = 0.25
k_p = 0.4
k_v = 1.
radius = 0.75

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
#initial_positions = generate_random_poses(N, width=2.6, height=1.3)
initial_positions = np.array([[-0.9992209 ,  0.28321623,  0.90871601,  1.02558021, -0.5912359 ,
         0.20542356, -1.25778341, -0.3406298 ,  0.47446269, -0.07931621,
         1.26279217,  1.22038745],
       [ 0.57694281,  0.12331217, -0.20646003,  0.61395032, -0.28886853,
        -0.60313725, -0.21284467,  0.42538513,  0.58611979, -0.19908898,
         0.19146862, -0.63949412],
       [-2.64683027,  2.87979268,  1.69410648,  1.96011402,  2.00228801,
        -1.73499395, -2.59374404, -1.9499213 , -1.84372041, -2.51696462,
        -2.89302544,  2.19846471]])
print(initial_positions)
r = robotarium.Robotarium(number_of_robots=N, show_figure=True,
                           initial_conditions=initial_positions,
                           sim_in_real_time=True)
dt = r.TIME_STEP
dxi = np.zeros((2, N))
# =========================================================
# GRAPH TOPOLOGY
# =========================================================

# Proximity-based adjacency matrix
def adj_matrix(x,R=radius):
    adjacency_mat = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(i+1,N):
            if np.linalg.norm(x[:,i]-x[:,j]) < R:
                adjacency_mat[i,j] = 1
                adjacency_mat[j,i] = 1
    return adjacency_mat

# Initialize adjacency mat
adjacency_mat = adj_matrix(initial_positions[:2,:])

# Laplacian for a 2D-rigid smile formation
L = np.diag(np.sum(adjacency_mat, axis=1)) - adjacency_mat
print(L)

# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
si_to_uni_dyn    = create_si_to_uni_dynamics(linear_velocity_gain=0.8)
uni_barrier_cert = create_uni_barrier_certificate_with_boundary()

# =========================================================
# WAYPOINT INITIALIZATION
# =========================================================
# Corner waypoints
waypoints = np.array([[-0.8, -0.8, 0, 0.8, 0.8,0],
                      [0.6, -0.5, 0.1, -0.5, 0.6, -0.1]])
close_enough = 0.15
state = 0 

# =========================================================
# PLOT INITIALIZATION
# =========================================================
ax = r._axes_handle
fig = r._fig

# GIF
writer = FFMpegWriter(fps=20)

marker_size_goal = determine_marker_size(r, 0.20)
font_size        = determine_font_size(r, 0.05)
line_width       = 5

# Waypoint goal markers and labels
g            = []
goal_labels  = []
for i in range(waypoints.shape[1]):
    gm, = ax.plot(waypoints[0, i], waypoints[1, i], 's',
                  markersize=marker_size_goal, 
                  markerfacecolor='none',
                  markeredgecolor='red',
                  markeredgewidth=line_width,
                  zorder=2)
    g.append(gm)
    gl = ax.text(waypoints[0, i] - 0.05, waypoints[1, i],
                 f'G{i+1}', fontsize=font_size, fontweight='bold',
                 color='k', zorder=2)
    goal_labels.append(gl)

# Get initial poses before drawing edges
x = r.get_poses()
c = np.average(x[:2,:], axis=1)
c_plot = ax.scatter(c[0],c[1],s=50, color='green', zorder=3)

# Follower-follower edges (blue): L == 1 gives the directed edge source
rows, cols = np.where(L == -1)
# Each undirected edge appears once as a directed entry with value 1
lf = []
for k in range(len(rows)):
    line, = ax.plot([x[0, rows[k]], x[0, cols[k]]],
                    [x[1, rows[k]], x[1, cols[k]]],
                    color='b', linewidth=line_width, zorder=1)
    lf.append(line)

# Leader label (off-screen initially)
leader_label = ax.text(500, 500, 'Leader',
                       fontsize=font_size, fontweight='bold', color='r', zorder=5)


r.step()

# =========================================================
# FIGURE RESIZE EVENT LISTENER
# =========================================================
def on_resize(event):
    """Dynamically scales markers and text when the figure window is resized."""
    new_goal_size = determine_marker_size(r, 0.20)
    font_size     = determine_font_size(r, 0.05)
        
    ax.figure.canvas.draw_idle()

ax.figure.canvas.mpl_connect('resize_event', on_resize)

# =========================================================
# DATA STORAGE INITIALIZATION
# =========================================================
# Pre-populate timestep 0 from x_init and readings after the first step.
# Lists of (K, N) frames are stacked to (K, N, T) at save time.
vicon_list      = [] # Robots positions
center_waypoint_list = [] # Current formation center and waypoint

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
with writer.saving(fig, "results/boids_leader_follower_sim_last.mp4", dpi=150):
    for t in range(iterations):
        x = r.get_poses()
        u = np.zeros((2, N))
        waypoint = waypoints[:, state]
        c = np.average(x[:2,:], axis=1)

        # ---------------------------------------------------------
        # Record sensor data
        # ---------------------------------------------------------
        vicon_list.append(x)
        center_waypoint_list.append([state,waypoint[0],waypoint[1],c[0],c[1]])

        # -----------------------------------
        # Update adjacency matrix, Laplacian, and weights
        # -----------------------------------
        adjacency_mat = adj_matrix(x[:2,:])
        L = np.diag(np.sum(adjacency_mat, axis=1)) - adjacency_mat

        # 1. Formation control
        for i in range(N):
            neighbors = topological_neighbors(L, i)
            for j in neighbors:
                # separation
                u[:, i] += k_s* (x[:2, i]-x[:2, j])/(np.sum((x[:2, i]-x[:2, j])**2)+eps**2)
                # alignment
                u[:,i] += k_a/len(neighbors) * (dxi[:,j] - dxi[:,i])
                # cohesion
                u[:,i] += k_c/len(neighbors) * (x[:2,j] - x[:2,i])
        # Waypoint tracking (only b_0 = 1)
        u[:,0] += k_p*(waypoint - x[:2,0]) - k_v * dxi[:,0]

        dxi += dt*u

        # Advance waypoint state if leader is close
        dist_to_waypoint = np.array([np.linalg.norm(x[:2,i] - waypoint) for i in range(N)])
        if min(dist_to_waypoint) < close_enough:
            state = (state + 1) % waypoints.shape[1]

        # 3. Velocity thresholding (3/4 of max linear velocity)
        norms = np.linalg.norm(dxi, axis=0)
        threshold = 0.75 * r.MAX_LINEAR_VELOCITY
        to_thresh = (norms > threshold)

        if np.any(to_thresh):
            dxi[:, to_thresh] *= threshold / norms[to_thresh]

        # 4. Conversion and Safety
        # Convert SI velocities to unicycle commands
        dxu = si_to_uni_dyn(dxi, x)
        # Apply barrier certificate in unicycle space
        dxu = uni_barrier_cert(dxu, x)

        # ---------------------------------------------------------
        # Update plot handles
        # ---------------------------------------------------------
        # Follower labels track their robots  
        c_plot.set_offsets([c[0],c[1]])

        # Leader label and leader-follower edge
        leader_label.set_position((x[0, 0] - 0.15, x[1, 0] + 0.15))

        for m in range(len(rows)):
            lf[m].remove()
        # Edges (blue): L == 1 gives the directed edge source
        rows, cols = np.where(L == -1)
        # Each undirected edge appears once as a directed entry with value 1
        lf = []
        for k in range(len(rows)):
            line, = ax.plot([x[0, rows[k]], x[0, cols[k]]],
                            [x[1, rows[k]], x[1, cols[k]]],
                            color='b', linewidth=line_width, zorder=1)
            lf.append(line)

        # GIF
        writer.grab_frame()

        r.set_velocities(np.arange(N), dxu)
        r.step()

# =========================================================
# SAVE DATA
# =========================================================
# Stack list of (K, N) frames into (K, N, T) arrays
np.save('boids_leader_follower.npy', np.stack(vicon_list, axis=2))
np.save('boids_leader_follower_centers.npy', np.transpose(center_waypoint_list))

# Final debug report
r.debug()