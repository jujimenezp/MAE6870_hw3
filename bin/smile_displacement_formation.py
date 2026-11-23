import numpy as np
from matplotlib.animation import FFMpegWriter
import rps.robotarium as robotarium
from rps.utilities.graph import topological_neighbors
from rps.utilities.transformations import create_si_to_uni_dynamics
from rps.utilities.barrier_certificates import create_si_barrier_certificate_with_boundary
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
N = 8
iterations = 2000
formation_control_gain = 10

# =========================================================
# DATA STORAGE INITIALIZATION
# =========================================================
# Pre-populate timestep 0 from x_init and readings after the first step.
# Lists of (K, N) frames are stacked to (K, N, T) at save time.
vicon_list      = []

# =========================================================
# ROBOTARIUM INITIALIZATION
# =========================================================
# Initializing with the updated Width/Height/Spacing from MATLAB
# initial_conditions = generate_random_poses(
#     N, 
#     width=2.5, 
#     height=1.5, 
#     spacing=0.6
# )
initial_conditions = np.array([[0.02576052,  0.54828738, -0.13920942, -1.00318009, -1.06295826,  1.10838442, 1.11857774,  0.57111998],
 [-0.59096927,  0.0222632,   0.34629195, -0.51565993,  0.20769649, -0.56014792, 0.28568192,  0.66814097],
 [-1.39302643,  0.58631807, -2.84446966, -0.57210556,  2.03477767,  0.44730757, -2.79137411,  0.06005052]])
print(initial_conditions)

r = robotarium.Robotarium(
    number_of_robots=N, 
    show_figure=True, 
    initial_conditions=initial_conditions,
    sim_in_real_time=True
)

# =========================================================
# GRAPH TOPOLOGY
# =========================================================
# Adjacency Matrix
adjacency_mat = np.zeros((N,N), dtype=int)

# Define undirected communication edges
edges = [
    # Smile (robots 1-5)
    (1, 2),(2, 3),(3, 4),(4, 5),(1, 3),(2, 4),(3, 5),
    # Eyes (robots 6-7)
    (1, 6),(2, 6),(4, 7),(5, 7),(6, 7),
    # Nose (robot 8)
    (3, 8),(6, 8),(7, 8)]

for i, j in edges:
    adjacency_mat[i-1, j-1] = 1
    adjacency_mat[j-1, i-1] = 1

# Laplacian for a 2D-rigid smile formation
L = np.diag(np.sum(adjacency_mat, axis=1)) - adjacency_mat
print(L)

# =========================================================
# FORMATION GEOMETRY
# =========================================================
desired_positions = np.array([[-0.7, -0.45, 0.00, 0.45, 0.7, -0.35, 0.35, 0.00],
                              [-0.15, -0.38, -0.5, -0.38, -0.15, 0.3, 0.3, 0.05]])


# =========================================================
# CONTROLLER AND SAFETY SETUP
# =========================================================
# Single-integrator barrier certificate for safety
si_barrier_cert = create_si_barrier_certificate_with_boundary()

# Mapping with specific MATLAB gains
si_to_uni_dyn = create_si_to_uni_dynamics(
    linear_velocity_gain=0.5, 
    angular_velocity_limit=np.pi/2
)

# =========================================================
# PLOT INITIALIZATION
# =========================================================
ax = r._axes_handle
fig = r._fig

writer = FFMpegWriter(fps=20)

# One colour per robot
CM = ['k', 'b', 'r', 'g', 'yellow', 'purple', 'orange', 'cyan']

marker_size_goal = determine_marker_size(r, 0.20)
font_size        = determine_font_size(r, 0.05)
line_width       = 5

# Get initial poses before drawing edges
x = r.get_poses()

# Follower-follower edges (blue): L == 1 gives the directed edge source
rows, cols = np.where(L == -1)
# Each undirected edge appears once as a directed entry with value 1
lf = []
for k in range(len(rows)):
    line, = ax.plot([x[0, rows[k]], x[0, cols[k]]],
                    [x[1, rows[k]], x[1, cols[k]]],
                    color='b', linewidth=line_width, zorder=1)
    lf.append(line)

# Follower labels (off-screen initially)
follower_labels = []
for j in range(N):
    fl = ax.text(500, 500, f'Robot {j+1}',
                 fontsize=font_size, fontweight='bold', color='k', zorder=5)
    follower_labels.append(fl)

r.step()

# =========================================================
# FIGURE RESIZE EVENT LISTENER
# =========================================================
def on_resize(event):
    """Dynamically scales markers and text when the figure window is resized."""
    new_goal_size = determine_marker_size(r, 0.20)
    font_size     = determine_font_size(r, 0.05)
        
    # Scale robot labels
    for fl in follower_labels:
        fl.set_fontsize(font_size)
        
    ax.figure.canvas.draw_idle()

ax.figure.canvas.mpl_connect('resize_event', on_resize)

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
with writer.saving(fig, "results/smile_displacement_sim.mp4", dpi=150):
    for t in range(int(iterations)):
        x = r.get_poses()
        dxi = np.zeros((2, N))

        # ---------------------------------------------------------
        # Record sensor data
        # ---------------------------------------------------------
        vicon_list.append(x)

        # Formation control algorithm using edge tension energy
        for i in range(N):
            neighbors = topological_neighbors(L, i)
            for j in neighbors:
                # gradient = gain * (||actual||^2 - ||desired||^2) * (pos_j - pos_i)
                dxi[:, i] += formation_control_gain * ((x[:2,j]-desired_positions[:,j]) - (x[:2,i]-desired_positions[:,i]))

        # ---------------------------------------------------------
        # Velocity thresholding (3/4 of max linear velocity)
        # ---------------------------------------------------------
        norms = np.linalg.norm(dxi, axis=0)
        threshold = 0.75 * r.MAX_LINEAR_VELOCITY
        to_thresh = norms > threshold
        
        if np.any(to_thresh):
            dxi[:, to_thresh] *= threshold / norms[to_thresh]

        # ---------------------------------------------------------
        # Safety and Dynamics conversion
        # ---------------------------------------------------------
        dxi = si_barrier_cert(dxi, x[:2, :])
        dxu = si_to_uni_dyn(dxi, x)

    # ---------------------------------------------------------
        # Update plot handles
        # ---------------------------------------------------------
        # Follower labels track their robots
        for q in range(N):
            follower_labels[q].set_position((x[0, q] - 0.15, x[1, q] + 0.15))

        # Follower-follower graph edges
        for m in range(len(rows)):
            lf[m].set_xdata([x[0, rows[m]], x[0, cols[m]]])
            lf[m].set_ydata([x[1, rows[m]], x[1, cols[m]]])

        #GIF
        writer.grab_frame()

        r.set_velocities(np.arange(N), dxu)
        r.step()

# =========================================================
# SAVE DATA
# =========================================================
# Stack list of (K, N) frames into (K, N, T) arrays
#np.save('smile_displacement_formation.npy',      np.stack(vicon_list,      axis=2))

# Final debug report
r.debug()