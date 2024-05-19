import torch
import matplotlib.pyplot as plt
import os
import pickle
import logging
from datetime import datetime

from src.models import SystemRobots, Controller
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_wp import f_loss_tl, f_loss_sum
from src.loss_wp import loss_TL_waypoints
from src.utils import calculate_collisions, set_params, generate_data
from src.utils import WrapLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random_seed = 3
torch.manual_seed(random_seed)

# sys_model = "corridor"
# sys_model = "robots"
sys_model = "corridor_wp"

# # # # # # # # Parameters and hyperparameters # # # # # # # #
params = set_params(sys_model)
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Qy, \
alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini = params
epochs = 2500
std_ini = 0
std_dist = 0.1
use_sp = True
alpha_sp = 5e-1 * 5
std_ini_param = 0.1
n_train = 100
n_test = 1000 - n_train
validation = True
validation_period = 150
n_validation = 100
show_plots = False

t_ext = t_end * 4

dict_tl = {
    "goal": True,
    "input": True,
    "obstacle": True,
    "collision_avoidance": True,
    "max_u": 80,
    "obstacle_pos": torch.tensor([[2., 0],[-2, 0]]),  # n_obstacles x 2
    "obstacle_radius": torch.tensor([[1.7], [1.7]]),  # n_obstacles x 1
    "wall_up": True,
}
dict_sum_loss = {
    "state": True,
    "input": False,
    "obstacle": False,
    "collision_avoidance": False,
    "wall_up": False,
    "wall_up_barrier": False,
}

dict_tl['ca_distance'] = min_dist
dict_sum_loss['ca_distance'] = min_dist
dict_sum_loss["Qy"] = Qy
dict_sum_loss['alpha_x'] = torch.tensor([1.])
dict_sum_loss['alpha_u'] = torch.tensor([alpha_u])
dict_sum_loss['alpha_obst'] = torch.tensor([alpha_obst])
dict_sum_loss['alpha_ca'] = torch.tensor([alpha_ca])
dict_sum_loss['alpha_wall'] = torch.tensor([5e0])
dict_sum_loss['alpha_barrier'] = torch.tensor([50 * 5e0])
dict_sum_loss['alpha_x'] = dict_sum_loss['alpha_x'] * 2e-5#4e-5 #1e-4#

# # # # # # # # Load data # # # # # # # #
file_path = os.path.join(BASE_DIR, 'data', sys_model)
filename = 'data_' + sys_model + '_stddist' + str(std_dist) + '_agents' + str(n_agents)
filename += '_RS' + str(random_seed) + '.pkl'
filename = os.path.join(file_path, filename)
if not os.path.isfile(filename):
    generate_data(sys_model, t_end*4, n_agents, random_seed, std_dist=std_dist)
assert os.path.isfile(filename)
filehandler = open(filename, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
assert data_saved['t_end'] >= t_end and data_saved['t_end'] >= t_ext
train_v = data_saved['data_v'][:n_train, :, :]
train_d = data_saved['data_d'][:n_train, :, :]
assert train_d.shape[0] == n_train and train_v.shape[0] == n_train
test_v = data_saved['data_v'][n_train:, :, :]
test_d = data_saved['data_d'][n_train:, :, :]
assert test_v.shape[0] == n_test and test_d.shape[0] == n_test
validation_v = data_saved['data_v'][n_train:n_train+n_validation, :, :]
validation_d = data_saved['data_d'][n_train:n_train+n_validation, :, :]
assert validation_v.shape[0] == n_validation and validation_d.shape[0] == n_validation

# # # # # # # # Set up logger # # # # # # # #
log_name = sys_model
log_name += '_col_av' if alpha_ca else ''
log_name += '_obstacle' if alpha_obst else ''
log_name += '_lin' if linear else '_nonlin'
now = datetime.now().strftime("%m_%d_%H_%Ms")
filename_log = os.path.join(BASE_DIR, 'log')
if not os.path.exists(filename_log):
    os.makedirs(filename_log)
filename_log = os.path.join(filename_log, log_name+'_log' + now)

logging.basicConfig(filename=filename_log, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger(sys_model)
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# # # # # # # # Define models # # # # # # # #
sys = SystemRobots(xbar, linear)
ctl = Controller(sys.f, sys.g, sys.n, sys.m, n_xi, l, use_sp=use_sp, t_end_sp=t_end, std_ini_param=std_ini_param)

# # # # # # # # Define optimizer and parameters # # # # # # # #
optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)

# # # # # # # # Figures # # # # # # # #
file_path = os.path.join(BASE_DIR, 'figures', 'temp')
filename_figure = 'fig_' + log_name
filename_figure = os.path.join(BASE_DIR, 'figures', 'temp', filename_figure)

# Plot before training
# Extended time
x_log = torch.zeros(t_ext, sys.n)
u_log = torch.zeros(t_ext, sys.m)
y_log = torch.zeros(t_ext, sys.m)
v_in = torch.zeros(t_ext + 1, sys.m)
d_in = torch.zeros(t_ext + 1, sys.m)
x = x0.detach()
x_ctl = x0.detach()
xi = torch.ones(ctl.psi_u.n_xi)
u = torch.zeros(sys.m)
for t in range(t_ext):
    x, y = sys(t, x, u, v_in[t, :], d_in[t, :])
    omega = (y, u, x_ctl)
    u, xi, x_ctl = ctl(t, y, xi, omega)
    x_log[t, :] = x.detach()
    u_log[t, :] = u.detach()
    y_log[t, :] = y.detach()
plot_trajectories(x_log, xbar, sys.n_agents, text="CL - before training - extended t", T=t_end, obst=alpha_obst)
if show_plots:
    plt.show()
else:
    plt.savefig(filename_figure + 'before_training' + '.png', format='png')
    plt.close()

# # # # # # # # Training # # # # # # # #
msg = "\n------------ Begin training ------------\n"
msg += "Problem: " + sys_model + " -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate
msg += " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj + " -- std_dist: %.2f\n" % std_dist
msg += " -- alpha_u: %.4f" % alpha_u + " -- alpha_ca: %i" % alpha_ca + " -- alpha_obst: %.1e\n" % alpha_obst
msg += " -- alpha_sp: %.1f" % alpha_sp
msg += "REN info -- n_xi: %i" % n_xi + " -- l: %i " % l + "use_sp: %r\n" % use_sp
msg += "--------- --------- ---------  ---------"
logger.info(msg)

for epoch in range(epochs):
    # batch data
    if n_traj == 1:
        train_v_batch = train_v[epoch % n_train:epoch % n_train + 1, :]
        train_d_batch = train_d[epoch % n_train:epoch % n_train + 1, :]
    else:
        inds = torch.randperm(n_train)[:n_traj]
        # NOTE: use ranperm instead of randint to avoid repeated samples in a batch
        train_v_batch = train_v[inds, :, :]
        train_d_batch = train_d[inds, :, :]
    optimizer.zero_grad()
    loss_tl, loss_sum = 0, 0
    for kk in range(n_traj):
        u = torch.zeros(sys.m)
        x = x0.detach()  # + noise
        x_ctl = x0.detach()   # + noise  # Here we do model mistmatch!
        xi = torch.ones(ctl.psi_u.n_xi)
        x_complete = torch.zeros(0, sys.n)
        u_complete = torch.zeros(0, sys.m)
        y_complete = torch.zeros(0, sys.m)
        for t in range(t_end):
            x, y = sys(t, x, u, train_v_batch[kk, t, :], train_d_batch[kk, t, :])
            omega = (y, u, x_ctl)
            u, xi, x_ctl = ctl(t, y, xi, omega)
            x_complete = torch.cat([x_complete, x.unsqueeze(0)], dim=0)
            u_complete = torch.cat([u_complete, u.unsqueeze(0)], dim=0)
            y_complete = torch.cat([y_complete, y.unsqueeze(0)], dim=0)
        loss_tl = loss_tl + torch.maximum(loss_TL_waypoints(x_complete[2:, :]),
                                          f_loss_tl(x_complete, u_complete, sys, dict_tl))
        loss_sum = loss_sum + f_loss_sum(x_complete, u_complete, sys, dict_sum_loss)
    loss_sp = alpha_sp*torch.max(torch.abs(ctl.sp.u))
    loss = loss_tl + loss_sum + loss_sp
    msg = "Epoch: {:>4d} --- Loss: {:>9.4f} ---||--- Loss tl: {:>9.4f}".format(epoch, loss, loss_tl)
    msg += " --- Loss sum: {:>9.4f} --- Loss sp: {:>9.4f}".format(loss_sum, loss_sp)
    loss.backward()
    optimizer.step()
    ctl.psi_u.set_model_param()
    logger.info(msg)
    if validation and epoch % validation_period == 0:
        # Extended time
        x_log = torch.zeros(t_ext, sys.n)
        u_log = torch.zeros(t_ext, sys.m)
        y_log = torch.zeros(t_ext, sys.m)
        v_in = test_v[0,:,:]
        d_in = test_d[0,:,:]
        x = x0.detach()
        x_ctl = x0.detach()
        xi = torch.ones(ctl.psi_u.n_xi)
        u = torch.zeros(sys.m)
        for t in range(t_ext):
            x, y = sys(t, x, u, v_in[t, :], d_in[t, :])
            omega = (y, u, x_ctl)
            u, xi, x_ctl = ctl(t, y, xi, omega)
            x_log[t, :] = x.detach()
            u_log[t, :] = u.detach()
            y_log[t, :] = y.detach()
        plot_trajectories(x_log, xbar, sys.n_agents, text="CL at epoch %i" % epoch, T=t_end, obst=alpha_obst)
        if show_plots:
            plt.show()
        else:
            plt.savefig(filename_figure + 'during_' + '%i_epoch' % epoch + '.png', format='png')
            plt.close()

# # # # # # # # Save trained model # # # # # # # #
fname = log_name + '_T' + str(t_end) + '_stddist' + str(std_dist) + '_RS' + str(random_seed)
fname += '.pt'
filename = os.path.join(BASE_DIR, 'trained_models')
if not os.path.exists(filename):
    os.makedirs(filename)
filename = os.path.join(filename, fname)
save_dict = {'psi_u': ctl.psi_u.state_dict(),
             'dict_sum_loss': dict_sum_loss,
             'dict_tl':dict_tl,
             'n_xi': n_xi,
             'l': l,
             'n_traj': n_traj,
             'epochs': epochs,
             'std_ini_param': std_ini_param,
             'use_sp': use_sp,
             'linear': linear
             }
if use_sp:
    save_dict['sp'] = ctl.sp.state_dict()
torch.save(save_dict, filename)
logger.info('[INFO] Saved trained model as: %s' % fname)

# # # # # # # # Print & plot results # # # # # # # #
x_log = torch.zeros(t_end, sys.n)
u_log = torch.zeros(t_end, sys.m)
y_log = torch.zeros(t_end, sys.m)
v_in = torch.zeros(t_end, sys.m)
d_in = torch.zeros(t_end, sys.m)
x = x0.detach()
x_ctl = x0.detach()
xi = torch.ones(ctl.psi_u.n_xi)
u = torch.zeros(sys.m)
for t in range(t_end):
    x, y = sys(t, x, u, v_in[t, :], d_in[t, :])
    omega = (y, u, x_ctl)
    u, xi, x_ctl = ctl(t, y, xi, omega)
    x_log[t, :] = x.detach()
    u_log[t, :] = u.detach()
    y_log[t, :] = y.detach()
plot_traj_vs_time(t_end, sys.n_agents, x_log, u_log)
# Number of collisions
n_coll = calculate_collisions(x_log, sys, min_dist)
print("Number of collisions after training: %d" % n_coll)
# Extended time
x_log = torch.zeros(t_ext, sys.n)
u_log = torch.zeros(t_ext, sys.m)
y_log = torch.zeros(t_ext, sys.m)
v_in = test_v[1, :, :]
d_in = test_d[1, :, :]
x = x0.detach()
x_ctl = x0.detach()
xi = torch.ones(ctl.psi_u.n_xi)
u = torch.zeros(sys.m)
for t in range(t_ext):
    x, y = sys(t, x, u, v_in[t, :], d_in[t, :])
    omega = (y, u, x_ctl)
    u, xi, x_ctl = ctl(t, y, xi, omega)
    x_log[t, :] = x.detach()
    u_log[t, :] = u.detach()
    y_log[t, :] = y.detach()
plot_trajectories(x_log, xbar, sys.n_agents, text="CL - after training - extended t", T=t_end, obst=alpha_obst)
if show_plots:
    plt.show()
else:
    plt.savefig(filename_figure + 'trained' + '.png', format='png')
    plt.close()

print("Hola!")
