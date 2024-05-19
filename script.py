import torch
import matplotlib.pyplot as plt
import os
import pickle
import logging
from datetime import datetime

from src.models import SystemRobots, Controller
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_functions import f_loss_u, f_loss_ca, f_loss_obst, f_loss_outputs, f_loss_side
from src.utils import calculate_collisions, set_params, generate_data
from src.utils import WrapLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random_seed = 3
torch.manual_seed(random_seed)

sys_model = "corridor"
prefix = ''

# # # # # # # # Parameters and hyperparameters # # # # # # # #
params = set_params(sys_model)
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Qy, \
alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini = params
n_xi,l = 8,8
n_traj = 1
epochs = 12000
std_ini = 0.2
learning_rate = 5e-3  # 3e-2
std_ini_param = 0.1
epoch_print = 20
use_sp = False
n_train = 100
n_test = 1000 - n_train
validation = True
validation_period = 50
n_validation = 100
alpha_side = 500

show_plots = False

t_ext = t_end * 4

# # # # # # # # Load data # # # # # # # #
file_path = os.path.join(BASE_DIR, 'data', sys_model)
filename = 'data_' + sys_model + '_stdini' + str(std_ini) + '_agents' + str(n_agents)
filename += '_RS' + str(random_seed) + '.pkl'
filename = os.path.join(file_path, filename)
if not os.path.isfile(filename):
    generate_data(sys_model, t_end, n_agents, random_seed, std_ini=std_ini)
assert os.path.isfile(filename)
filehandler = open(filename, 'rb')
data_saved = pickle.load(filehandler)
filehandler.close()
train_x0 = data_saved['data_x0'][:n_train, :]
assert train_x0.shape[0] == n_train
test_x0 = data_saved['data_x0'][n_train:, :]
assert test_x0.shape[0] == n_test
validation_x0 = data_saved['data_x0'][n_train:n_train+n_validation, :]
assert validation_x0.shape[0] == n_validation

# # # # # # # # Set up logger # # # # # # # #
log_name = prefix + sys_model
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
fig_path = os.path.join(BASE_DIR, 'figures', 'temp')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
filename_figure = 'fig_' + log_name
filename_figure = os.path.join(fig_path, filename_figure)

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
msg += " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj + " -- std_ini: %.2f\n" % std_ini
msg += " -- alpha_u: %.4f" % alpha_u + " -- alpha_ca: %i" % alpha_ca + " -- alpha_obst: %.1e\n" % alpha_obst
msg += "REN info -- n_xi: %i" % n_xi + " -- l: %i " % l + "use_sp: %r\n" % use_sp
msg += "--------- --------- ---------  ---------"
logger.info(msg)
best_valid_loss = 1e9
best_params = None
best_params_sp = None
for epoch in range(epochs):
    # batch data
    if n_traj == 1:
        train_x0_batch = train_x0[epoch%n_train:epoch%n_train+1, :]
    else:
        inds = torch.randperm(n_train)[:n_traj]
        train_x0_batch = train_x0[inds, :]
    optimizer.zero_grad()
    loss_y, loss_u, loss_ca, loss_obst, loss_side = 0, 0, 0, 0, 0
    for kk in range(n_traj):
        v_in = torch.zeros(t_end, sys.m)
        d_in = torch.zeros(t_end, sys.m)
        u = torch.zeros(sys.m)
        x = x0.detach() + train_x0_batch[kk]
        x_ctl = x0.detach()  # Here we do model mistmatch!
        xi = torch.ones(ctl.psi_u.n_xi)
        for t in range(t_end):
            x, y = sys(t, x, u, v_in[t, :], d_in[t, :])
            omega = (y, u, x_ctl)
            u, xi, x_ctl = ctl(t, y, xi, omega)
            loss_y = loss_y + f_loss_outputs(t, y, sys, Qy)/n_traj
            loss_u = loss_u + alpha_u * f_loss_u(t, u)/n_traj
            loss_ca = loss_ca + alpha_ca * f_loss_ca(x, sys, min_dist)/n_traj
            loss_side = loss_side + alpha_side * f_loss_side(x)/n_traj
            if alpha_obst != 0:
                loss_obst = loss_obst + alpha_obst * f_loss_obst(x)/n_traj
    loss = loss_y + loss_u + loss_ca + loss_obst + loss_side
    msg = "Epoch: {:>4d} --- Loss: {:>9.4f} ---||--- Loss y: {:>9.2f}".format(epoch, loss/t_end, loss_y)
    msg += " --- Loss u: {:>9.4f} --- Loss ca: {:>9.2f} --- Loss obst: {:>9.2f}".format(loss_u, loss_ca, loss_obst)
    msg += " --- Loss side: {:>9.2f}".format(loss_side)
    loss.backward()
    optimizer.step()
    ctl.psi_u.set_model_param()
    # record state dict if best on valid
    if validation and epoch % validation_period == 0 and epoch>0:
        with torch.no_grad():
            loss_y, loss_u, loss_ca, loss_obst, loss_side = 0, 0, 0, 0, 0
            for kk in range(n_validation):
                v_in = torch.zeros(t_end, sys.m)
                d_in = torch.zeros(t_end, sys.m)
                u = torch.zeros(sys.m)
                x = x0.detach() + validation_x0[kk]
                x_ctl = x0.detach()  # Here we do model mistmatch!
                xi = torch.ones(ctl.psi_u.n_xi)
                for t in range(t_end):
                    x, y = sys(t, x, u, v_in[t, :], d_in[t, :])
                    omega = (y, u, x_ctl)
                    u, xi, x_ctl = ctl(t, y, xi, omega)
                    loss_y = loss_y + f_loss_outputs(t, y, sys, Qy)/n_validation
                    loss_u = loss_u + alpha_u * f_loss_u(t, u)/n_validation
                    loss_ca = loss_ca + alpha_ca * f_loss_ca(x, sys, min_dist)/n_validation
                    loss_side = loss_side + alpha_side * f_loss_side(x)/n_validation
                    if alpha_obst != 0:
                        loss_obst = loss_obst + alpha_obst * f_loss_obst(x)/n_validation
            loss = loss_y + loss_u + loss_ca + loss_obst + loss_side
        msg += ' ---||--- Original validation loss: %.2f' % (loss/t_end)
        # compare with the best valid loss
        if loss < best_valid_loss:
            best_valid_loss = loss
            best_params = ctl.psi_u.state_dict()
            if use_sp:
                best_params_sp = ctl.sp.state_dict()
            msg += ' (best so far)'
    logger.info(msg)
    if (epoch < epoch_print and epoch % 2 == 0) or epoch % validation_period == 0:
        # Extended time
        x_log = torch.zeros(t_ext, sys.n)
        u_log = torch.zeros(t_ext, sys.m)
        y_log = torch.zeros(t_ext, sys.m)
        v_in = torch.zeros(t_ext + 1, sys.m)
        d_in = torch.zeros(t_ext + 1, sys.m)
        x = x0.detach() + train_x0_batch[0]
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
        plot_trajectories(x_log, xbar, sys.n_agents, text="CL at epoch %i" % epoch, T=t_end, obst=False)
        if show_plots:
            plt.show()
        else:
            plt.savefig(filename_figure + 'during_' + '%i_epoch' % epoch + '.png', format='png')
            plt.close()

# Set parameters to the best seen during training
if validation and best_params is not None:
    ctl.psi_u.load_state_dict(best_params)
    ctl.psi_u.eval()
    if use_sp:
        ctl.sp.load_state_dict(best_params_sp)
        ctl.sp.eval()
    ctl.psi_u.set_model_param()

# # # # # # # # Save trained model # # # # # # # #
fname = log_name + '_T' + str(t_end) + '_stdini' + str(std_ini) + '_RS' + str(random_seed)
fname += '.pt'
filename = os.path.join(BASE_DIR, 'trained_models')
if not os.path.exists(filename):
    os.makedirs(filename)
filename = os.path.join(filename, fname)
save_dict = {'psi_u': ctl.psi_u.state_dict(),
             'Q': Qy,
             'alpha_u': alpha_u,
             'alpha_ca': alpha_ca,
             'alpha_obst': alpha_obst,
             'alpha_side': alpha_side,
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
msg = 'Number of collisions after training: %.1f.' % n_coll
logger.info(msg)

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
plot_trajectories(x_log, xbar, sys.n_agents, text="CL - after training - extended t", T=t_end, obst=alpha_obst)
if show_plots:
    plt.show()
else:
    plt.savefig(filename_figure + 'trained' + '.png', format='png')
    plt.close()

print("Hola!")

