import torch
import torch.nn.functional as F


def f_loss_outputs(t, y, sys, Q=None):
    gamma = 1
    if Q is None:
        Q = torch.eye(sys.m)
    ybar = F.linear(sys.xbar, sys.C)
    dy = y - ybar
    yQy = F.linear(dy, Q) * dy
    return yQy.sum()  # * (gamma**(100-t))

def f_loss_states(t, x, sys, Q=None):
    gamma = 1
    if Q is None:
        Q = torch.eye(sys.n)
    xbar = sys.xbar
    dx = x - xbar
    xQx = F.linear(dx, Q) * dx
    return xQx.sum()  # * (gamma**(100-t))


def f_loss_u(t, u):
    loss_u = (u ** 2).sum()
    return loss_u


def f_loss_ca(x, sys, min_dist=0.5):
    min_sec_dist = min_dist + 0.2
    # collision avoidance:
    deltaqx = x[0::4].repeat(sys.n_agents, 1) - x[0::4].repeat(sys.n_agents, 1).transpose(0, 1)
    deltaqy = x[1::4].repeat(sys.n_agents, 1) - x[1::4].repeat(sys.n_agents, 1).transpose(0, 1)
    distance_sq = deltaqx ** 2 + deltaqy ** 2
    mask = torch.logical_not(torch.eye(sys.n // 4))
    loss_ca = (1/(distance_sq + 1e-3) * (distance_sq.detach() < (min_sec_dist ** 2)) * mask).sum()/2
    return loss_ca


def normpdf(q, mu, cov):
    d = 2
    mu = mu.view(1, d)
    cov = cov.view(1, d)  # the diagonal of the covariance matrix
    qs = torch.split(q, 2)
    out = torch.tensor(0)
    for qi in qs:
        # if qi[1]<1.5 and qi[1]>-1.5:
        den = (2*torch.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
        num = torch.exp((-0.5 * (qi - mu)**2 / cov).sum())
        out = out + num/den
    return out


def f_loss_obst(x, sys=None):
    qx = x[::4].unsqueeze(1)
    qy = x[1::4].unsqueeze(1)
    n_agents = qx.shape[0]
    q = torch.cat((qx,qy), dim=1).view(1,-1).squeeze()
    # mu1 = torch.tensor([[-2, 0]])
    # mu2 = torch.tensor([[2, 0.0]])
    # cov = torch.tensor([[1, 0.2]])
    mu1 = torch.tensor([[-2.5, 0]])
    mu2 = torch.tensor([[2.5, 0.0]])
    mu3 = torch.tensor([[-1.5, 0.0]])
    mu4 = torch.tensor([[1.5, 0.0]])
    cov = torch.tensor([[0.2, 0.2]])
    Q1 = normpdf(q, mu=mu1, cov=cov)
    Q2 = normpdf(q, mu=mu2, cov=cov)
    Q3 = normpdf(q, mu=mu3, cov=cov)
    Q4 = normpdf(q, mu=mu4, cov=cov)
    # QQ = (Q1 + Q2).sum()
    # bias = 0.0965*n_agents
    # return torch.relu(QQ - bias)
    QQ = (Q1 + Q2 + Q3 + Q4).sum()
    mask = QQ > 0.007  # *n_agents
    return QQ * mask


def f_loss_side(x):
    qx = x[::4]
    qy = x[1::4]
    side = torch.relu(qx - 2.5) + torch.relu(-2.5 - qx)  # + torch.relu(qy - 6) + torch.relu(-6 - qy)
    return side.sum()
