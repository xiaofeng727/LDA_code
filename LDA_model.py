import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LDA(torch.nn.Module):
    def __init__(self, LayerNo, PhaseNo):
        super(LDA, self).__init__()

        # soft threshold
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        # sparcity bactracking
        self.gamma = 1.0
        # a parameter for backtracking
        self.sigma = 15000.0
        # parameter for activation function
        self.delta = 0.01
        # set phase number
        self.PhaseNo = PhaseNo
        self.init = True

        self.alphas = nn.Parameter(0.5 * torch.ones(LayerNo))
        self.betas = nn.Parameter(0.1 * torch.ones(LayerNo))

        # size: out channels  x in channels x filter size x filter size
        # every block shares weights
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

    def set_PhaseNo(self, PhaseNo):
        # used when adding more phases
        self.PhaseNo = PhaseNo

    def set_init(self, init):
        self.init = init

    def activation(self, x):
        """ activation function from eq. (33) in paper """

        # index for x < -delta and x > delta
        index = torch.sign(F.relu(torch.abs(x) - self.delta))
        output = index * F.relu(x)
        # add parts when -delta <= x <= delta
        output += (1 - index) * (1 / (4 * self.delta) * torch.square(x) + 1 / 2 * x + self.delta / 4)
        return output

    def activation_der(self, x):
        """ derivative of activation function from eq. (33) in paper """

        # index for x < -delta and x > delta
        index = torch.sign(F.relu(torch.abs(x) - self.delta))
        output = index * torch.sign(F.relu(x))
        # add parts when -delta <= x <= delta
        output += (1 - index) * (1 / (2 * self.delta) * x + 1 / 2)
        return output

    def grad_r(self, x):
        """ implementation of eq. (10) in paper  """

        # first obtain forward passs to get features g_i, i = 1, 2, ..., n_c
        # This is the feature extraction map, we can change it to other networks
        # x_input: n x 1 x 33 x 33
        x_input = x.view(-1, 1, 33, 33)
        soft_thr = self.soft_thr * self.gamma
        # shape from input to output: batch size x height x width x n channels
        x1 = F.conv2d(x_input, self.conv1, padding=1)  # (batch,  1, 33, 33) -> (batch, 32, 33, 33)
        x2 = F.conv2d(self.activation(x1), self.conv2, padding=1)  # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        x3 = F.conv2d(self.activation(x2), self.conv3, padding=1)  # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        g = F.conv2d(self.activation(x3), self.conv4, padding=1)  # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        n_channel = g.shape[1]

        # compute norm over channel and compute g_factor
        norm_g = torch.norm(g, dim=1)
        I1 = torch.sign(F.relu(norm_g - soft_thr))[:, None, :, :]
        I1 = torch.tile(I1, [1, n_channel, 1, 1])
        I0 = 1 - I1

        g_factor = I1 * F.normalize(g, dim=1) + I0 * g / soft_thr

        # implementation for eq. (9): multiply grad_g to g_factor from the left
        # result derived from chain rule and that gradient of convolution is convolution transpose
        g_r = F.conv_transpose2d(g_factor, self.conv4, padding=1)
        g_r *= self.activation_der(x3)
        g_r = F.conv_transpose2d(g_r, self.conv3, padding=1)
        g_r *= self.activation_der(x2)
        g_r = F.conv_transpose2d(g_r, self.conv2, padding=1)
        g_r *= self.activation_der(x1)
        g_r = F.conv_transpose2d(g_r, self.conv1, padding=1)

        return g_r.reshape(-1, 1089)

    def R(self, x):
        """ implementation of eq. (9) in paper: the smoothed regularizer  """

        # first obtain forward passs to get features g_i, i = 1, 2, ..., n_c
        # x_input: n x 1 x 33 x 33
        x_input = x.view(-1, 1, 33, 33)
        soft_thr = self.soft_thr * self.gamma

        # shape from input to output: batch size x height x width x n channels
        x1 = F.conv2d(x_input, self.conv1, padding=1)  # (batch,  1, 33, 33) -> (batch, 32, 33, 33)
        x2 = F.conv2d(self.activation(x1), self.conv2, padding=1)  # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        x3 = F.conv2d(self.activation(x2), self.conv3, padding=1)  # (batch, 32, 33, 33) -> (batch, 32, 33, 33)
        g = F.conv2d(self.activation(x3), self.conv4, padding=1)  # (batch, 32, 33, 33) -> (batch, 32, 33, 33)

        norm_g = torch.norm(g, dim=1)
        I1 = torch.sign(F.relu(norm_g - soft_thr))
        I0 = 1 - I1

        r = 1 / (2 * soft_thr) * torch.square(norm_g) * I0 + (norm_g - soft_thr) * I1
        r = r.reshape(-1, 1089)

        r = torch.sum(r, -1, keepdim=True)

        return r

    def phi(self, x, y, Phi):
        """ The implementation for the loss function """
        # x is the reconstruction result
        # y is the ground truth

        r = self.R(x)

        # print("r.shape",r.shape)
        # print(y.shape)
        f = 1 / 2 * torch.sum(torch.square(x @ torch.transpose(Phi, 0, 1) - y),
                              dim=1, keepdim=True)

        # print("f.shape",f.shape)
        # a=torch.square(x @ torch.transpose(Phi, 0, 1) - y)

        return f + r

    def phase(self, x, PhiTPhi, PhiTb, y, Phi, phase):
        """
        x is the reconstruction output from last phase
        y is Phi True_x, the sampled ground truth

        """

        alpha = torch.abs(self.alphas[phase])
        beta = torch.abs(self.betas[phase])

        # Implementation of eq. 2/7 (ISTANet paper) Immediate reconstruction
        # here we obtain z (in LDA paper from eq. 12)
        z = x - alpha * torch.mm(x, PhiTPhi)
        z = z + alpha * PhiTb

        # gradient of r, the smoothed regularizer
        grad_r_z = self.grad_r(z)
        tau = alpha * beta / (alpha + beta)
        # u: resnet structure
        u = z - tau * grad_r_z

        if not self.init:
            grad_r_x = self.grad_r(x)
            v = z - alpha * grad_r_x

            """ The rest is to just find out phi(u) and phi(v), which one is smaller """
            phi_u = self.phi(u, y, Phi)
            phi_v = self.phi(v, y, Phi)

            u_ind = torch.sign(F.relu(phi_v - phi_u))

            v_ind = 1 - u_ind

            x_next = u_ind * u + v_ind * v

        else:
            x_next = u
            self.init = False

            """ update soft threshold, step 7-8 algorithm 1 """
        norm_grad_phi_x_next = \
            torch.norm(
                (x_next @ PhiTPhi - PhiTb) + \
                self.grad_r(x_next),
                dim=-1, keepdim=True
            )
        sig_gam_eps = self.sigma * self.gamma * self.soft_thr
        self.gamma *= 0.9 if (torch.mean(norm_grad_phi_x_next) < sig_gam_eps) else 1.0

        return x_next

    def forward(self, Phix, Phi, Qinit):

        self.gamma = 1.0
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        for phase in range(self.PhaseNo):
            x = self.phase(x, PhiTPhi, PhiTb, Phix, Phi, phase)

        x_final = x

        return x_final