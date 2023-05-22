import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class EncoderAC_v2(nn.Module):
    def __init__(self, num_inputs, actor_dim, hidden_dim):
        super(EncoderAC_v2, self).__init__()
        dropout_rate = 0.2
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.linear4 = nn.Linear(hidden_dim + 2, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(p=dropout_rate)
        self.linear5 = nn.Linear(num_inputs, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.dropout5 = nn.Dropout(p=dropout_rate)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.critic1 = nn.Sequential(
            nn.Linear(hidden_dim + 10 + 2, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + 10 + 2, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weights_init_1)

    def global_encoder(self, states):
        # encode = torch.zeros(self.hidden_dim)
        nums = len(states)
        if np.shape(states) == (nums,):
            states = [states]
            encode = torch.zeros(self.hidden_dim)
        else:
            nums = np.shape(states)[1]
            bs = np.shape(states)[0]
            encode = torch.zeros(bs, self.hidden_dim)
        for i in range(nums - 1):
            state = [uav[i] for uav in states]
            state = torch.FloatTensor(state)
            # x = F.leaky_relu(self.bn1(self.linear1(state)))
            x = F.leaky_relu(self.linear1(state))
            # x = self.dropout1(x)
            # x = F.leaky_relu(self.bn2(self.linear2(x)))
            x = F.leaky_relu(self.linear2(x))
            # x = self.dropout2(x)
            encode = encode + x
        distance = [uav[-1] for uav in states]
        distance = torch.FloatTensor(distance)

        encode = torch.cat([encode, distance], 1)
        # encode = F.leaky_relu(self.bn3(self.linear4(encode)))
        encode = F.leaky_relu(self.linear4(encode))
        # encode = self.dropout3(encode)
        return encode

    def num_pred(self, states):
        encode = self.global_encoder(states)
        predict = self.pred(encode)
        return predict

    def local_encoder(self, state):
        # local_info = F.leaky_relu(self.bn4(self.linear5(state)))
        local_info = F.leaky_relu(self.linear5(state))
        # local_info = self.dropout4(local_info)
        # local_info = F.leaky_relu(self.bn5(self.linear3(local_info)))
        local_info = F.leaky_relu(self.linear3(local_info))
        # local_info = self.dropout5(local_info)
        return local_info

    def sample(self, states):
        encode = self.global_encoder(states)
        alphas = []
        nums = len(states)
        if np.shape(states) == (nums,):
            states = [states]
        else:
            nums = np.shape(states)[1]

        for i in range(nums - 1):
            state = [uav[i] for uav in states]
            state = torch.FloatTensor(state)
            local_info = self.local_encoder(state)
            sum_info = torch.cat([encode, local_info], 1)
            alpha = self.actor(sum_info) + 1
            if alphas == []:
                alphas = alpha
            else:
                alphas = torch.cat([alphas, alpha], 1)

        distribution = Dirichlet(alphas)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)
        log_prob = log_prob.unsqueeze(1)
        mean = alphas / torch.sum(alphas, dim=1, keepdim=True)
        return action, log_prob, mean, alphas,

    def q_value(self, states, action):
        encode = self.global_encoder(states)
        distance = [uav[-1] for uav in states]
        distance = torch.FloatTensor(distance)
        encode = torch.cat([encode, distance], 1)
        padded_action = pad_to_10(action)
        xu = torch.cat([encode, padded_action], 1)
        x1 = self.critic1(xu)
        x2 = self.critic2(xu)
        return x1, x2

class EncoderAC_v3(nn.Module):
    def __init__(self, num_inputs, actor_dim, hidden_dim):
        super(EncoderAC_v3, self).__init__()
        dropout_rate = 0.2
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.linear4 = nn.Linear(hidden_dim + 2, hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.dropout4 = nn.Dropout(p=dropout_rate)
        self.linear5 = nn.Linear(num_inputs, hidden_dim)
        self.bn5 = nn.BatchNorm1d(hidden_dim)
        self.dropout5 = nn.Dropout(p=dropout_rate)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.critic1 = nn.Sequential(
            nn.Linear(hidden_dim + 10 + 2, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + 10 + 2, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

        self.critic3 = nn.Sequential(
            nn.Linear(hidden_dim + 10 + 2, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.critic4 = nn.Sequential(
            nn.Linear(hidden_dim + 10 + 2, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weights_init_1)

    def global_encoder(self, states):
        # encode = torch.zeros(self.hidden_dim)
        nums = len(states)
        if np.shape(states) == (nums,):
            states = [states]
            encode = torch.zeros(self.hidden_dim)
        else:
            nums = np.shape(states)[1]
            bs = np.shape(states)[0]
            encode = torch.zeros(bs, self.hidden_dim)
        for i in range(nums - 1):
            state = [uav[i] for uav in states]
            state = torch.FloatTensor(state)
            # x = F.leaky_relu(self.bn1(self.linear1(state)))
            x = F.leaky_relu(self.linear1(state))
            # x = self.dropout1(x)
            # x = F.leaky_relu(self.bn2(self.linear2(x)))
            x = F.leaky_relu(self.linear2(x))
            # x = self.dropout2(x)
            encode = encode + x
        distance = [uav[-1] for uav in states]
        distance = torch.FloatTensor(distance)

        encode = torch.cat([encode, distance], 1)
        # encode = F.leaky_relu(self.bn3(self.linear4(encode)))
        encode = F.leaky_relu(self.linear4(encode))
        # encode = self.dropout3(encode)
        return encode

    def num_pred(self, states):
        encode = self.global_encoder(states)
        predict = self.pred(encode)
        return predict

    def local_encoder(self, state):
        # local_info = F.leaky_relu(self.bn4(self.linear5(state)))
        local_info = F.leaky_relu(self.linear5(state))
        # local_info = self.dropout4(local_info)
        # local_info = F.leaky_relu(self.bn5(self.linear3(local_info)))
        local_info = F.leaky_relu(self.linear3(local_info))
        # local_info = self.dropout5(local_info)
        return local_info

    def sample(self, states):
        encode = self.global_encoder(states)
        alphas = []
        nums = len(states)
        if np.shape(states) == (nums,):
            states = [states]
        else:
            nums = np.shape(states)[1]

        for i in range(nums - 1):
            state = [uav[i] for uav in states]
            state = torch.FloatTensor(state)
            local_info = self.local_encoder(state)
            sum_info = torch.cat([encode, local_info], 1)
            alpha = self.actor(sum_info) + 1
            if alphas == []:
                alphas = alpha
            else:
                alphas = torch.cat([alphas, alpha], 1)

        distribution = Dirichlet(alphas)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)
        log_prob = log_prob.unsqueeze(1)
        mean = alphas / torch.sum(alphas, dim=1, keepdim=True)
        return action, log_prob, mean, alphas,

    def q_value(self, states, action):
        encode = self.global_encoder(states)
        distance = [uav[-1] for uav in states]
        distance = torch.FloatTensor(distance)
        encode = torch.cat([encode, distance], 1)
        padded_action = pad_to_10(action)
        xu = torch.cat([encode, padded_action], 1)
        x1 = self.critic1(xu)
        x2 = self.critic2(xu)
        x3 = self.critic3(xu)
        x4 = self.critic4(xu)
        return x1, x2, x3, x4
