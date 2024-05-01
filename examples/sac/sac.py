import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
import numpy as np
from torch.nn import KLDivLoss

kl_div = KLDivLoss(reduction='batchmean')


class SAC(object):
    def __init__(self, num_inputs, action_space, args, eval_batch=None, CUP_flag=False, other_policies=None,own_idx=1):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.hidden_size = args.hidden_size

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.eval_batch = eval_batch

        self.guided_policy = CUP_flag

        self.action_space = action_space

        self.kl_scale = args.kl_scale

        self.other_policy_list = other_policies

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

            # # save initial policy
            if not os.path.exists(f"policy_{own_idx}.pth"):
                policy_dict = self.policy.state_dict()
                torch.save(policy_dict, f"policy_{own_idx}.pth")
            else:
                print(f"policy_{own_idx}.pth already exists")


        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def _KL(self, p, q):
        p = p.detach().cpu().numpy()
        q = q.detach().cpu().numpy()

        a = []
        b = []

        for i in range(len(p)):
            if p[i] == 0:
                p[i] = 1e-8
            if q[i] == 0:
                q[i] = 1e-8
            a.append(p[i].item())
            b.append(q[i].item())

        a = np.array(a)
        b = np.array(b)

        probs_a = np.exp(a) / np.sum(np.exp(a))
        probs_b = np.exp(b) / np.sum(np.exp(b))

        kl = np.sum(probs_a * np.log(probs_a / probs_b))
        kl = np.mean(kl)

        return kl

            

    def compute_KL_score(self,
                    other_policies=None,
                    eval_batch=None,
                    num_inputs=None,
                    hidden_size=None,
                    action_space=None,
                   ):
        if other_policies is None:
            raise ValueError("other_policies cannot be None")
        if eval_batch is None:
            raise ValueError("eval_batch cannot be None")
        
        import numpy as np

        states = np.copy(eval_batch)
        states = torch.FloatTensor(states).to(self.device)

        advantages = []

        
        for p in other_policies:
            temp_policy = GaussianPolicy(num_inputs, self.action_space.shape[0], hidden_size, self.action_space).to(self.device)
            policy_dict = torch.load(p)
            temp_policy.load_state_dict(policy_dict)
            temp_policy.eval()
            pi,log_pi, _ = temp_policy.sample(states)
            with torch.no_grad():
                qf1_pi, qf2_pi = self.critic(states, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                EA = min_qf_pi - self.alpha * log_pi
                EA = EA.mean()
                advantages.append(EA.cpu().numpy())
        
        max_idx = np.argmax(advantages)

        best_policy_idx = other_policies[max_idx]
        best_policy = GaussianPolicy(num_inputs, self.action_space.shape[0], hidden_size, self.action_space).to(self.device)
        policy_dict = torch.load(best_policy_idx)
        best_policy.load_state_dict(policy_dict)

        curr_actions_prob = self.policy.sample(states)[1]
        best_actions_prob = best_policy.sample(states)[1]

        KL = self._KL(curr_actions_prob,best_actions_prob)
        return KL


    def select_action(self, state, evaluate=False):
        if type(state)==tuple:
            state = state[0]
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates, guided_itr=False):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        KL = 0
        curr_mean = [0,0]
        curr_std = [0,0]
            
        if self.guided_policy and guided_itr and self.kl_scale > 1e-2:
            KL = self.compute_KL_score(other_policies=self.other_policy_list, eval_batch=self.eval_batch, num_inputs=state_batch.shape[1], hidden_size=self.hidden_size, action_space=action_batch)
            policy_loss += KL*self.kl_scale
            curr_mean = self.policy.last_mean
            curr_std = self.policy.last_std
        elif guided_itr and self.kl_scale < 1e-2:
            curr_mean = self.policy.last_mean
            curr_std = self.policy.last_std




        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), KL, curr_mean, curr_std

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

