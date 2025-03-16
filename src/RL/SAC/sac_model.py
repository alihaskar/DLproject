import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer to store and sample experiences."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network for SAC."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256], log_std_min: float = -20, log_std_max: float = 2):
        """
        Initialize Actor network.
        
        Args:
            state_dim: Dimension of input state
            action_dim: Dimension of output actions
            hidden_dims: List of hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(Actor, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        self.layers = nn.ModuleList()
        prev_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Mean and log_std output layers
        self.mean_linear = nn.Linear(prev_dim, action_dim)
        self.log_std_linear = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        x = state
        
        # Pass through hidden layers with ReLU
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Get mean and log_std
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy given a state.
        
        Returns:
            Tuple of (sampled action, log probability, mean action)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        
        # Calculate log probability, accounting for the tanh squashing
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh correction to log_prob
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, torch.tanh(mean)

class Critic(nn.Module):
    """Critic network for SAC that estimates Q-values."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        """
        Initialize Critic network.
        
        Args:
            state_dim: Dimension of input state
            action_dim: Dimension of input actions
            hidden_dims: List of hidden layer dimensions
        """
        super(Critic, self).__init__()
        
        # Build network layers
        self.layers = nn.ModuleList()
        prev_dim = state_dim + action_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Q-value output layer
        self.q_value = nn.Linear(prev_dim, 1)
    
    def forward(self, state, action) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.cat([state, action], dim=1)
        
        # Pass through hidden layers with ReLU
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # Get Q-value
        q = self.q_value(x)
        
        return q

class SACAgent:
    """Soft Actor-Critic agent for trading with transaction cost reduction."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        batch_size: int = 256,
        buffer_capacity: int = 100000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions for networks
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            lr_alpha: Learning rate for entropy coefficient
            gamma: Discount factor
            tau: Target network soft update parameter
            alpha: Initial entropy coefficient
            automatic_entropy_tuning: Whether to automatically tune entropy
            batch_size: Batch size for training
            buffer_capacity: Maximum size of replay buffer
            device: Device to run the models on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.batch_size = batch_size
        self.device = device
        
        logger.info(f"Initializing SAC agent with device: {device}")
        
        # Initialize actor network
        self.actor = Actor(state_dim, action_dim, hidden_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Initialize critic networks (two Q-networks for Double Q-learning)
        self.critic1 = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dims).to(device)
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Initialize target critic networks
        self.critic1_target = Critic(state_dim, action_dim, hidden_dims).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dims).to(device)
        
        # Copy parameters to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Set up automatic entropy tuning
        if self.automatic_entropy_tuning:
            # Target entropy is -dim(action_space) by default
            self.target_entropy = -action_dim
            # Initialize log alpha (entropy coefficient)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        
        self.training_steps = 0
    
    def select_action(self, state, evaluate=False) -> np.ndarray:
        """
        Select an action given a state.
        
        Args:
            state: Current state
            evaluate: If True, use deterministic actions for evaluation
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # Use mean action for evaluation (deterministic)
                _, _, action = self.actor.sample(state)
            else:
                # Sample action from policy for training
                action, _, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> Tuple[float, float, float]:
        """
        Train the agent using experiences from the replay buffer.
        
        Returns:
            Tuple of (actor_loss, critic_loss, alpha_loss)
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        self.training_steps += 1
        
        # Sample batch from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert experiences to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.FloatTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in experiences]).unsqueeze(1).to(self.device)
        
        # Update critic networks
        with torch.no_grad():
            # Sample next actions and their log probs from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Get target Q-values
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            
            # Use minimum Q-value (to reduce overestimation bias)
            next_q = torch.min(next_q1, next_q2)
            
            # Calculate target with entropy term
            target_q = rewards + (1 - dones) * self.gamma * (next_q - self.alpha * next_log_probs)
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        # Calculate critic losses (MSE)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = critic1_loss + critic2_loss
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor network (delayed)
        if self.training_steps % 2 == 0:
            # Sample actions and log probs from current policy
            actions_pi, log_probs_pi, _ = self.actor.sample(states)
            
            # Get Q-values for sampled actions
            q1_pi = self.critic1(states, actions_pi)
            q2_pi = self.critic2(states, actions_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            
            # Calculate actor loss (negative of expected Q-value minus entropy regularization)
            actor_loss = (self.alpha * log_probs_pi - q_pi).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update alpha (entropy coefficient)
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp().item()
            else:
                alpha_loss = torch.tensor(0.0).to(self.device)
        else:
            actor_loss = torch.tensor(0.0).to(self.device)
            alpha_loss = torch.tensor(0.0).to(self.device)
        
        # Update target networks with soft update
        if self.training_steps % 2 == 0:
            self._soft_update_target(self.critic1, self.critic1_target)
            self._soft_update_target(self.critic2, self.critic2_target)
        
        return actor_loss.item(), critic_loss.item(), alpha_loss.item()
    
    def _soft_update_target(self, source, target):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Update dimensions if available in the checkpoint
        if 'state_dim' in checkpoint:
            self.state_dim = checkpoint['state_dim']
        if 'action_dim' in checkpoint:
            self.action_dim = checkpoint['action_dim']
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        
        logger.info(f"Model loaded from {path}") 