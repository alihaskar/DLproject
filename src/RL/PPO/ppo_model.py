import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class ActorNetwork(nn.Module):
    """
    Actor network for PPO that outputs action probabilities.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize the actor network.
        
        Args:
            input_dim: Dimension of input state
            output_dim: Dimension of output actions
            hidden_dims: List of hidden layer dimensions
        """
        super(ActorNetwork, self).__init__()
        
        # Build network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer for action probabilities
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.network(x)
        action_logits = self.output_layer(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
        
class CriticNetwork(nn.Module):
    """
    Critic network for PPO that outputs state value estimates.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize the critic network.
        
        Args:
            input_dim: Dimension of input state
            hidden_dims: List of hidden layer dimensions
        """
        super(CriticNetwork, self).__init__()
        
        # Build network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer for state value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

class PPOMemory:
    """
    Memory buffer for storing trajectories during an episode.
    """
    
    def __init__(self, batch_size: int = 64):
        """
        Initialize memory buffer.
        
        Args:
            batch_size: Batch size for training
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store(self, state, action, prob, val, reward, done):
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            prob: Action probability
            val: State value estimate
            reward: Reward received
            done: Whether episode is done
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        """Clear the memory buffer."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        
    def generate_batches(self):
        """
        Generate batches for training.
        
        Returns:
            List of batch indices
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches
    
    def __len__(self):
        """Return the current size of the memory."""
        return len(self.states)
        
class PPOAgent:
    """
    PPO Agent for trading with transaction cost reduction.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64],
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 64,
        n_epochs: int = 10,
        entropy_coef: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize PPO Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor
            gae_lambda: Lambda for Generalized Advantage Estimation
            policy_clip: Clipping parameter for PPO
            batch_size: Batch size for training
            n_epochs: Number of epochs to train on each batch of data
            entropy_coef: Coefficient for entropy bonus to encourage exploration
            device: Device to use for training (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.device = device
        self.batch_size = batch_size
        
        # Initialize actor and critic networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize memory buffer
        self.memory = PPOMemory(batch_size)
        
    def select_action(self, state, training: bool = True) -> Tuple[int, float, float]:
        """
        Select an action based on current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action, action probability, state value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            if training:
                # Sample action from probability distribution during training
                dist = distributions.Categorical(action_probs)
                action = dist.sample().item()
                prob = action_probs[0, action].item()
            else:
                # Take most likely action during testing for deterministic behavior
                action = torch.argmax(action_probs).item()
                prob = action_probs[0, action].item()
                
            value = value.item()
            
        return action, prob, value
        
    def store_transition(self, state, action, prob, val, reward, done):
        """Store a transition in memory."""
        self.memory.store(state, action, prob, val, reward, done)
        
    def train(self):
        """
        Train the PPO agent using collected experiences.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) == 0:
            return {"actor_loss": 0, "critic_loss": 0, "total_loss": 0}
            
        # Calculate advantages
        states = torch.FloatTensor(self.memory.states).to(self.device)
        actions = torch.LongTensor(self.memory.actions).to(self.device)
        old_probs = torch.FloatTensor(self.memory.probs).to(self.device)
        
        # Training for several epochs
        metrics = {"actor_loss": 0, "critic_loss": 0, "total_loss": 0}
        
        for _ in range(self.n_epochs):
            # Generate advantages for the entire episode
            advantages, returns = self._compute_advantages()
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Generate mini-batches
            batches = self.memory.generate_batches()
            
            # Update policy for each mini-batch
            for batch_indices in batches:
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                old_prob_batch = old_probs[batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]
                
                # Get current action probabilities and state values
                action_probs = self.actor(state_batch)
                dist = distributions.Categorical(action_probs)
                new_probs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()
                state_values = self.critic(state_batch).squeeze()
                
                # Calculate probability ratio
                prob_ratio = torch.exp(new_probs - torch.log(old_prob_batch + 1e-10))
                
                # Calculate surrogate losses
                weighted_advantages = advantage_batch
                surrogate1 = prob_ratio * weighted_advantages
                surrogate2 = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * weighted_advantages
                
                # Calculate losses
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                critic_loss = F.mse_loss(state_values, return_batch)
                
                # Add entropy bonus to encourage exploration
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                # Backpropagate
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Record metrics
                metrics["actor_loss"] += actor_loss.item()
                metrics["critic_loss"] += critic_loss.item()
                metrics["total_loss"] += total_loss.item()
        
        # Average metrics across epochs and batches
        batch_count = len(batches) * self.n_epochs
        metrics["actor_loss"] /= batch_count
        metrics["critic_loss"] /= batch_count
        metrics["total_loss"] /= batch_count
        
        # Clear memory after update
        self.memory.clear()
        
        return metrics
        
    def _compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages, returns
        """
        rewards = np.array(self.memory.rewards)
        values = np.array(self.memory.vals)
        dones = np.array(self.memory.dones)
        
        # Initialize arrays
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Final state value is 0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        return advantages, returns
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        logger.info(f"Model loaded from {path}") 