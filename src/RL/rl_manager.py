from .DQN.train_dqn import DQNTrainer
from .PPO.train_ppo import PPOTrainer
from .SAC.train_sac import SACTrainer

class RLManager:
    def __init__(self, data_path: str):
        """
        Initialize RL manager with data path
        
        Args:
            data_path: Path to the input data CSV file
        """
        self.data_path = data_path
        self._dqn_trainer = None
        self._ppo_trainer = None
        self._sac_trainer = None
    
    @property
    def dqn_trainer(self):
        if self._dqn_trainer is None:
            self._dqn_trainer = DQNTrainer(self.data_path)
        return self._dqn_trainer
    
    @property
    def ppo_trainer(self):
        if self._ppo_trainer is None:
            self._ppo_trainer = PPOTrainer(self.data_path)
        return self._ppo_trainer
    
    @property
    def sac_trainer(self):
        if self._sac_trainer is None:
            self._sac_trainer = SACTrainer(self.data_path)
        return self._sac_trainer
    
    def dqn(self):
        """Train and evaluate DQN model"""
        return self.dqn_trainer.train()
    
    def ppo(self):
        """Train and evaluate PPO model"""
        return self.ppo_trainer.train()
    
    def sac(self):
        """Train and evaluate SAC model"""
        return self.sac_trainer.train() 