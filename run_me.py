from src.drl import DRL
import argparse

def main():
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning Trading System')
    parser.add_argument('--data_path', type=str, default='data/cmma.csv', help='Path to input data')
    parser.add_argument('--mode', type=str, choices=['regimes', 'metalabel', 'dqn', 'ppo', 'sac', 'all'], 
                       default='all', help='Mode to run')
    args = parser.parse_args()
    
    # Initialize DRL system
    drl = DRL(data_path=args.data_path)
    
    if args.mode in ['regimes', 'all']:
        regimes_df = drl.regimes()
        print("Regimes detection completed")
        
    if args.mode in ['metalabel', 'all']:
        metalabel_df = drl.metalabel()
        print("Metalabeling completed")
        
    if args.mode in ['dqn', 'all']:
        drl.rl.dqn()
        print("DQN training completed")
        
    if args.mode in ['ppo', 'all']:
        drl.rl.ppo()
        print("PPO training completed")
        
    if args.mode in ['sac', 'all']:
        drl.rl.sac()
        print("SAC training completed")

if __name__ == "__main__":
    main() 