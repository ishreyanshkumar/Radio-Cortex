from oran_ns3_env import ORANEnv, NS3Config
import numpy as np

config = NS3Config(num_cells=3)
env = ORANEnv(config)

print("="*40)
print(f"Num Cells: {config.num_cells}")
print(f"Action Space: {env.action_space}")
print(f"Action Low Shape: {env.action_space.low.shape}")
print("="*40)
