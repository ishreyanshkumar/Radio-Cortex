import torch
import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from policies.bdh_policy import BDHPolicy

def test_scalability():
    print("Testing BDHPolicy Scalability...")
    
    # Initialize policy with 3 cells and 20 UEs default
    # Dimensions: 20*12 (UEs) + 3*5 (Cells) = 240 + 15 = 255
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = BDHPolicy(state_dim=255, action_dim=26, device=device).to(device)
    policy.eval()
    
    print(f"Policy initialized on {device}")
    
    # Case 1: Standard 20 UEs
    state_20 = torch.randn(1, 255).to(device)
    with torch.no_grad():
        act_20, _, val_20 = policy(state_20)
    print(f"✓ Case 20 UEs: Input {state_20.shape} -> Action {act_20.shape}")
    assert act_20.shape == (1, 20 + 3*2) # 20 UE actions, 6 Cell actions
    
    # Case 2: 50 UEs
    # D = 50*12 + 3*5 = 600 + 15 = 615
    state_50 = torch.randn(1, 615).to(device)
    with torch.no_grad():
        act_50, _, val_50 = policy(state_50)
    print(f"✓ Case 50 UEs: Input {state_50.shape} -> Action {act_50.shape}")
    assert act_50.shape == (1, 50 + 6)
    
    # Case 3: 100 UEs
    # D = 100*12 + 3*5 = 1200 + 15 = 1215
    state_100 = torch.randn(1, 1215).to(device)
    with torch.no_grad():
        act_100, _, val_100 = policy(state_100)
    print(f"✓ Case 100 UEs: Input {state_100.shape} -> Action {act_100.shape}")
    assert act_100.shape == (1, 106)
    
    # Case 4: Permutation Invariance (Heuristic)
    # If we swap UE 1 and UE 2, the actions for those UEs should swap, 
    # and other actions/value should remain the same.
    state_val = torch.randn(1, 255).to(device)
    # Swap UE 0 features (0:12) with UE 1 features (12:24)
    state_swapped = state_val.clone()
    state_swapped[:, 0:12] = state_val[:, 12:24]
    state_swapped[:, 12:24] = state_val[:, 0:12]
    
    act_orig, _, val_orig = policy(state_val)
    act_swap, _, val_swap = policy(state_swapped)
    
    # Cell actions are first (Indices 0:6)
    cell_orig = act_orig[:, :6]
    cell_swap = act_swap[:, :6]
    
    # UE actions are next (Indices 6:26)
    ue_orig = act_orig[:, 6:]
    ue_swap = act_swap[:, 6:]
    
    print(f"Testing Permutation Invariance...")
    diff_cell = torch.abs(cell_orig - cell_swap).max().item()
    diff_val = torch.abs(val_orig - val_swap).max().item()
    print(f"  Cell Action Diff: {diff_cell:.6f}")
    print(f"  Global Value Diff: {diff_val:.6f}")
    
    # UE actions should be swapped at indices 0 and 1
    # ue_swap[0] should be ue_orig[1]
    diff_ue_swap_match = torch.abs(ue_swap[:, 0] - ue_orig[:, 1]).max().item()
    print(f"  UE Action Swap Match: {diff_ue_swap_match:.6f}")
    
    assert diff_cell < 1e-4
    assert diff_val < 1e-4
    assert diff_ue_swap_match < 1e-4
    print("✓ Permutation Equivariance Verified!")

if __name__ == "__main__":
    try:
        test_scalability()
        print("\nALL SCALABILITY TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
