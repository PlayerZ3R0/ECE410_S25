import random
import numpy as np
import gym
from gym import Wrapper
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

def create_frozen_map(size, hole_positions):
    """
    Build a size×size map:
      S = start (0,0)
      G = goal  (size-1,size-1)
      H = holes at hole_positions
      F = frozen (all other cells)
    """
    grid = np.full((size, size), 'F', dtype='<U1')
    grid[0, 0] = 'S '
    grid[size-1, size-1] = 'G '
    for r, c in hole_positions:
        if (r, c) not in {(0,0), (size-1,size-1)}:
            grid[r, c] = 'H '
    # return as list of strings
    return ["".join(row) for row in grid]

class CustomRewardWrapper(Wrapper):
    """
    Wrap FrozenLakeEnv to give:
      -1.0 for any non-terminal step
      -5.0 if you fall in a hole (H)
     +10.0 if you reach the goal (G)
    """
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        # env.desc is an array of bytes, decode to char:
        cell = self.env.desc.reshape(-1)[obs].decode('utf-8')
        if cell == 'H':
            reward = -5.0
        elif cell == 'G':
            reward = 10.0
        else:
            reward = -1.0
        return obs, reward, done, info

if __name__ == "__main__":
    SIZE = 5
    N_HOLES = 4

    # pick 16 random hole positions (excluding start & goal)
    all_cells = [(i, j) for i in range(SIZE) for j in range(SIZE)]
    forbidden = {(0,0), (SIZE-1, SIZE-1)}
    candidates = [pos for pos in all_cells if pos not in forbidden]
    random.seed(42)
    holes = random.sample(candidates, N_HOLES)

    # build the map description
    desc = create_frozen_map(SIZE, holes)

    # instantiate the deterministic FrozenLake
    env = FrozenLakeEnv(desc=desc, is_slippery=False)
    env = CustomRewardWrapper(env)

    # show the map
    print("Map layout (S=start, G=goal, H=hole, F=frozen):\n")
    for row in desc:
        print(row)
    print("\nHoles at:", holes)

    # run a quick random‐policy episode to test
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        total_reward += r
    print(f"\nEpisode finished with total reward: {total_reward}")
