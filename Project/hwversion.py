import pyrtl
import random
import pprint
import math

# -------------------------------------------------------------------------
# Hardware Design: FrozenLake Q-Learning Accelerator
# -------------------------------------------------------------------------
# This PyRTL code defines the hardware architecture for a chiplet that
# accelerates the Q-learning algorithm for a parameterized FrozenLake environment.
#
# Key Features:
# - On-chip Q-Table: A memory block whose size adapts to the state space.
# - Fixed-Point Arithmetic: Uses 16-bit fixed-point numbers.
# - FSM Controller: A state machine orchestrates the learning process.
# - Hardware LFSR: For efficient pseudo-random number generation (epsilon-greedy).
# - Scalable Design: Modified to handle a 10x10 (100 state) environment.
# -------------------------------------------------------------------------

def fixed_point_const(val, frac_bits=8):
    """Converts a float to a PyRTL fixed-point constant."""
    return int(val * (2**frac_bits))

def create_frozenlake_accelerator(
    states=100, # Changed for 10x10 grid
    actions=4,
    q_bits=16,
    frac_bits=8,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1
):
    """
    Creates the PyRTL hardware design for the Q-learning accelerator.

    Args:
        states (int): Number of states in the environment.
        actions (int): Number of possible actions (4 for Up, Down, Left, Right).
        q_bits (int): Bit width for Q-values (fixed-point).
        frac_bits (int): Number of fractional bits in the Q-values.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration probability.

    Returns:
        A dictionary of PyRTL WireVectors for simulation.
    """
    state_bits = math.ceil(math.log2(states))
    action_bits = math.ceil(math.log2(actions))
    
    # --- I/O Wires ---
    start_episode = pyrtl.Input(1, 'start_episode')
    current_state_in = pyrtl.Input(state_bits, 'current_state_in')
    reward_in = pyrtl.Input(q_bits, 'reward_in')
    next_state_in = pyrtl.Input(state_bits, 'next_state_in')
    done_in = pyrtl.Input(1, 'done_in') # New input to signal episode end

    action_out = pyrtl.Output(action_bits, 'action_out')
    episode_done = pyrtl.Output(1, 'episode_done')
    q_table_addr_debug = pyrtl.Output(state_bits + action_bits, 'q_table_addr_debug')
    q_table_val_debug = pyrtl.Output(q_bits, 'q_table_val_debug')

    # --- Internal Components & Wires ---
    q_table = pyrtl.MemBlock(bitwidth=q_bits, addrwidth=state_bits + action_bits, name='q_table')
    q_write_addr = pyrtl.WireVector(q_table.addrwidth, 'q_write_addr')
    q_write_data = pyrtl.WireVector(q_table.bitwidth, 'q_write_data')
    we = pyrtl.WireVector(1, 'we')
    q_table[q_write_addr] <<= pyrtl.MemBlock.EnabledWrite(q_write_data, enable=we)
    q_read_addr = pyrtl.WireVector(q_table.addrwidth, 'q_read_addr')
    q_read_data = q_table[q_read_addr]

    current_state = pyrtl.Register(state_bits, 'current_state_reg')
    chosen_action = pyrtl.Register(action_bits, 'chosen_action_reg')

    ALPHA_FP = pyrtl.Const(fixed_point_const(alpha, frac_bits), q_bits)
    GAMMA_FP = pyrtl.Const(fixed_point_const(gamma, frac_bits), q_bits)
    EPSILON_FP = pyrtl.Const(fixed_point_const(epsilon, frac_bits), q_bits)

    lfsr = pyrtl.Register(16, 'lfsr')
    feedback = lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[5]
    with pyrtl.ConditionalUpdate() as lfsr_logic:
        with lfsr == 0:
            lfsr.next |= 1
        with pyrtl.otherwise:
            lfsr.next |= pyrtl.concat(feedback, lfsr[16:1])

    fsm_state = pyrtl.Register(3, 'fsm_state')
    IDLE, FETCH_Q, CHOOSE_ACTION, AWAIT_ENV, FETCH_NEXT_Q, UPDATE, WRITE_BACK = [pyrtl.Const(i, 3) for i in range(7)]
    we <<= fsm_state == WRITE_BACK # Write enable is active only in the WRITE_BACK state

    q_vals = [pyrtl.WireVector(q_bits, f'q_val_{i}') for i in range(actions)]
    max_next_q = pyrtl.WireVector(q_bits, 'max_next_q')
    
    with pyrtl.ConditionalUpdate() as FSM_LOGIC:
        with fsm_state == IDLE:
            with start_episode:
                fsm_state.next |= FETCH_Q
                current_state.next |= current_state_in
        with fsm_state == FETCH_Q:
            fsm_state.next |= CHOOSE_ACTION
        with fsm_state == CHOOSE_ACTION:
            explore = lfsr < EPSILON_FP
            random_action = lfsr[action_bits-1:0]
            
            max_q_val = pyrtl.WireVector(q_bits, 'max_q_val')
            best_action = pyrtl.WireVector(action_bits, 'best_action')
            
            max_q_val <<= pyrtl.select(q_vals[0] > q_vals[1], q_vals[0], q_vals[1])
            max_q_val <<= pyrtl.select(max_q_val > q_vals[2], max_q_val, q_vals[2])
            max_q_val <<= pyrtl.select(max_q_val > q_vals[3], max_q_val, q_vals[3])
            
            best_action <<= pyrtl.select(q_vals[0] == max_q_val, pyrtl.Const(0, bitwidth=action_bits),
                              pyrtl.select(q_vals[1] == max_q_val, pyrtl.Const(1, bitwidth=action_bits),
                              pyrtl.select(q_vals[2] == max_q_val, pyrtl.Const(2, bitwidth=action_bits), 
                                           pyrtl.Const(3, bitwidth=action_bits))))
            
            chosen_action.next |= pyrtl.select(explore, truecase=random_action, falsecase=best_action)
            fsm_state.next |= AWAIT_ENV
        with fsm_state == AWAIT_ENV:
            fsm_state.next |= FETCH_NEXT_Q
        with fsm_state == FETCH_NEXT_Q:
            fsm_state.next |= UPDATE
        with fsm_state == UPDATE:
            old_q_val = q_vals[chosen_action]
            term1 = (GAMMA_FP * max_next_q) >> frac_bits
            term2 = reward_in + term1
            term3 = term2 - old_q_val
            term4 = (ALPHA_FP * term3) >> frac_bits
            new_q_val = old_q_val + term4
            q_write_data.next |= new_q_val
            fsm_state.next |= WRITE_BACK
        with fsm_state == WRITE_BACK:
            current_state.next |= next_state_in
            with done_in: # Check the 'done' signal from the environment
                 fsm_state.next |= IDLE
            with pyrtl.otherwise:
                 fsm_state.next |= FETCH_Q

    action_out <<= chosen_action
    episode_done <<= fsm_state == IDLE
    q_write_addr <<= pyrtl.concat(current_state, chosen_action)
    q_table_addr_debug <<= q_write_addr
    q_table_val_debug <<= q_write_data

    return {
        'start_episode': start_episode, 'current_state_in': current_state_in,
        'reward_in': reward_in, 'next_state_in': next_state_in, 'done_in': done_in,
        'action_out': action_out, 'episode_done': episode_done, 'fsm_state': fsm_state, 
        'q_table': q_table, 'q_vals': q_vals, 'max_next_q': max_next_q,
    }


# -------------------------------------------------------------------------
# Software Testbench & Simulation for 10x10 Environment
# -------------------------------------------------------------------------

class FrozenLakeEnv:
    """A software model of the 10x10 FrozenLake environment."""
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.states = self.grid_size * self.grid_size
        # Define a sparse set of holes for the 10x10 grid
        self.holes = {13, 21, 28, 33, 45, 51, 59, 62, 74, 88, 92}
        self.goal = self.states - 1 # Goal is at state 99
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, state, action):
        """Action: 0=Left, 1=Down, 2=Right, 3=Up"""
        row, col = divmod(state, self.grid_size)
        
        if action == 0: col = max(col - 1, 0)
        elif action == 1: row = min(row + 1, self.grid_size - 1)
        elif action == 2: col = min(col + 1, self.grid_size - 1)
        elif action == 3: row = max(row - 1, 0)
        
        next_s = row * self.grid_size + col
        
        if next_s in self.holes:
            return next_s, 0.0, True
        elif next_s == self.goal:
            return next_s, 1.0, True
        else:
            return next_s, 0.0, False

if __name__ == "__main__":
    print("--- Setting up PyRTL Hardware Accelerator for 10x10 Grid ---")
    pyrtl.reset_working_block()
    
    GRID_SIZE = 10
    NUM_STATES = GRID_SIZE * GRID_SIZE
    FRAC_BITS = 8
    
    hw = create_frozenlake_accelerator(states=NUM_STATES, frac_bits=FRAC_BITS)

    print("\n--- Starting Simulation ---")
    sim_trace = pyrtl.SimulationTrace()
    sim = pyrtl.Simulation(tracer=sim_trace)
    
    env = FrozenLakeEnv(grid_size=GRID_SIZE)
    num_episodes = 20000 # Increased episodes for larger state space

    for episode in range(num_episodes):
        current_state = env.reset()
        sim.step({
            'start_episode': 1,
            'current_state_in': current_state,
            'reward_in': 0,
            'next_state_in': 0,
            'done_in': 0
        })
        
        for step in range(200): # Max steps per episode
            sim.step({'start_episode': 0})

            # FSM state: FETCH_Q
            # Manually "read" the 4 Q-values for the current hardware state.
            fsm_state = sim.inspect(hw['fsm_state'])
            if fsm_state == 1: 
                q_table_sim = sim.inspect_mem(hw['q_table'])
                current_hw_state = sim.inspect(hw['current_state_in'])
                for i in range(4):
                    addr = (current_hw_state << 2) + i
                    sim.wire_fast_update(hw['q_vals'][i], q_table_sim.get(addr, 0))
            
            sim.step({}) # Let FSM transition to CHOOSE_ACTION and then AWAIT_ENV
            action = sim.inspect(hw['action_out'])
            
            next_state, reward, done = env.step(current_state, action)
            
            sim.step({
                'reward_in': fixed_point_const(reward, FRAC_BITS),
                'next_state_in': next_state,
                'done_in': 1 if done else 0
            })

            # FSM state: FETCH_NEXT_Q. Provide Q-values for next_state
            fsm_state = sim.inspect(hw['fsm_state'])
            if fsm_state == 4:
                q_table_sim = sim.inspect_mem(hw['q_table'])
                max_q = 0
                for i in range(4):
                    addr = (next_state << 2) + i
                    q_val = q_table_sim.get(addr, 0)
                    if q_val > max_q:
                        max_q = q_val
                sim.wire_fast_update(hw['max_next_q'], max_q)

            sim.step({}) # UPDATE state
            sim.step({}) # WRITE_BACK state

            current_state = next_state
            if done:
                break
        
        if (episode + 1) % 2000 == 0:
            print(f"  ... Episode {episode + 1}/{num_episodes} completed.")

    print("\n--- Simulation Finished ---")
    print("Final Q-Table (first 10 states) learned by the accelerator:")

    final_q_table = sim.inspect_mem(hw['q_table'])
    pretty_q_table = {}
    for state in range(10): # Only print first 10 states for brevity
        pretty_q_table[state] = {}
        for action in range(4):
            action_map = {0: 'L', 1: 'D', 2: 'R', 3: 'U'}
            addr = (state << 2) + action
            q_val_fixed = final_q_table.get(addr, 0)
            q_val_float = q_val_fixed / (2**FRAC_BITS)
            pretty_q_table[state][action_map[action]] = f"{q_val_float:.4f}"

    pprint.pprint(pretty_q_table)

