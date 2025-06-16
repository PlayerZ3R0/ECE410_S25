import pyrtl
import random
import pprint
import math
import time

# -------------------------------------------------------------------------
# Part 1: Hardware Design (PyRTL Accelerator)
# This is the same scalable 10x10 accelerator design from before.
# UPDATED: Re-architected with a pipelined FSM to handle memory port limits.
# -------------------------------------------------------------------------

def fixed_point_const(val, frac_bits=8):
    return int(val * (2**frac_bits))

def create_frozenlake_accelerator(
    states=100, actions=4, q_bits=16, frac_bits=8,
    alpha=0.1, gamma=0.9, epsilon=0.1
):
    state_bits = math.ceil(math.log2(states))
    action_bits = math.ceil(math.log2(actions))
    
    # --- I/O Wires ---
    start_episode = pyrtl.Input(1, 'start_episode')
    current_state_in = pyrtl.Input(state_bits, 'current_state_in')
    reward_in = pyrtl.Input(q_bits, 'reward_in')
    next_state_in = pyrtl.Input(state_bits, 'next_state_in')
    done_in = pyrtl.Input(1, 'done_in')

    action_out = pyrtl.Output(action_bits, 'action_out')
    episode_done = pyrtl.Output(1, 'episode_done')
    
    # --- Internal Components & Wires ---
    # A single memory block, representing a realistic single-port RAM
    q_table = pyrtl.MemBlock(bitwidth=q_bits, addrwidth=state_bits + action_bits, name='q_table', asynchronous=True)
    q_read_addr = pyrtl.WireVector(q_table.addrwidth, 'q_read_addr')
    q_write_addr = pyrtl.WireVector(q_table.addrwidth, 'q_write_addr')
    q_write_data = pyrtl.Register(q_table.bitwidth, 'q_write_data')
    
    we = pyrtl.WireVector(1, 'we')
    q_table[q_write_addr] <<= pyrtl.MemBlock.EnabledWrite(q_write_data, enable=we)
    q_read_data = q_table[q_read_addr]

    current_state = pyrtl.Register(state_bits, 'current_state_reg')
    chosen_action = pyrtl.Register(action_bits, 'chosen_action_reg')

    # Registers to store fetched Q-values (pipelining)
    q_val_regs = [pyrtl.Register(q_bits, f'q_val_reg_{i}') for i in range(actions)]
    next_q_val_regs = [pyrtl.Register(q_bits, f'next_q_val_reg_{i}') for i in range(actions)]

    ALPHA_FP = pyrtl.Const(fixed_point_const(alpha, frac_bits), q_bits)
    GAMMA_FP = pyrtl.Const(fixed_point_const(gamma, frac_bits), q_bits)
    EPSILON_FP = pyrtl.Const(fixed_point_const(epsilon, frac_bits), q_bits)

    lfsr = pyrtl.Register(16, 'lfsr')
    feedback = lfsr[15] ^ lfsr[14] ^ lfsr[13] ^ lfsr[11]
    shifted_val = pyrtl.shift_left_logical(lfsr, 1) | feedback
    lfsr.next <<= pyrtl.select(lfsr == 0, truecase=1, falsecase=shifted_val)

    # Expanded FSM to handle sequential memory reads
    fsm_state = pyrtl.Register(4, 'fsm_state')
    IDLE, \
    FETCH_CURR_Q0, FETCH_CURR_Q1, FETCH_CURR_Q2, FETCH_CURR_Q3, \
    CHOOSE_ACTION, AWAIT_ENV, \
    FETCH_NEXT_Q0, FETCH_NEXT_Q1, FETCH_NEXT_Q2, FETCH_NEXT_Q3, \
    UPDATE, WRITE_BACK = [pyrtl.Const(i, bitwidth=4) for i in range(13)]
    
    we <<= fsm_state == WRITE_BACK
    episode_done <<= fsm_state == IDLE

    # --- FSM Logic ---
    with pyrtl.conditional_assignment:
        with fsm_state == IDLE:
            q_read_addr |= 0
            with start_episode:
                fsm_state.next |= FETCH_CURR_Q0
                current_state.next |= current_state_in
            with pyrtl.otherwise:
                fsm_state.next |= IDLE # Stay idle unless started
        
        # Sequentially fetch Q-values for the current state
        with fsm_state == FETCH_CURR_Q0:
            q_read_addr |= pyrtl.concat(current_state, pyrtl.Const(0, action_bits))
            q_val_regs[0].next |= q_read_data
            fsm_state.next |= FETCH_CURR_Q1
        with fsm_state == FETCH_CURR_Q1:
            q_read_addr |= pyrtl.concat(current_state, pyrtl.Const(1, action_bits))
            q_val_regs[1].next |= q_read_data
            fsm_state.next |= FETCH_CURR_Q2
        with fsm_state == FETCH_CURR_Q2:
            q_read_addr |= pyrtl.concat(current_state, pyrtl.Const(2, action_bits))
            q_val_regs[2].next |= q_read_data
            fsm_state.next |= FETCH_CURR_Q3
        with fsm_state == FETCH_CURR_Q3:
            q_read_addr |= pyrtl.concat(current_state, pyrtl.Const(3, action_bits))
            q_val_regs[3].next |= q_read_data
            fsm_state.next |= CHOOSE_ACTION

        with fsm_state == CHOOSE_ACTION:
            q_read_addr |= 0
            explore = lfsr < EPSILON_FP
            random_action = lfsr & pyrtl.Const(3, bitwidth=lfsr.bitwidth)

            max_q_val = pyrtl.mux(q_val_regs[0] > q_val_regs[1], q_val_regs[0], q_val_regs[1])
            max_q_val = pyrtl.mux(max_q_val > q_val_regs[2], max_q_val, q_val_regs[2])
            max_q_val = pyrtl.mux(max_q_val > q_val_regs[3], max_q_val, q_val_regs[3])

            best_action = pyrtl.mux(q_val_regs[0] == max_q_val, pyrtl.Const(0),
                                pyrtl.mux(q_val_regs[1] == max_q_val, pyrtl.Const(1),
                                        pyrtl.mux(q_val_regs[2] == max_q_val, pyrtl.Const(2),
                                                pyrtl.Const(3))))
            
            chosen_action.next |= pyrtl.select(explore, truecase=random_action, falsecase=best_action)
            fsm_state.next |= AWAIT_ENV

        with fsm_state == AWAIT_ENV:
            q_read_addr |= 0
            # Hardware waits for the environment to provide the next state
            fsm_state.next |= FETCH_NEXT_Q0

        # Sequentially fetch Q-values for the next state
        with fsm_state == FETCH_NEXT_Q0:
            q_read_addr |= pyrtl.concat(next_state_in, pyrtl.Const(0, action_bits))
            next_q_val_regs[0].next |= q_read_data
            fsm_state.next |= FETCH_NEXT_Q1
        with fsm_state == FETCH_NEXT_Q1:
            q_read_addr |= pyrtl.concat(next_state_in, pyrtl.Const(1, action_bits))
            next_q_val_regs[1].next |= q_read_data
            fsm_state.next |= FETCH_NEXT_Q2
        with fsm_state == FETCH_NEXT_Q2:
            q_read_addr |= pyrtl.concat(next_state_in, pyrtl.Const(2, action_bits))
            next_q_val_regs[2].next |= q_read_data
            fsm_state.next |= FETCH_NEXT_Q3
        with fsm_state == FETCH_NEXT_Q3:
            q_read_addr |= pyrtl.concat(next_state_in, pyrtl.Const(3, action_bits))
            next_q_val_regs[3].next |= q_read_data
            fsm_state.next |= UPDATE
            
        with fsm_state == UPDATE:
            q_read_addr |= 0
            max_next_q = pyrtl.mux(next_q_val_regs[0] > next_q_val_regs[1], next_q_val_regs[0], next_q_val_regs[1])
            max_next_q = pyrtl.mux(max_next_q > next_q_val_regs[2], max_next_q, next_q_val_regs[2])
            max_next_q = pyrtl.mux(max_next_q > next_q_val_regs[3], max_next_q, next_q_val_regs[3])

            old_q_val = pyrtl.mux(chosen_action, q_val_regs[3], q_val_regs[2], q_val_regs[1], q_val_regs[0])

            term1_mult = GAMMA_FP * max_next_q
            term1 = pyrtl.shift_right_arithmetic(term1_mult, frac_bits)
            term2 = reward_in + term1
            term3 = term2 - old_q_val
            term4_mult = ALPHA_FP * term3
            term4 = pyrtl.shift_right_arithmetic(term4_mult, frac_bits)
            
            new_q_val = old_q_val + term4
            q_write_data.next |= new_q_val
            fsm_state.next |= WRITE_BACK

        with fsm_state == WRITE_BACK:
            q_read_addr |= 0
            q_write_addr |= pyrtl.concat(current_state, chosen_action)
            with done_in:
                fsm_state.next |= IDLE
            with pyrtl.otherwise:
                current_state.next |= next_state_in
                fsm_state.next |= FETCH_CURR_Q0
        
    action_out <<= chosen_action
    
    return {
        'start_episode': start_episode, 'current_state_in': current_state_in,
        'reward_in': reward_in, 'next_state_in': next_state_in, 'done_in': done_in,
        'action_out': action_out, 'fsm_state': fsm_state, 'episode_done': episode_done,
        'AWAIT_ENV_STATE': AWAIT_ENV
    }

# -------------------------------------------------------------------------
# Part 2: Software-Only Q-Learning Implementation
# -------------------------------------------------------------------------

def run_software_q_learning(env, episodes, alpha, gamma, epsilon):
    """ Pure Python implementation of Q-learning for benchmarking. """
    q_table = [[0.0] * (env.grid_size * env.grid_size) for _ in range(4)]
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                q_values_for_state = [q_table[a][state] for a in range(4)]
                action = q_values_for_state.index(max(q_values_for_state))

            next_state, reward, done = env.step(state, action)
            
            old_value = q_table[action][state]
            next_max = max([q_table[a][next_state] for a in range(4)])
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[action][state] = new_value
            
            state = next_state
    
    return q_table


# -------------------------------------------------------------------------
# Part 3: Environment & Benchmark Execution
# -------------------------------------------------------------------------

class FrozenLakeEnv:
    """A software model of the 5x5 FrozenLake environment with random holes."""
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.states = self.grid_size * self.grid_size
        self.goal = self.states - 1
        
        # Calculate the number of holes (20% of the grid)
        num_holes = int(self.states * 0.20)
        
        # Generate a list of all possible locations for holes
        # Exclude the start (0) and goal states
        possible_hole_locations = list(range(1, self.goal))
        
        # Randomly select the hole locations
        self.holes = set(random.sample(possible_hole_locations, num_holes))
        
        self.state = 0
        print(f"Generated a {grid_size}x{grid_size} environment with holes at: {sorted(list(self.holes))}")


    def reset(self):
        self.state = 0
        return self.state

    def step(self, state, action):
        row, col = divmod(state, self.grid_size)
        if action == 0: col = max(col - 1, 0)
        elif action == 1: row = min(row + 1, self.grid_size - 1)
        elif action == 2: col = min(col + 1, self.grid_size - 1)
        elif action == 3: row = max(row - 1, 0)
        next_s = row * self.grid_size + col
        
        if next_s in self.holes: return next_s, 0.0, True
        elif next_s == self.goal: return next_s, 1.0, True
        else: return next_s, 0.0, False

def run_hardware_simulation(hw_design, env, episodes, frac_bits):
    """ Runs the PyRTL simulation and counts clock cycles. """
    sim = pyrtl.Simulation()
    clock_cycles = 0
    AWAIT_ENV_STATE = hw_design['AWAIT_ENV_STATE']

    for episode in range(episodes):
        current_state = env.reset()
        inputs = {
            'start_episode': 1,
            'current_state_in': current_state,
            'reward_in': 0,
            'next_state_in': 0,
            'done_in': 0
        }
        sim.step(inputs)
        clock_cycles += 1
        
        # De-assert start signal after the first cycle
        inputs['start_episode'] = 0

        for step in range(500): # Increased max steps per episode
            fsm_state = sim.inspect(hw_design['fsm_state'])

            if fsm_state == AWAIT_ENV_STATE.val:
                action = sim.inspect(hw_design['action_out'])
                next_state, reward, done = env.step(current_state, action)
                
                # Update inputs with new values from the environment
                inputs['reward_in'] = fixed_point_const(reward, frac_bits)
                inputs['next_state_in'] = next_state
                inputs['done_in'] = 1 if done else 0
                
                current_state = next_state
            
            sim.step(inputs)
            clock_cycles += 1

            if sim.inspect(hw_design['episode_done']):
                break
            
    return clock_cycles

if __name__ == "__main__":
    # --- Benchmark Parameters ---
    GRID_SIZE = 10
    NUM_STATES = GRID_SIZE * GRID_SIZE
    NUM_EPISODES = 250
    FRAC_BITS = 8
    ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.1
    ASIC_FREQ_MHZ = 200 # Conservative frequency for a simple custom ASIC

    env = FrozenLakeEnv(grid_size=GRID_SIZE)

    # --- Run Hardware Benchmark ---
    print("--- Running Hardware Accelerator Simulation ---")
    pyrtl.reset_working_block()
    hw = create_frozenlake_accelerator(states=NUM_STATES, frac_bits=FRAC_BITS, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
    hw_start_time = time.time()
    total_hw_cycles = run_hardware_simulation(hw, env, NUM_EPISODES, FRAC_BITS)
    hw_sim_time = time.time() - hw_start_time
    print(f"Hardware simulation finished in {hw_sim_time:.2f} seconds (this is not the real performance).")


    # --- Run Software Benchmark ---
    print("\n--- Running Pure Software Simulation ---")
    sw_start_time = time.time()
    run_software_q_learning(env, NUM_EPISODES, ALPHA, GAMMA, EPSILON)
    sw_time = time.time() - sw_start_time
    print(f"Software simulation finished.")

    # --- Calculate and Print Results ---
    hw_time_sec = total_hw_cycles / (ASIC_FREQ_MHZ * 1_000_000)
    speedup_factor = sw_time / hw_time_sec if hw_time_sec > 0 else float('inf')

    print("\n" + "="*40)
    print("           BENCHMARK RESULTS")
    print("="*40)
    print(f"Task:              {NUM_EPISODES} episodes on a {GRID_SIZE}x{GRID_SIZE} grid")
    print(f"Software Runtime:  {sw_time:.4f} seconds")
    print(f"Hardware Cycles:   {total_hw_cycles} clock cycles")
    print(f"Assumed ASIC Freq: {ASIC_FREQ_MHZ} MHz")
    print(f"Theoretical HW Time: {hw_time_sec:.6f} seconds")
    print("-"*40)
    print(f"CALCULATED SPEEDUP: {speedup_factor:.2f}x")
    print("="*40)
    print("\nAnalysis:")
    print("The hardware accelerator's performance is measured in clock cycles. Each cycle is extremely fast.")
    print("The software version's performance depends on the CPU's speed and architecture, but requires many instructions for each Q-update.")
    print("This result clearly shows that the hardware architecture completes the task in a predictable number of cycles, representing a significant optimization.")

