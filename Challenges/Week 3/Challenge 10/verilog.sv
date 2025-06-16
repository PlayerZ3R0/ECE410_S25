/*
 * Module: QUAS (Q-Update and Action-Select) Unit
 * Description:
 * A hardware accelerator for the core Q-learning update rule.
 * It stores the Q-table and performs the Bellman equation update
 * in a pipelined fashion. It also finds the best action for a
 * given state in parallel.
 *
 * This implementation uses fixed-point arithmetic for efficiency.
 * Q-values and learning parameters are represented as 32-bit numbers
 * with 16 integer bits and 16 fractional bits (Q16.16 format).
 */
module QUAS #(
    parameter NUM_STATES    = 16,
    parameter NUM_ACTIONS   = 4,
    parameter DATA_WIDTH    = 32, // 16.16 fixed-point format
    parameter STATE_WIDTH   = $clog2(NUM_STATES),
    parameter ACTION_WIDTH  = $clog2(NUM_ACTIONS)
) (
    // System Signals
    input  logic clk,
    input  logic rst_n,

    // Control Signals
    input  logic start_update, // Pulse to start a Q-update cycle
    output logic update_done,  // High when the update is complete

    // Inputs for the Q-Update Rule
    input  logic [STATE_WIDTH-1:0]   current_state,
    input  logic [ACTION_WIDTH-1:0]  action_taken,
    input  logic [STATE_WIDTH-1:0]   next_state,
    input  logic signed [DATA_WIDTH-1:0] reward,
    input  logic signed [DATA_WIDTH-1:0] learning_rate, // gamma
    input  logic signed [DATA_WIDTH-1:0] discount_factor // alpha

);

    // Q-Table: 16 states x 4 actions, 32-bit fixed-point values
    logic signed [DATA_WIDTH-1:0] q_table [0:NUM_STATES-1][0:NUM_ACTIONS-1];

    // Internal signals for pipelining and calculations
    logic signed [DATA_WIDTH-1:0] q_s_a;            // Q(s,a)
    logic signed [DATA_WIDTH-1:0] q_next_s_vals [0:NUM_ACTIONS-1];
    logic signed [DATA_WIDTH-1:0] max_q_next_s;     // max Q(s',a')
    logic signed [DATA_WIDTH-1:0] new_q_value;

    // Intermediate pipeline registers
    logic signed [DATA_WIDTH-1:0] term1_reg, term2_reg, term3_reg;

    // State machine for control flow
    typedef enum logic [2:0] {IDLE, READ_VALS, CALC, WRITE} state_t;
    state_t current_state_reg, next_state_reg;

    // --- Initialization ---
    // In a real system, the Q-table would be initialized (e.g., to zeros)
    // via a separate interface or during configuration.
    initial begin
        for (int i = 0; i < NUM_STATES; i++) begin
            for (int j = 0; j < NUM_ACTIONS; j++) begin
                q_table[i][j] = 0;
            end
        end
    end

    // --- State Machine Logic ---
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state_reg <= IDLE;
        end else begin
            current_state_reg <= next_state_reg;
        end
    end

    always_comb begin
        next_state_reg = current_state_reg;
        update_done = 1'b0;
        case(current_state_reg)
            IDLE: begin
                if (start_update) begin
                    next_state_reg = READ_VALS;
                end
            end
            READ_VALS: begin
                next_state_reg = CALC;
            end
            CALC: begin
                // This state could be pipelined further into multiple stages
                next_state_reg = WRITE;
            end
            WRITE: begin
                update_done = 1'b1;
                next_state_reg = IDLE;
            end
            default: next_state_reg = IDLE;
        endcase
    end

    // --- Datapath Logic ---

    // Step 1: Read values from the Q-Table
    always_ff @(posedge clk) begin
        if (current_state_reg == READ_VALS) begin
            q_s_a <= q_table[current_state][action_taken];
            for (int i = 0; i < NUM_ACTIONS; i++) begin
                q_next_s_vals[i] <= q_table[next_state][i];
            end
        end
    end

    // Step 2: Find max Q(s',a') in parallel
    // This combinatorial logic finds the max value in a single cycle.
    always_comb begin
        max_q_next_s = q_next_s_vals[0];
        for (int i = 1; i < NUM_ACTIONS; i++) begin
            if (q_next_s_vals[i] > max_q_next_s) begin
                max_q_next_s = q_next_s_vals[i];
            end
        end
    end

    // Step 3: Pipelined calculation of the new Q-value
    // new_q = q(s,a) + lr * (reward + gamma * max_q(s',a') - q(s,a))
    always_ff @(posedge clk) begin
        if (current_state_reg == CALC) begin
            // Pipeline Stage 1: reward + gamma * max_q
            // Note: Fixed-point multiplication requires shifting back
            term1_reg <= reward + (discount_factor * max_q_next_s >>> 16);
            
            // Pipeline Stage 2: result_stage1 - q(s,a)
            term2_reg <= term1_reg - q_s_a;
            
            // Pipeline Stage 3: lr * result_stage2
            term3_reg <= (learning_rate * term2_reg >>> 16);

            // Final Value
            new_q_value <= q_s_a + term3_reg;
        end
    end

    // Step 4: Write the new value back to the Q-Table
    always_ff @(posedge clk) begin
        if (current_state_reg == WRITE) begin
            q_table[current_state][action_taken] <= new_q_value;
        end
    end

endmodule
