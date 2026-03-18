"""
==============================================================================
DEEP REINFORCEMENT LEARNING USING SOFT ACTOR-CRITIC (SAC) FOR INVERTED PENDULUM

This program is a complete, standalone Deep Reinforcement Learning (DRL) script 
that trains an AI to balance an inverted pendulum.

CORE COMPONENTS:
* Custom Physics Engine: Bypasses standard pre-built libraries to use precise 
  Lagrangian mechanics, Runge-Kutta 4th Order (RK4) integration, and viscous 
  friction, operating at synchronized 20ms time steps.
* Soft Actor-Critic (SAC): Uses an entropy-maximizing DRL algorithm where an 
  Actor network outputs a continuous probability distribution (bell curve) of 
  motor forces, and Twin Critic networks evaluate those forces to prevent 
  overestimation errors.
* Architecture: Utilizes five 64-node neural networks (1 Actor, 2 Critics, 
  2 Target Critics).
==============================================================================
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import deque

# ==============================================================================
# 1. SYSTEM & TENSORFLOW SETUP
# ==============================================================================
# Silence informational logs and CPU warnings (e.g., AVX/FMA warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Handle GPU memory growth to prevent CUDA_ERROR_INVALID_HANDLE
# This stops TensorFlow from greedily allocating 100% of the VRAM at launch.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# ==============================================================================
# 2. Hyperparameters for Deep Reinforcement Learning
# ==============================================================================
EPISODES = 500            # Total number of attempts the AI has to learn the task.
GAMMA = 0.99              # Discount factor: How much the AI values future rewards (0.99 = heavily values the future).
TAU = 0.005               # Polyak averaging rate: How fast the target networks track the main networks.
LR_ACTOR = 0.0005         # Learning rate for the Policy (Actor) neural network.
LR_CRITIC = 0.0005        # Learning rate for the Value (Critic) neural networks.
BATCH_SIZE = 64           # Number of samples used per training step.
MEMORY_SIZE = 100000      # Maximum capacity of the Replay Buffer.
ALPHA = 0.05              # Entropy Temperature: Balances exploitation (high reward) vs exploration (randomness).

# --- Timing & Physics Hyperparameters ---
SIMULATION_TIME = 5.0     # Maximum physical seconds an episode is allowed to run.
PHYSICS_DT = 0.02         # Physics Delta Time: The physics engine calculates state updates every 20ms (50 Hz).
CONTROL_DT = PHYSICS_DT         # Control Delta Time: The DRL controller chooses a new action every 20ms (50 Hz). Should be multiples of PHYSICS_DT
ACTION_REPEAT = int(round(CONTROL_DT / PHYSICS_DT))  # How many physics frames run per each AI decision (Evaluates to 1 here).
MAX_STEPS = int(SIMULATION_TIME / CONTROL_DT) # Maximum AI decisions per episode (250 steps).

# ==============================================================================
# 3. INVERTED PENDULUM DYNAMICS ENGINE (RK4 INTEGRATION)
# ==============================================================================
class ContinuousCartPole:
    """
    Simulates the physics of an inverted pendulum mounted on a movable cart.
    Uses non-linear Lagrangian mechanics and Runge-Kutta 4th Order (RK4) integration.
    """
    def __init__(self):
        # Physical constants
        self.gravity = 9.81
        self.masscart = 1.0       # Mass of the cart (kg)
        self.masspole = 0.1       # Mass of the pole (kg)
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5         # Distance from the pivot to the pole's center of mass (m)
        self.polemass_length = self.masspole * self.length
        
        # Friction coefficients (Viscous damping)
        self.cart_friction = 0.5   # Friction between cart and track
        self.pole_friction = 0.05  # Friction inside the rotational pivot joint
        
        # Environment constraints
        self.max_force = 10.0     # Maximum force the motor can exert (Newtons)
        self.tau = PHYSICS_DT     # Time step for integration
        self.x_threshold = 4.8    # Distance the cart can travel before falling off the track
        
        # Tracking variables
        self.state = None
        self.last_force = 0.0
        self.fig = None
        self.ax = None

    def reset(self):
        """Resets the environment with slight random noise to force robustness."""
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(4,))  # Initial state. state = [x, x_dot, theta, theta_dot] 
        self.last_force = 0.0
        return np.array(self.state, dtype=np.float32)

    def _get_derivatives(self, state, force):
        """
        Calculates the instantaneous continuous derivatives: x_dot = f(x, u)
        State array breakdown: [x_position, x_velocity, pole_angle, pole_angular_velocity]
        """
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        # 1. Calculate effective horizontal force (Motor Force - Friction + Centrifugal Force)
        temp = (force - self.cart_friction * x_dot + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        
        # 2. Calculate the torque lost to friction in the pivot joint
        friction_torque = (self.pole_friction * theta_dot) / self.polemass_length
        
        # 3. Angular acceleration of the pole (theta double-dot)
        thetaacc = (self.gravity * sintheta - costheta * temp - friction_torque) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        
        # 4. Linear acceleration of the cart (x double-dot)
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Return the derivatives of the state: [velocity, acceleration, angular_velocity, angular_acceleration]
        return np.array([x_dot, xacc, theta_dot, thetaacc])

    def step(self, action):
        """Applies an action (force) to the cart and steps the physics forward in time."""
        # Clip the AI's requested action to the physical limits of the motor
        force = np.clip(action[0], -self.max_force, self.max_force)
        self.last_force = force
        done = False
        
        # Zero-Order Hold (ZOH) Loop: The control input 'force' remains constant over the time step.
        for _ in range(ACTION_REPEAT):
            # Runge-Kutta 4th Order (RK4) Integration
            # Samples the physics slopes at the start, two midpoints, and the end of the time step.
            k1 = self._get_derivatives(self.state, force)
            k2 = self._get_derivatives(self.state + 0.5 * self.tau * k1, force)
            k3 = self._get_derivatives(self.state + 0.5 * self.tau * k2, force)
            k4 = self._get_derivatives(self.state + self.tau * k3, force)
            
            # Update the state using the weighted average of the 4 RK4 slopes
            self.state = self.state + (self.tau / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            x, x_dot, theta, theta_dot = self.state
            
            # Terminal condition check: Did the cart crash or the pole fall over (>90 degrees)?
            done = bool(x < -self.x_threshold or x > self.x_threshold or theta < -math.pi / 2 or theta > math.pi / 2)
            if done: 
                break 

        # --- Gaussian Reward Function ---
        # Provides a smooth, exponential bell curve peaking at 1.0 when perfectly upright and centered.
        if not done:
            reward = np.exp(-0.5 * (theta**2 + 0.05 * x**2 + 0.01 * theta_dot**2 + 0.001 * force**2))
        else:
            reward = -5.0 # Harsh penalty for dropping the pole

        return self.state, reward, done

    def render(self):
        """Creates a live Matplotlib animation of the physical system."""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 4))
            
        self.ax.clear()
        x, _, theta, _ = self.state
        cart_w, cart_h = 1.0, 0.6
        pole_len = 2 * self.length
        
        # Trigonometry to find the tip of the pole
        pole_x = x + pole_len * math.sin(theta)
        pole_y = pole_len * math.cos(theta)
        
        # Draw track, cart, pole, and pivot dot
        self.ax.axhline(0, color='black', lw=2)
        self.ax.add_patch(plt.Rectangle((x - cart_w/2, -cart_h/2), cart_w, cart_h, color='royalblue'))
        self.ax.plot([x, pole_x], [0, pole_y], lw=6, color='darkorange')
        self.ax.plot(x, 0, 'ko', markersize=6)
        
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-1, 2.5)
        self.ax.set_title(f"Force applied: {self.last_force:.1f} N | Angle: {math.degrees(theta):.1f}°")
        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close()


# ==============================================================================
# 4. REPLAY BUFFER & NEURAL NETWORKS
# ==============================================================================
class ReplayBuffer:
    """Stores past experiences (transitions) to allow off-policy learning."""
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly samples a batch to break sequential correlation in training data."""
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.float32).reshape(-1, 1), 
                np.array(rewards, dtype=np.float32).reshape(-1, 1), 
                np.array(next_states, dtype=np.float32), 
                np.array(dones, dtype=np.float32).reshape(-1, 1))

    def size(self):
        return len(self.memory)

def build_critic_network(state_dim, action_dim):
    """
    The Critic (Q-Network) evaluates the 'goodness' of taking a specific Action in a specific State.
    Input: State (4 dims) + Action (1 dim). Output: Q-Value (1 dim).
    """
    state_input = tf.keras.layers.Input(shape=(state_dim,))
    action_input = tf.keras.layers.Input(shape=(action_dim,))
    concat = tf.keras.layers.Concatenate()([state_input, action_input])
    
    # Slim Architecture: 64 nodes per layer
    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs=[state_input, action_input], outputs=output)

class ActorNetwork(tf.keras.Model):
    """
    The Actor (Policy Network) looks at a State and decides what Action to take.
    Because the action space is continuous, it outputs a Gaussian Distribution (Mean and Std Dev).
    """
    def __init__(self, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        
        # Slim Architecture: 64 nodes per layer
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        
        # Network outputs the Mean (mu) and Log Standard Deviation (log_std) of the action
        self.mu = tf.keras.layers.Dense(action_dim, activation=None)
        self.log_std = tf.keras.layers.Dense(action_dim, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        
        # Clip log_std to prevent the network from requesting infinite variance or exactly 0 variance
        log_std = tf.clip_by_value(self.log_std(x), -20.0, 2.0)
        std = tf.exp(log_std)
        
        # --- Reparameterization Trick ---
        # Sample standard normal noise, then scale and shift it by our network's outputs. 
        noise = tf.random.normal(tf.shape(mu))
        raw_action = mu + noise * std
        
        # Squash the unbounded raw action into a neat [-1, 1] range
        action = tf.tanh(raw_action)
        
        # Calculate the log probability of drawing this specific action from our bell curve
        pre_sum_log_prob = -0.5 * (((raw_action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + tf.math.log(2 * np.pi))
        log_prob = tf.reduce_sum(pre_sum_log_prob, axis=1, keepdims=True)
        
        # Mathematical correction for the Tanh squash (Change of Variables trick)
        log_prob -= tf.reduce_sum(tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)
        
        # Scale the [-1, 1] action up to the motor's physical limit ([-10N, 10N])
        return action * self.max_action, log_prob


# ==============================================================================
# 5. SOFT ACTOR-CRITIC (SAC) AGENT
# ==============================================================================
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.actor = ActorNetwork(action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_ACTOR)

        # Twin Critics prevent overestimation bias (a common problem in Q-learning)
        self.critic_1 = build_critic_network(state_dim, action_dim)
        self.critic_2 = build_critic_network(state_dim, action_dim)
        
        # Target Critics are slow-moving copies used to stabilize the Bellman equation
        self.target_critic_1 = build_critic_network(state_dim, action_dim)
        self.target_critic_2 = build_critic_network(state_dim, action_dim)
        
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())
        
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)
        
        self.memory = ReplayBuffer(MEMORY_SIZE)

    def select_action(self, state):
        """Called during environment interaction to get a single action."""
        state = tf.cast([state], tf.float32)
        action, _ = self.actor(state)
        return action[0].numpy() 

    def update_target_networks(self):
        """Polyak Averaging: Slowly shifts target networks towards the main networks (by TAU %)."""
        for target_weight, weight in zip(self.target_critic_1.weights, self.critic_1.weights):
            target_weight.assign(weight * TAU + target_weight * (1 - TAU))
        for target_weight, weight in zip(self.target_critic_2.weights, self.critic_2.weights):
            target_weight.assign(weight * TAU + target_weight * (1 - TAU))

    def train_step(self):
        """The core Deep Reinforcement Learning update step."""
        if self.memory.size() < BATCH_SIZE: return

        # 1. Pull random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        # FIX: Explicitly convert ALL sampled data to TensorFlow Tensors (Prevents Keras 3 crash)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Open Gradient Tape to record mathematical operations for backpropagation
        with tf.GradientTape(persistent=True) as tape:
            # ------------------------------------------------------------------
            # CRITIC UPDATE PHASE (Learn to predict Q-Values accurately)
            # ------------------------------------------------------------------
            # Predict the next actions from the next states
            next_actions, next_log_probs = self.actor(next_states)
            
            # Keras 3 Multi-input tuple format: Ask target critics for future value
            target_q1 = self.target_critic_1((next_states, next_actions))
            target_q2 = self.target_critic_2((next_states, next_actions))
            
            # Use the most pessimistic target, and SUBTRACT the entropy bonus (ALPHA)
            target_q = tf.minimum(target_q1, target_q2) - ALPHA * next_log_probs
            
            # Bellman Equation: Current Reward + Discounted Future Value
            target_q = rewards + GAMMA * target_q * (1 - dones)
            
            # Ask current critics what they think the value is
            current_q1 = self.critic_1((states, actions))
            current_q2 = self.critic_2((states, actions))
            
            # Calculate Mean Squared Error (MSE) loss
            critic_1_loss = tf.reduce_mean((current_q1 - target_q)**2)
            critic_2_loss = tf.reduce_mean((current_q2 - target_q)**2)

            # ------------------------------------------------------------------
            # ACTOR UPDATE PHASE (Learn to pick better Actions)
            # ------------------------------------------------------------------
            # Evaluate the CURRENT states to get new proposed actions
            new_actions, log_probs = self.actor(states)
            
            # Ask the newly updated critics how good these proposed actions are
            q1_new = self.critic_1((states, new_actions))
            q2_new = self.critic_2((states, new_actions))
            q_new = tf.minimum(q1_new, q2_new)
            
            # Objective: Maximize expected Q-Value while ALSO maximizing Entropy (randomness)
            # Since optimizers strictly minimize, we minimize the negative of this objective.
            actor_loss = tf.reduce_mean(ALPHA * log_probs - q_new)

        # ------------------------------------------------------------------
        # APPLY GRADIENTS (Backpropagation)
        # ------------------------------------------------------------------
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))

        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        del tape # Free memory
        
        # Finally, slowly blend new critic weights into target critics
        self.update_target_networks()

    def save(self, file_prefix):
        """Saves TensorFlow network weights to the hard drive."""
        self.actor.save_weights(f"{file_prefix}_actor.weights.h5")
        self.critic_1.save_weights(f"{file_prefix}_critic1.weights.h5")
        self.critic_2.save_weights(f"{file_prefix}_critic2.weights.h5")

    def load(self, file_prefix):
        """Loads previously trained weights."""
        self.actor.load_weights(f"{file_prefix}_actor.weights.h5")
        self.critic_1.load_weights(f"{file_prefix}_critic1.weights.h5")
        self.critic_2.load_weights(f"{file_prefix}_critic2.weights.h5")


# ==============================================================================
# 6. TELEMETRY AND PLOTTING
# ==============================================================================
def simulate_and_plot(agent, env, title):
    """Runs a single episode without training and generates 3 performance graphs."""
    state = env.reset()
    times, angles, positions, forces = [], [], [], []
    
    for step in range(MAX_STEPS):
        current_time = step * CONTROL_DT
        times.append(current_time)
        positions.append(state[0])
        angles.append(math.degrees(state[2])) 
        
        action = agent.select_action(state)
        forces.append(action[0]) 
        
        next_state, _, done = env.step(action)
        state = next_state
        
        if done:
            times.append((step + 1) * CONTROL_DT)
            positions.append(state[0])
            angles.append(math.degrees(state[2]))
            forces.append(0.0) 
            break
            
    # Build a 3-panel figure
    plt.figure(figsize=(16, 5))
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    # Left: Pendulum Angle
    plt.subplot(1, 3, 1)
    plt.plot(times, angles, color='red', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5) 
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pendulum Angle (Degrees)')
    plt.title('Angle vs. Time')
    plt.xlim(0, SIMULATION_TIME)
    plt.grid(True)
    
    # Middle: Cart Position
    plt.subplot(1, 3, 2)
    plt.plot(times, positions, color='blue', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5) 
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cart Position (Meters)')
    plt.title('Position vs. Time')
    plt.xlim(0, SIMULATION_TIME)
    plt.grid(True)
    
    # Right: Control Effort (Motor Force)
    plt.subplot(1, 3, 3)
    plt.plot(times, forces, color='green', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(env.max_force, color='red', linestyle=':', alpha=0.5) # Upper physical limit
    plt.axhline(-env.max_force, color='red', linestyle=':', alpha=0.5) # Lower physical limit
    plt.xlabel('Time (seconds)')
    plt.ylabel('Force Applied (Newtons)')
    plt.title('Control Effort vs. Time')
    plt.xlim(0, SIMULATION_TIME)
    plt.grid(True)
    
    plt.tight_layout()
    print(f"\n[!] A plot window has opened. Close it to continue execution.")
    plt.show() 


# ==============================================================================
# 7. MAIN TRAINING LOOP
# ==============================================================================
def train(agent, env):
    print("\n--- Starting Warmup Phase ---")
    # Seed the memory buffer with random interactions so the networks don't 
    # mathematically collapse on their very first gradient update.
    state = env.reset()
    for _ in range(BATCH_SIZE * 2):
        action = np.random.uniform(-env.max_force, env.max_force, size=(1,))
        next_state, reward, done = env.step(action)
        agent.memory.add(state, action, reward, next_state, done)
        state = env.reset() if done else next_state
    
    print("--- Training Phase Begins ---")
    best_score = -float('inf')
    
    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        
        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action) 
            
            agent.memory.add(state, action, reward, next_state, done)
            agent.train_step() # Triggers the backpropagation math
            
            state = next_state
            total_reward += reward
            
            if done: break
            
        print(f"Episode: {e+1:04d}/{EPISODES}, Score: {total_reward:6.2f}")
        
        # --- MODEL CHECKPOINTING ---
        # Safeguard against "catastrophic forgetting" by always saving the absolute best model.
        if total_reward > best_score:
            best_score = total_reward
            agent.save("best_cartpole")
            print(f"    -> *** New Best Score! Model saved to disk. ***")

    # When training finishes, revert the agent's brain back to its all-time peak performance.
    print(f"\nTraining Complete. Loading the best overall model (Score: {best_score:.2f}).")
    agent.load("best_cartpole")
    return agent


# ==============================================================================
# 8. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    env = ContinuousCartPole()
    agent = SACAgent(state_dim=4, action_dim=1, max_action=env.max_force)
    
    # 1. Graph baseline "clueless" performance
    print("Generating 'Before Training' Plot...")
    simulate_and_plot(agent, env, "Telemetry BEFORE Training (Untrained Agent)")
    
    # 2. Train the agent
    trained_agent = train(agent, env)
    
    # 3. Graph the fully trained performance
    print("\nGenerating 'After Training' Plot...")
    simulate_and_plot(trained_agent, env, "Telemetry AFTER Training (Trained Agent)")
    
    # 4. Show a live visual animation of the final result
    print("\nStarting Visual Animation...")
    state = env.reset()
    for _ in range(MAX_STEPS):
        action = trained_agent.select_action(state)
        state, _, done = env.step(action)
        env.render()
        if done: break
    env.close()