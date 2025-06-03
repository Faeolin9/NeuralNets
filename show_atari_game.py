"""
Script to demonstrate the Atari game environment that the DQN agent learns to play.
This shows the actual game that was referenced in the reinforcement learning section.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time

def preprocess_frame(frame):
    """Preprocess Atari frame like in the DQN implementation"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84))
    # Normalize to 0-1
    normalized = resized / 255.0
    return normalized

def show_atari_game(game_name='ALE/Breakout-v5', num_episodes=3, max_steps_per_episode=1000):
    """
    Demonstrates an Atari game environment with a random agent.
    
    Args:
        game_name: Name of the Atari game
        num_episodes: Number of episodes to show
        max_steps_per_episode: Maximum steps per episode
    """
    
    print(f"=== Demonstrating Atari Game: {game_name} ===")
    print("This is the actual game environment that the DQN agent learns to play.")
    print("The agent will play randomly to show the game mechanics.\n")
    
    try:
        # Create the environment
        env = gym.make(game_name, render_mode='rgb_array')
        
        print(f"Environment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        print(f"Number of possible actions: {env.action_space.n}")
        
        # Get action meanings if available
        if hasattr(env.unwrapped, 'get_action_meanings'):
            action_meanings = env.unwrapped.get_action_meanings()
            print(f"Action meanings: {action_meanings}")
        
        print("\n" + "="*50)
        
        # Collect frames and game info for visualization
        all_frames = []
        all_scores = []
        all_actions = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Reset environment
            observation, info = env.reset()
            total_reward = 0
            episode_frames = []
            episode_actions = []
            
            for step in range(max_steps_per_episode):
                # Random action (in practice, this would be from the DQN agent)
                action = env.action_space.sample()
                
                # Take action
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # Store frame and action
                frame = env.render()
                episode_frames.append(frame)
                episode_actions.append(action)
                
                # Store some frames for visualization (every 10th frame)
                if step % 10 == 0:
                    all_frames.append(frame)
                    all_actions.append(action)
                
                if terminated or truncated:
                    break
            
            print(f"Episode {episode + 1} finished after {step + 1} steps")
            print(f"Total reward: {total_reward}")
            all_scores.append(total_reward)
        
        env.close()
        
        # Visualize some frames from the game
        print(f"\nShowing sample frames from the game...")
        visualize_game_frames(all_frames[:12], all_actions[:12], game_name)
        
        # Show score progression
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(all_scores) + 1), all_scores)
        plt.title('Scores per Episode (Random Agent)')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Show preprocessed frame example
        plt.subplot(1, 2, 2)
        if all_frames:
            original_frame = all_frames[0]
            preprocessed = preprocess_frame(original_frame)
            plt.imshow(preprocessed, cmap='gray')
            plt.title('Preprocessed Frame (84x84, Grayscale)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return all_frames, all_scores
        
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nThis might be because the Atari ROMs are not installed.")
        print("To install them, run:")
        print("pip install gymnasium[atari]")
        print("pip install gymnasium[accept-rom-license]")
        print("pip install ale-py")
        
        # Show a simulated example instead
        show_simulated_atari_example()
        return None, None

def visualize_game_frames(frames, actions, game_name):
    """Visualize a grid of game frames"""
    if not frames:
        print("No frames to display")
        return
    
    n_frames = min(len(frames), 12)
    rows = 3
    cols = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    fig.suptitle(f'Sample Frames from {game_name}', fontsize=16)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        if i < n_frames:
            ax.imshow(frames[i])
            ax.set_title(f'Frame {i*10}, Action: {actions[i]}')
        else:
            ax.set_visible(False)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_simulated_atari_example():
    """Show a simulated example if real Atari environment is not available"""
    print("\n=== Simulated Atari Game Example ===")
    print("Since the actual Atari environment might not be available,")
    print("here's what the game would look like:")
    
    # Create a simulated game screen
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Simulate different game states
    game_states = [
        "Game Start - Ball at paddle",
        "Ball moving up",
        "Ball hitting bricks", 
        "Bricks destroyed",
        "Ball moving down",
        "Game Over"
    ]
    
    for i, (ax, state) in enumerate(zip(axes.flat, game_states)):
        # Create a simple simulation of Breakout
        screen = np.zeros((210, 160, 3), dtype=np.uint8)
        
        # Add paddle (bottom)
        paddle_x = 70 + i * 5
        screen[190:200, paddle_x:paddle_x+20] = [255, 255, 255]
        
        # Add ball
        ball_x = 80 + i * 10
        ball_y = 180 - i * 20
        if ball_y < 50:
            ball_y = 50
        screen[ball_y:ball_y+4, ball_x:ball_x+4] = [255, 255, 255]
        
        # Add bricks (top)
        for row in range(6):
            for col in range(8):
                if not (i > 2 and row < 2 and col in [3, 4]):  # Simulate destroyed bricks
                    brick_x = col * 20 + 10
                    brick_y = row * 8 + 30
                    color = [255, 100, 100] if row < 2 else [100, 255, 100] if row < 4 else [100, 100, 255]
                    screen[brick_y:brick_y+6, brick_x:brick_x+18] = color
        
        ax.imshow(screen)
        ax.set_title(state)
        ax.axis('off')
    
    plt.suptitle('Simulated Atari Breakout Game States', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\nIn the actual game:")
    print("- The paddle (white bar at bottom) is controlled by the agent")
    print("- The ball bounces around the screen")
    print("- The goal is to destroy all colored bricks at the top")
    print("- The agent gets points for destroying bricks")
    print("- The game ends when the ball falls below the paddle")

def demonstrate_dqn_concept():
    """Explain how DQN works with this game"""
    print("\n" + "="*60)
    print("HOW DQN LEARNS TO PLAY THIS GAME")
    print("="*60)
    
    print("""
The Deep Q-Network (DQN) agent learns to play Atari games through:

1. OBSERVATION: The agent sees the game screen (210x160 pixels, RGB)
   - Preprocessed to 84x84 grayscale
   - 4 consecutive frames stacked for motion information

2. ACTIONS: The agent can choose from several actions:
   - NOOP (do nothing)
   - FIRE (start game/shoot)
   - RIGHT (move paddle right)
   - LEFT (move paddle left)

3. REWARDS: The agent receives rewards based on game events:
   - +1 to +7 points for destroying different colored bricks
   - 0 points for just moving around
   - Game ends when ball is lost

4. LEARNING PROCESS:
   - Agent starts by taking random actions (exploration)
   - Gradually learns which actions lead to higher rewards
   - Uses neural network to predict Q-values (expected future rewards)
   - Balances exploration vs exploitation using epsilon-greedy strategy

5. TRAINING RESULTS:
   - After millions of game frames, DQN can achieve superhuman performance
   - Learns strategies like aiming for specific brick patterns
   - Can play for hours without losing the ball
    """)

if __name__ == "__main__":
    print("Atari Game Demonstration for DQN Reinforcement Learning")
    print("="*60)
    
    # Try to show actual Atari game
    frames, scores = show_atari_game('ALE/Breakout-v5', num_episodes=2, max_steps_per_episode=500)
    
    # Explain the DQN concept
    demonstrate_dqn_concept()
    
    print("\nThis demonstration shows the game environment that was referenced")
    print("in the reinforcement learning section of the Jupyter notebook.")
    print("The DQN agent learns to master this game through trial and error!")