"""
Simple demonstration of what the Atari game looks like that the DQN agent learns to play.
This version works without requiring Atari ROM installation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import time

class BreakoutSimulator:
    """Simple Breakout game simulator to show what the DQN agent learns to play"""
    
    def __init__(self, width=160, height=210):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        # Paddle
        self.paddle_width = 20
        self.paddle_height = 6
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.paddle_y = self.height - 20
        self.paddle_speed = 8
        
        # Ball
        self.ball_size = 4
        self.ball_x = self.width // 2
        self.ball_y = self.paddle_y - 10
        self.ball_dx = 3
        self.ball_dy = -3
        
        # Bricks
        self.brick_rows = 6
        self.brick_cols = 8
        self.brick_width = 18
        self.brick_height = 6
        self.bricks = np.ones((self.brick_rows, self.brick_cols))
        
        # Game state
        self.score = 0
        self.lives = 3
        self.game_over = False
        
        return self.get_screen()
    
    def get_screen(self):
        """Generate the current game screen"""
        screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw bricks
        colors = [
            [255, 100, 100],  # Red
            [255, 150, 100],  # Orange
            [255, 255, 100],  # Yellow
            [100, 255, 100],  # Green
            [100, 100, 255],  # Blue
            [255, 100, 255],  # Purple
        ]
        
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self.bricks[row, col]:
                    x = col * (self.brick_width + 2) + 10
                    y = row * (self.brick_height + 2) + 30
                    color = colors[row % len(colors)]
                    screen[y:y+self.brick_height, x:x+self.brick_width] = color
        
        # Draw paddle
        screen[self.paddle_y:self.paddle_y+self.paddle_height, 
               self.paddle_x:self.paddle_x+self.paddle_width] = [255, 255, 255]
        
        # Draw ball
        screen[self.ball_y:self.ball_y+self.ball_size, 
               self.ball_x:self.ball_x+self.ball_size] = [255, 255, 255]
        
        return screen
    
    def step(self, action):
        """Take one step in the game
        Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        """
        reward = 0
        
        # Move paddle
        if action == 2:  # RIGHT
            self.paddle_x = min(self.width - self.paddle_width, self.paddle_x + self.paddle_speed)
        elif action == 3:  # LEFT
            self.paddle_x = max(0, self.paddle_x - self.paddle_speed)
        
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x >= self.width - self.ball_size:
            self.ball_dx = -self.ball_dx
        
        if self.ball_y <= 0:
            self.ball_dy = -self.ball_dy
        
        # Ball collision with paddle
        if (self.ball_y + self.ball_size >= self.paddle_y and 
            self.ball_y <= self.paddle_y + self.paddle_height and
            self.ball_x + self.ball_size >= self.paddle_x and 
            self.ball_x <= self.paddle_x + self.paddle_width):
            self.ball_dy = -abs(self.ball_dy)  # Always bounce up
            
            # Add some angle based on where ball hits paddle
            hit_pos = (self.ball_x - self.paddle_x) / self.paddle_width
            self.ball_dx = int(6 * (hit_pos - 0.5))  # -3 to +3
        
        # Ball collision with bricks
        ball_center_x = self.ball_x + self.ball_size // 2
        ball_center_y = self.ball_y + self.ball_size // 2
        
        for row in range(self.brick_rows):
            for col in range(self.brick_cols):
                if self.bricks[row, col]:
                    brick_x = col * (self.brick_width + 2) + 10
                    brick_y = row * (self.brick_height + 2) + 30
                    
                    if (ball_center_x >= brick_x and ball_center_x <= brick_x + self.brick_width and
                        ball_center_y >= brick_y and ball_center_y <= brick_y + self.brick_height):
                        
                        self.bricks[row, col] = 0
                        self.ball_dy = -self.ball_dy
                        
                        # Different points for different rows
                        points = [7, 7, 5, 5, 3, 3, 1, 1][row] if row < 8 else 1
                        reward = points
                        self.score += points
                        break
        
        # Ball falls below paddle
        if self.ball_y > self.height:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
                reward = -1
            else:
                # Reset ball position
                self.ball_x = self.width // 2
                self.ball_y = self.paddle_y - 10
                self.ball_dx = 3
                self.ball_dy = -3
        
        # Check if all bricks destroyed
        if np.sum(self.bricks) == 0:
            reward = 100
            self.game_over = True
        
        return self.get_screen(), reward, self.game_over

def demonstrate_breakout_game():
    """Demonstrate the Breakout game that DQN learns to play"""
    print("=== Atari Breakout Game Demonstration ===")
    print("This is the game that the DQN agent learns to master!")
    print("\nGame Rules:")
    print("- Control the paddle to bounce the ball")
    print("- Destroy all bricks to win")
    print("- Don't let the ball fall below the paddle")
    print("- Different colored bricks give different points")
    
    # Create game
    game = BreakoutSimulator()
    
    # Simulate a few game states
    frames = []
    actions = []
    scores = []
    
    # Reset game
    screen = game.reset()
    frames.append(screen.copy())
    actions.append("Game Start")
    scores.append(game.score)
    
    # Simulate some gameplay with semi-intelligent actions
    for step in range(100):
        # Simple AI: move paddle towards ball
        paddle_center = game.paddle_x + game.paddle_width // 2
        ball_center = game.ball_x + game.ball_size // 2
        
        if ball_center < paddle_center - 5:
            action = 3  # LEFT
            action_name = "LEFT"
        elif ball_center > paddle_center + 5:
            action = 2  # RIGHT
            action_name = "RIGHT"
        else:
            action = 0  # NOOP
            action_name = "NOOP"
        
        screen, reward, done = game.step(action)
        
        # Store some frames for visualization
        if step % 10 == 0:
            frames.append(screen.copy())
            actions.append(f"{action_name} (Step {step})")
            scores.append(game.score)
        
        if done:
            break
    
    # Visualize the game progression
    visualize_game_progression(frames, actions, scores)
    
    return frames, game.score

def visualize_game_progression(frames, actions, scores):
    """Visualize how the game progresses"""
    n_frames = min(len(frames), 12)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Atari Breakout Game Progression\n(What the DQN Agent Learns to Play)', fontsize=16)
    
    for i in range(12):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        if i < n_frames:
            ax.imshow(frames[i])
            ax.set_title(f'{actions[i]}\nScore: {scores[i]}', fontsize=10)
        else:
            ax.set_visible(False)
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def show_dqn_input_processing():
    """Show how DQN processes the game input"""
    print("\n" + "="*60)
    print("HOW DQN PROCESSES THE GAME")
    print("="*60)
    
    game = BreakoutSimulator()
    screen = game.reset()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original screen
    axes[0].imshow(screen)
    axes[0].set_title('1. Original Game Screen\n(210x160 RGB)')
    axes[0].axis('off')
    
    # Convert to grayscale
    gray = np.dot(screen[...,:3], [0.2989, 0.5870, 0.1140])
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('2. Convert to Grayscale\n(210x160)')
    axes[1].axis('off')
    
    # Resize to 84x84
    from scipy import ndimage
    resized = ndimage.zoom(gray, (84/210, 84/160), order=1)
    axes[2].imshow(resized, cmap='gray')
    axes[2].set_title('3. Resize to 84x84\n(DQN Input Size)')
    axes[2].axis('off')
    
    # Stack 4 frames (simulate)
    stacked = np.stack([resized, resized, resized, resized], axis=-1)
    axes[3].imshow(stacked[:,:,0], cmap='gray')
    axes[3].set_title('4. Stack 4 Frames\n(84x84x4 for motion)')
    axes[3].axis('off')
    
    plt.suptitle('DQN Input Processing Pipeline', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nDQN Neural Network Architecture:")
    print("Input: 84x84x4 (4 stacked grayscale frames)")
    print("Conv2D: 32 filters, 8x8, stride 4 → ReLU")
    print("Conv2D: 64 filters, 4x4, stride 2 → ReLU") 
    print("Conv2D: 64 filters, 3x3, stride 1 → ReLU")
    print("Flatten → Dense(512) → ReLU")
    print("Output: Dense(4) → Q-values for [NOOP, FIRE, RIGHT, LEFT]")

def explain_dqn_learning():
    """Explain how DQN learns to play"""
    print("\n" + "="*60)
    print("HOW DQN LEARNS TO MASTER BREAKOUT")
    print("="*60)
    
    print("""
LEARNING PROCESS:

1. RANDOM EXPLORATION (Early Training):
   - Agent takes random actions
   - Learns basic game mechanics
   - Discovers that moving paddle catches ball
   - Score: 0-10 points typically

2. BASIC STRATEGY (Mid Training):
   - Learns to track ball with paddle
   - Starts hitting bricks occasionally
   - Develops hand-eye coordination
   - Score: 10-50 points

3. ADVANCED STRATEGY (Late Training):
   - Learns to aim ball at specific brick patterns
   - Discovers "tunnel" strategy (hit side to get ball above bricks)
   - Maximizes points by targeting high-value bricks first
   - Score: 200-400+ points

4. SUPERHUMAN PERFORMANCE:
   - Can play for hours without losing ball
   - Consistently executes optimal strategies
   - Achieves scores impossible for most humans
   - Score: 500+ points regularly

KEY INSIGHTS DQN DISCOVERS:
- Paddle position should anticipate ball trajectory
- Creating "tunnels" through brick walls is highly effective
- Different brick colors have different point values
- Ball angle can be controlled by paddle hit position
    """)

if __name__ == "__main__":
    print("Atari Breakout Game Demonstration")
    print("This shows the actual game from the reinforcement learning section!")
    print("="*70)
    
    # Demonstrate the game
    frames, final_score = demonstrate_breakout_game()
    
    # Show input processing
    show_dqn_input_processing()
    
    # Explain learning process
    explain_dqn_learning()
    
    print(f"\nFinal Score in Demo: {final_score}")
    print("\nThis is the game environment that the DQN agent in your notebook")
    print("was designed to learn! The agent starts knowing nothing about the game")
    print("and through millions of trial-and-error attempts, learns to play")
    print("better than human experts.")