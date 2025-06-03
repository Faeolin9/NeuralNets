"""
Text-based demonstration of the Atari Breakout game that the DQN agent learns to play.
This version works without any additional dependencies.
"""

import numpy as np
import time
import os

class TextBreakout:
    """Text-based Breakout game to demonstrate what DQN learns"""
    
    def __init__(self, width=40, height=20):
        self.width = width
        self.height = height
        self.reset()
    
    def reset(self):
        """Reset game to initial state"""
        # Paddle
        self.paddle_width = 6
        self.paddle_x = self.width // 2 - self.paddle_width // 2
        self.paddle_y = self.height - 2
        
        # Ball
        self.ball_x = self.width // 2
        self.ball_y = self.paddle_y - 1
        self.ball_dx = 1
        self.ball_dy = -1
        
        # Bricks (6 rows, different colors represented by numbers)
        self.brick_rows = 6
        self.bricks = []
        for row in range(self.brick_rows):
            brick_row = []
            for col in range(0, self.width - 4, 4):
                brick_row.append(6 - row)  # Different values for different rows
            self.bricks.append(brick_row)
        
        self.score = 0
        self.lives = 3
        self.game_over = False
    
    def display(self):
        """Display the current game state"""
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * (self.width + 2))
        print(f"ATARI BREAKOUT - Score: {self.score} Lives: {self.lives}")
        print("=" * (self.width + 2))
        
        # Create display grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Add bricks
        for row_idx, brick_row in enumerate(self.bricks):
            y = row_idx + 2
            for col_idx, brick_value in enumerate(brick_row):
                if brick_value > 0:
                    x_start = col_idx * 4 + 2
                    # Different characters for different brick types
                    brick_chars = {6: '█', 5: '▓', 4: '▒', 3: '░', 2: '▪', 1: '·'}
                    char = brick_chars.get(brick_value, '█')
                    for i in range(3):  # 3-wide bricks
                        if x_start + i < self.width:
                            grid[y][x_start + i] = char
        
        # Add paddle
        for i in range(self.paddle_width):
            if self.paddle_x + i < self.width:
                grid[self.paddle_y][self.paddle_x + i] = '='
        
        # Add ball
        if 0 <= self.ball_y < self.height and 0 <= self.ball_x < self.width:
            grid[self.ball_y][self.ball_x] = 'o'
        
        # Print grid
        for row in grid:
            print('|' + ''.join(row) + '|')
        
        print("=" * (self.width + 2))
        print("Controls: A=Left, D=Right, S=Stay, Q=Quit")
        print("DQN Agent Actions: LEFT, RIGHT, NOOP, FIRE")
    
    def step(self, action):
        """Take one game step
        Actions: 0=NOOP, 1=LEFT, 2=RIGHT
        """
        reward = 0
        
        # Move paddle
        if action == 1:  # LEFT
            self.paddle_x = max(0, self.paddle_x - 2)
        elif action == 2:  # RIGHT
            self.paddle_x = min(self.width - self.paddle_width, self.paddle_x + 2)
        
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with walls
        if self.ball_x <= 0 or self.ball_x >= self.width - 1:
            self.ball_dx = -self.ball_dx
        
        if self.ball_y <= 0:
            self.ball_dy = -self.ball_dy
        
        # Ball collision with paddle
        if (self.ball_y >= self.paddle_y - 1 and 
            self.ball_x >= self.paddle_x and 
            self.ball_x < self.paddle_x + self.paddle_width):
            self.ball_dy = -1
            # Add some angle based on paddle hit position
            hit_pos = (self.ball_x - self.paddle_x) / self.paddle_width
            if hit_pos < 0.3:
                self.ball_dx = -1
            elif hit_pos > 0.7:
                self.ball_dx = 1
            else:
                self.ball_dx = 0 if self.ball_dx == 0 else (1 if self.ball_dx > 0 else -1)
        
        # Ball collision with bricks
        for row_idx, brick_row in enumerate(self.bricks):
            y = row_idx + 2
            if self.ball_y == y or self.ball_y == y + 1:
                for col_idx, brick_value in enumerate(brick_row):
                    if brick_value > 0:
                        x_start = col_idx * 4 + 2
                        if self.ball_x >= x_start and self.ball_x < x_start + 3:
                            self.bricks[row_idx][col_idx] = 0
                            self.ball_dy = -self.ball_dy
                            reward = brick_value  # Higher rows give more points
                            self.score += reward
                            break
        
        # Ball falls below paddle
        if self.ball_y >= self.height:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                # Reset ball
                self.ball_x = self.width // 2
                self.ball_y = self.paddle_y - 1
                self.ball_dx = 1
                self.ball_dy = -1
        
        # Check if all bricks destroyed
        total_bricks = sum(sum(1 for brick in row if brick > 0) for row in self.bricks)
        if total_bricks == 0:
            reward += 50
            self.game_over = True
        
        return reward, self.game_over

def demonstrate_dqn_game():
    """Demonstrate the game with AI player"""
    print("ATARI BREAKOUT - DQN DEMONSTRATION")
    print("=" * 50)
    print("This is the game that the DQN agent learns to master!")
    print("\nThe agent will play automatically using a simple AI...")
    print("Press Enter to start the demonstration")
    input()
    
    game = TextBreakout()
    step_count = 0
    
    while not game.game_over and step_count < 200:
        game.display()
        
        # Simple AI: move paddle towards ball
        paddle_center = game.paddle_x + game.paddle_width // 2
        
        if game.ball_x < paddle_center - 1:
            action = 1  # LEFT
            action_name = "LEFT"
        elif game.ball_x > paddle_center + 1:
            action = 2  # RIGHT
            action_name = "RIGHT"
        else:
            action = 0  # NOOP
            action_name = "STAY"
        
        print(f"AI Action: {action_name}")
        
        reward, done = game.step(action)
        if reward > 0:
            print(f"Brick destroyed! +{reward} points")
        
        time.sleep(0.3)  # Slow down for visibility
        step_count += 1
    
    game.display()
    print(f"\nGame Over! Final Score: {game.score}")
    print(f"Steps taken: {step_count}")

def explain_dqn_learning():
    """Explain how DQN learns this game"""
    print("\n" + "=" * 60)
    print("HOW DQN LEARNS TO PLAY BREAKOUT")
    print("=" * 60)
    
    print("""
WHAT YOU JUST SAW:
- The paddle (===) controlled by AI
- The ball (o) bouncing around
- Colored bricks (█▓▒░▪·) that give different points
- Simple strategy: move paddle toward ball

HOW DQN IMPROVES ON THIS:

1. INPUT PROCESSING:
   - Real game: 210x160 pixel color images
   - DQN converts to 84x84 grayscale
   - Stacks 4 consecutive frames for motion info
   - Neural network processes these images

2. LEARNING PROCESS:
   - Starts with random actions (exploration)
   - Gradually learns which actions lead to rewards
   - Uses deep neural network to predict Q-values
   - Q-value = expected future reward for each action

3. ADVANCED STRATEGIES DQN DISCOVERS:
   - Precise ball tracking and interception
   - "Tunnel" strategy: hit ball through side to get above bricks
   - Optimal brick targeting (high-value bricks first)
   - Ball angle control using paddle position

4. NETWORK ARCHITECTURE:
   Input: 84x84x4 stacked frames
   → Conv2D layers (feature extraction)
   → Dense layers (decision making)
   → Output: Q-values for [NOOP, FIRE, RIGHT, LEFT]

5. TRAINING RESULTS:
   - After millions of frames: superhuman performance
   - Can play for hours without losing
   - Achieves scores of 400+ (humans average ~30)
   - Discovers strategies humans never thought of

COMPARISON:
- Simple AI (what you saw): Reactive, follows ball
- DQN Agent: Predictive, plans ahead, optimizes strategy
- Human Expert: Good intuition but limited reaction time
- Trained DQN: Combines perfect reactions with optimal strategy
    """)

def show_game_states():
    """Show different game states"""
    print("\n" + "=" * 50)
    print("DIFFERENT GAME STATES DQN ENCOUNTERS")
    print("=" * 50)
    
    states = [
        "Game Start - Ball ready to launch",
        "Early Game - Many bricks remaining", 
        "Mid Game - Creating tunnels through bricks",
        "Late Game - Few bricks left, ball above paddle",
        "Critical - Ball falling, must save",
        "Victory - All bricks destroyed"
    ]
    
    for i, state in enumerate(states, 1):
        print(f"\n{i}. {state}")
        print("   DQN Decision: Analyze 84x84x4 image → Neural Network → Best Action")
        print("   Human Decision: See game → Think → React (much slower)")

if __name__ == "__main__":
    print("ATARI BREAKOUT GAME DEMONSTRATION")
    print("From the Reinforcement Learning Section")
    print("=" * 50)
    
    # Show the game demonstration
    demonstrate_dqn_game()
    
    # Explain how DQN learns
    explain_dqn_learning()
    
    # Show different game states
    show_game_states()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("This text version shows the basic game mechanics that DQN masters.")
    print("The real DQN processes actual Atari game pixels and achieves")
    print("superhuman performance through deep reinforcement learning!")
    print("\nThe neural network in your Jupyter notebook was designed to")
    print("learn this exact type of game through trial and error.")