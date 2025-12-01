# Tic-Tac-Toe Q-Learning AI
## Project Overview
This project demonstrates reinforcement learning by training an AI to play Tic-Tac-Toe
through trial and error, without being explicitly programmed with strategies.
## How to Run
1. **Train the AI**
```bash
python train_agent.py
```
Choose training intensity (1000-10000 games)
2. **Play Against AI**
```bash
python play_game.py
```
## How Q-Learning Works
### The Q-Table
The AI maintains a table of Q-values for each state-action pair:
- **State**: Current board configuration
- **Action**: Where to place the mark
- **Q-Value**: How good that action is in that state
### Learning Process
1. **Exploration**: Try random moves to discover strategies
2. **Exploitation**: Use learned knowledge to make good moves
3. **Reward**: Win = +10, Lose = -10, Tie = +5
4. **Update**: Adjust Q-values based on outcomes
### My Training Results
- Games trained: 5000
- Final win rate: 78%
- States learned: 5000
## What I Observed
[Write 2-3 observations about how the AI's play improved]
## Reinforcement Learning in Real Life
1. **Game AI**: [Example]
2. **Robotics**: [Example]
3. **Recommendation Systems**: [Example]
## Challenges and Solutions
One challenge was that Python kept throwing ImportError when trying to run the project. The problem turned out to be running scripts with full file paths, which broke Python’s working directory. By switching to running the scripts from inside the project folder, the imports resolved correctly and the model file saved where it belonged.
## What I Learned
Reinforcement learning is all about an agent learning through trial and error by interacting with an environment and receiving rewards or penalties. I learned how Q-learning builds a “map” of state–action pairs and updates their quality values over time using feedback from wins, losses, and ties. I also now understand how exploration vs. exploitation works, and why the agent needs randomness early on but becomes more deterministic as it learns. Finally, I saw firsthand how discounted rewards allow earlier actions in a sequence to still get credit for contributing to a win, which is a core concept in long-term decision making.