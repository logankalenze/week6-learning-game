# train_agent.py
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
import random

class Trainer:
    """Train the Q-learning agent to play Tic-Tac-Toe"""

    def __init__(self):
        self.game = TicTacToe()
        self.agent = QLearningAgent(player_id=1)  # Agent is X
        self.wins = {'agent': 0, 'random': 0, 'tie': 0}

    def play_training_game(self):
        self.game.reset()
        history = []

        while True:
            # Agent's turn
            state = self.game.get_state()
            actions = self.game.get_available_actions()
            if not actions:
                break

            action = self.agent.choose_action(state, actions)
            self.game.make_move(action, 1)

            history.append((state, action, 0))

            winner = self.game.check_winner()
            if winner is not None:
                self.process_game_end(winner, history)
                break

            # Random opponent
            actions = self.game.get_available_actions()
            if actions:
                opp_action = random.choice(actions)
                self.game.make_move(opp_action, -1)

            winner = self.game.check_winner()
            if winner is not None:
                self.process_game_end(winner, history)
                break

    def process_game_end(self, winner, history):
        # Rewards
        if winner == 1:
            reward = 10
            self.wins['agent'] += 1
        elif winner == -1:
            reward = -10
            self.wins['random'] += 1
        else:
            reward = 5
            self.wins['tie'] += 1

        # Backpropagate rewards
        history.reverse()
        for i, (state, action, _) in enumerate(history):
            if i == 0:
                self.agent.update_q_value(state, action, reward, "", [])
            else:
                next_state = history[i-1][0]
                next_actions = self.game.get_available_actions()
                discounted = reward * (0.9 ** i)
                self.agent.update_q_value(state, action, discounted, next_state, next_actions)

    def train(self, episodes=1000):
        print("\n==== TRAINING TIC-TAC-TOE AI ====")

        for ep in range(1, episodes + 1):
            self.play_training_game()

            if ep in [100, 500, 1000, 5000, 10000]:
                print(f"\nEpisodes: {ep}")
                print(f"Wins: {self.wins['agent']}")
                print(f"Losses: {self.wins['random']}")
                print(f"Ties: {self.wins['tie']}")

            # Reduce exploration gradually
            if ep == 500: self.agent.exploration_rate = 0.2
            if ep == 1000: self.agent.exploration_rate = 0.1
            if ep == 5000: self.agent.exploration_rate = 0.05

        self.agent.save_model("trained_model.json")
        print("\nTraining complete!")
        return self.agent


def main():
    print("\n==== TIC-TAC-TOE AI TRAINING ====")
    print("1) Quick training (1000 games)")
    print("2) Standard training (5000 games)")
    print("3) Intensive training (10000 games)")
    choice = input("Choose: ")

    episodes = {"1": 1000, "2": 5000, "3": 10000}.get(choice, 1000)

    trainer = Trainer()
    trainer.train(episodes)


if __name__ == "__main__":
    main()
