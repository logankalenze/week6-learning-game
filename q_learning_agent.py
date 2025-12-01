# q_learning_agent.py
import random
import json
import os

class QLearningAgent:
    """
    Q-learning agent for Tic-Tac-Toe
    """

    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.player_id = player_id
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.training = True

    def get_q_value(self, state, action):
        key = f"{state}_{action}"
        if key not in self.q_table:
            self.q_table[key] = 0.0
        return self.q_table[key]

    def choose_action(self, state, available_actions):
        if not available_actions:
            return None

        if self.training and random.random() < self.exploration_rate:
            return random.choice(available_actions)

        best_action = None
        best_value = float("-inf")

        for action in available_actions:
            q = self.get_q_value(state, action)
            if q > best_value:
                best_value = q
                best_action = action

        if best_action is None:
            best_action = random.choice(available_actions)

        return best_action

    def update_q_value(self, state, action, reward, next_state, next_actions):
        key = f"{state}_{action}"
        current_q = self.get_q_value(state, action)

        if next_actions:
            max_next_q = max(self.get_q_value(next_state, a) for a in next_actions)
        else:
            max_next_q = 0

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[key] = new_q

    def set_training(self, training):
        self.training = training
        if not training:
            self.exploration_rate = 0

    def save_model(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.q_table, f, indent=2)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.q_table = json.load(f)
            print(f"Model loaded from {filepath}")
            return True
        return False

    def get_stats(self):
        return {
            "states_learned": len(set(key.split("_")[0] for key in self.q_table)),
            "total_q_values": len(self.q_table),
            "avg_q_value": sum(self.q_table.values()) / len(self.q_table) if self.q_table else 0,
        }
