# play_game.py
from tic_tac_toe import TicTacToe
from q_learning_agent import QLearningAgent
import random

def play_against_ai():
    game = TicTacToe()
    agent = QLearningAgent(player_id=-1)

    if not agent.load_model("trained_model.json"):
        print("No trained model found!")
        return

    agent.set_training(False)

    print("\n==== PLAY TIC-TAC-TOE VS AI ====")
    print("You are X (1). AI is O (-1).")

    human_turn = random.choice([True, False])
    print("\nYou go first!" if human_turn else "\nAI goes first!")

    game.reset()

    while True:
        game.display()

        if human_turn:
            # Human move
            moves = game.get_available_actions()
            print(f"Available moves: {moves}")

            while True:
                try:
                    move = input("Your move (row,col): ")
                    r, c = map(int, move.split(","))
                    if (r, c) in moves:
                        game.make_move((r, c), 1)
                        break
                    else:
                        print("Invalid move.")
                except:
                    print("Enter as row,col (e.g., 1,2).")
        else:
            # AI move
            state = str([-x for x in game.board.flatten()])
            moves = game.get_available_actions()
            action = agent.choose_action(state, moves)
            game.make_move(action, -1)
            print(f"AI moves to {action}")

        winner = game.check_winner()
        if winner is not None:
            game.display()
            if winner == 1:
                print("\n🎉 YOU WIN!")
            elif winner == -1:
                print("\n🤖 AI WINS!")
            else:
                print("\n🤝 It's a tie.")
            break

        human_turn = not human_turn


def main():
    while True:
        print("\n==== MAIN MENU ====")
        print("1) Play vs AI")
        print("2) Exit")
        choice = input("Choice: ")

        if choice == "1":
            play_against_ai()
        else:
            break

if __name__ == "__main__":
    main()
