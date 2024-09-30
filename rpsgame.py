import random

def get_computer_choice():
    """Randomly choose rock, paper, or scissors for the computer."""
    choices = ['rock', 'paper', 'scissors']
    return random.choice(choices)

def get_user_choice():
    """Get the user's choice and validate the input."""
    while True:
        user_input = input("Enter rock, paper, or scissors: ").lower()
        if user_input in ['rock', 'paper', 'scissors']:
            return user_input
        else:
            print("Invalid choice. Please try again.")

def determine_winner(user_choice, computer_choice):
    """Determine the winner of a round."""
    if user_choice == computer_choice:
        return "It's a tie!"
    elif (user_choice == 'rock' and computer_choice == 'scissors') or \
         (user_choice == 'paper' and computer_choice == 'rock') or \
         (user_choice == 'scissors' and computer_choice == 'paper'):
        return "User wins!"
    else:
        return "Computer wins!"

def play_game():
    user_points = 0
    computer_points = 0
    rounds = 3

    for round in range(1, rounds + 1):
        print(f"\nRound {round}")
        
        user_choice = get_user_choice()
        computer_choice = get_computer_choice()
        
        print(f"Computer chose: {computer_choice}")
        
        result = determine_winner(user_choice, computer_choice)
        print(result)
        
        if result == "User wins!":
            user_points += 1
        elif result == "Computer wins!":
            computer_points += 1
        
        print(f"Score: User - {user_points}, Computer - {computer_points}")

    # Final results
    print("\nFinal Results:")
    print(f"User Points: {user_points}, Computer Points: {computer_points}")
    
    if user_points > computer_points:
        print("User is the overall winner!")
    elif user_points < computer_points:
        print("Computer is the overall winner!")
    else:
        print("It's an overall tie!")

if __name__ == "__main__":
    play_game()
