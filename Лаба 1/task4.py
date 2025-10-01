import random

CHOICES = ["камень", "ножницы", "бумага", "ящерица", "спок"]

RULES = {
    "ножницы": ["бумага", "ящерица"],
    "бумага": ["камень", "спок"],
    "камень": ["ножницы", "ящерица"],
    "ящерица": ["спок", "бумага"],
    "спок": ["ножницы", "камень"],
}


def get_user_choice() -> str:
    """Запрашивает выбор игрока."""
    while True:
        choice = input(f"Ваш ход ({'/'.join(CHOICES)}): ").strip().lower()
        if choice in CHOICES:
            return choice
        print("Ошибка: выберите из списка.")


def get_winner(player: str, computer: str) -> str:
    """Определяет победителя."""
    if player == computer:
        return "draw"
    if computer in RULES[player]:
        return "player"
    return "computer"


def main():
    """Запускает матч до заданного количества побед."""
    while True:
        try:
            max_wins = int(input("До скольки побед играем? ").strip())
            if max_wins > 0:
                break
            print("Ошибка: число должно быть > 0.")
        except ValueError:
            print("Ошибка: нужно ввести целое число.")

    score_player = 0
    score_computer = 0

    while score_player < max_wins and score_computer < max_wins:
        player_choice = get_user_choice()
        computer_choice = random.choice(CHOICES)

        print(f"Ход компьютера: {computer_choice}")
        winner = get_winner(player_choice, computer_choice)

        if winner == "draw":
            print("Ничья!")
        elif winner == "player":
            print(f"{player_choice.capitalize()} побеждает {computer_choice}!")
            score_player += 1    # инкремент игрока
        else:
            print(f"{computer_choice.capitalize()} побеждает {player_choice}!")
            score_computer += 1  # инкремент компа

        print(f"Счёт: Вы - {score_player}, Компьютер - {score_computer}\n")

    if score_player > score_computer:
        print("Вы победили в матче!")
    else:
        print("Компьютер победил!")


if __name__ == "__main__":
    main()
