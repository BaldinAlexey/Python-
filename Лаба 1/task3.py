import random

WORDS = ["лотос", "столп", "комод", "парус", "молот", "язык"]


def choose_word() -> str:
    """Выбирает случайное слово из списка."""
    return random.choice(WORDS)


def get_user_word(length: int) -> str:
    """Запрашивает у пользователя слово правильной длины."""
    while True:
        guess = input(f"Введите слово из {length} букв: ").strip().lower()
        if len(guess) != length:
            print(f"Ошибка: слово должно содержать {length} букв.")
            continue
        if not guess.isalpha():
            print("Ошибка: можно использовать только буквы.")
            continue
        return guess


def check_word(secret: str, guess: str) -> str:
    """
    Сравнивает загаданное слово и ввод игрока.
    [X] — правильная буква на месте,
    (X) — есть в слове, но не на месте,
     X  — буквы нет.
    """
    result = []                    # список строк
    secret_letters = list(secret)  # список букв слова, чтобы вычеркивать

    # правильные буквы
    for i in range(len(secret)):
        if guess[i] == secret[i]:
            result.append(f"[{guess[i]}]")
            secret_letters[i] = None        # пометка просмотра
        else:
            result.append(None)

    # буквы на других местах
    for i in range(len(secret)):
        if result[i] is None:
            if guess[i] in secret_letters:
                result[i] = f"({guess[i]})"
                secret_letters[secret_letters.index(guess[i])] = None   # .index(x) ищет первый индекс элемента x в списке
            else:
                result[i] = guess[i]

    return " ".join(result)


def main():
    """Запускает игру Wordle."""
    secret = choose_word()
    attempts = 6    # кол-во попыток
    print(f"\nЗагадано слово из {len(secret)} букв. У вас {attempts} попыток.")

    for attempt in range(1, attempts + 1):
        guess = get_user_word(len(secret))
        result = check_word(secret, guess)
        print(f"Попытка {attempt}: {result}")

        if guess == secret:
            print(f"Вы угадали слово \"{secret}\" за {attempt} попыток!")
            return

    print(f"Не угадали. Слово было: {secret}")


if __name__ == "__main__":
    main()
