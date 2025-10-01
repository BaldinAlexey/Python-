import random

def generate_secret_number(length: int) -> str:
    """
    Генерация случайного числа заданной длины без повторяющихся цифр
    """
    digits = "0123456789"
    secret = random.sample(digits, length)

    # число не должно начинаться с нуля
    if secret[0] == "0":
        secret[0], secret[1] = secret[1], secret[0]

    return "".join(secret)


def get_user_guess(length: int) -> str:
    """
    Запрашивает у пользователя число правильной длины.
    Обработка ошибок: только цифры, уникальные, без лишних символов
    """
    while True:
        guess = input(f"Ваш вариант ({length} цифр): ").strip()

        if not guess.isdigit():
            print("Ошибка: нужно ввести только цифры!")
            continue
        if len(guess) != length:
            print(f"Ошибка: число должно содержать ровно {length} цифр!")
            continue
        if len(set(guess)) != length:
            print("Ошибка: цифры не должны повторяться!")
            continue

        return guess


def count_cows_and_bulls(secret: str, guess: str) -> tuple[int, int]:
    """
    Подсчёт количества коров и быков
    """
    cows = 0
    bulls = 0

    # Считаем коров (совпадения по позиции)
    for i in range(len(secret)):
        if guess[i] == secret[i]:
            cows += 1

    # Считаем быков (угаданные цифры, но не на месте)
    for g in guess:
        if g in secret:
            bulls += 1

    # Убираем лишних быков, которые уже посчитаны как коровы
    bulls -= cows

    return cows, bulls


def play_game(length: int) -> int:
    """
    Запускает одну игру
    Возвращает количество попыток
    """
    secret = generate_secret_number(length)
    attempts = 0
    print(f"\nЗагадано число из {length} цифр.")

    while True:
        guess = get_user_guess(length)
        attempts += 1
        cows, bulls = count_cows_and_bulls(secret, guess)

        if cows == length:
            print(f" Вы угадали число {secret} за {attempts} попыток!")
            return attempts
        print(f"Найдено {cows} коров и {bulls} быков.")


def main():
    print("Игра 'Быки и коровы'!")
    stats: list[int] = []

    while True:
        try:
            length = int(input("Выберите длину числа (3, 4 или 5): "))
            if length not in (3, 4, 5):
                print("Ошибка: выберите 3, 4 или 5.")
                continue
        except ValueError:
            print("Ошибка: введите число!")
            continue

        attempts = play_game(length)
        stats.append(attempts)

        again = input("Хотите сыграть ещё? (да/нет): ").strip().lower()
        if again != "да":
            print("\n Статистика:")
            print(f"Всего игр: {len(stats)}")
            print(f"Лучший результат: {min(stats)} попыток")
            print(f"Худший результат: {max(stats)} попыток")
            print(f"Средний результат: {sum(stats) / len(stats):.1f} попыток")
            break


if __name__ == "__main__":
    main()
