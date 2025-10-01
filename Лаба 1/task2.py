def get_positive_integer() -> int:
    """Запрашивает у пользователя целое число > 0."""
    while True:
        try:
            number = int(input("Введите целое число больше 0: ").strip())
            if number > 0:
                return number
            print("Ошибка: число должно быть больше 0.")
        except ValueError:
            print("Ошибка: нужно ввести целое число.")


def get_divisors(number: int) -> list[int]:
    """Возвращает список делителей числа (развёрнутый вариант)."""
    divisors = []
    for i in range(1, number + 1):  # перебираем все числа от 1 до number включительно
        if number % i == 0:         # если number делится на i без остатка
            divisors.append(i)      # добавляем i в список делителей
    return divisors


def is_prime(number: int) -> bool:
    """Проверяет, простое ли число (развёрнутый вариант)."""
    if number < 2:
        return False

    # перебираем все возможные делители от 2 до квадратного корня числа
    limit = int(number ** 0.5) + 1
    for i in range(2, limit):
        if number % i == 0:   # нашли делитель → число составное
            return False
    return True               # если ни одного делителя не нашли


def is_perfect(number: int, divisors: list[int]) -> bool:
    """Проверяет, является ли число совершенным (развёрнутый вариант)."""
    sum_divisors = 0
    for d in divisors:             # перебираем все делители
        if d != number:            # исключаем само число
            sum_divisors += d      # суммируем остальные делители
    return sum_divisors == number


def main():
    # получаем число от пользователя
    number = get_positive_integer()

    # находим делители
    divisors = get_divisors(number)
    print(f"Делители числа {number}: {divisors}")

    # проверяем простоту числа
    if is_prime(number):
        print(f"Число {number} является простым")
    else:
        print(f"Число {number} не является простым")

    # проверяем, является ли число совершенным
    if is_perfect(number, divisors):
        
        divisors_sum_str = ""               # строка суммы делителей
        for i in range(len(divisors) - 1):  # исключаем само число
            divisors_sum_str += str(divisors[i])
            if i < len(divisors) - 2:       # "-2" - чтобы не ставить "+" в конце
                divisors_sum_str += "+"
        print(f"Число {number} является совершенным ({divisors_sum_str}={number})")
    else:
        print(f"Число {number} не является совершенным")


if __name__ == "__main__":
    main()