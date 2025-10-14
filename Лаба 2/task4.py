class Servo:
    """Базовый класс сервопривода."""

    def __init__(self, angle=0.0, speed=0.0, acceleration=0.0, power=0.0):
        self._angle = angle
        self._speed = speed
        self._acceleration = acceleration
        self._power = power  # мощность — будем использовать для сравнения

    def __str__(self):
        return f"Servo(угол={self._angle}, скорость={self._speed}, ускорение={self._acceleration}, мощность={self._power})"

    def __repr__(self):
        return f"Servo({self._angle}, {self._speed}, {self._acceleration}, {self._power})"

    def __eq__(self, other):
        return self._power == other._power

    def __lt__(self, other):
        return self._power < other._power

    def rotate(self, delta_angle):
        """Повернуть на заданный угол."""
        self._angle += delta_angle
        print(f"Поворот: новый угол = {self._angle}")

    def move(self, delta_speed):
        """Изменить скорость вращения."""
        self._speed += delta_speed
        print(f"Изменение скорости: новая скорость = {self._speed}")


class RotationalServo(Servo):
    """Класс вращательного двигателя (наследует Servo)."""

    def __init__(self, angle=0.0, speed=0.0, acceleration=0.0, power=0.0, rotation_axis="Z"):
        super().__init__(angle, speed, acceleration, power)
        self._rotation_axis = rotation_axis

    def __str__(self):
        return f"RotationalServo(ось={self._rotation_axis}, угол={self._angle}, мощность={self._power})"

    def __repr__(self):
        return f"RotationalServo({self._rotation_axis}, {self._angle}, {self._speed}, {self._acceleration}, {self._power})"


class SynchronousServo(RotationalServo):
    """Синхронный сервопривод — конкретный тип вращательного двигателя."""

    def __init__(self, angle=0.0, speed=0.0, acceleration=0.0, power=0.0, rotation_axis="Z", frequency=50):
        super().__init__(angle, speed, acceleration, power, rotation_axis)
        self._frequency = frequency  # частота синхронизации

    def __str__(self):
        return f"SynchronousServo(ось={self._rotation_axis}, угол={self._angle}, мощность={self._power}, частота={self._frequency} Гц)"

    def __repr__(self):
        return f"SynchronousServo({self._rotation_axis}, {self._angle}, {self._speed}, {self._acceleration}, {self._power}, {self._frequency})"


class Manipulator:
    """Модель шестизвенного манипулятора."""

    def __init__(self):
        self._segments = []

    def add_servo(self, servo):
        """Добавить звено в манипулятор."""
        self._segments.append(servo)

    def __add__(self, vector):
        """
        Перегрузка оператора +
        vector — кортеж или список с углами поворота для каждого звена
        """
        if len(vector) != len(self._segments):
            print("⚠️ Количество элементов в векторе не совпадает с количеством звеньев")
            return self

        for servo, delta in zip(self._segments, vector):
            servo.rotate(delta)

        return self

    def __str__(self):
        return " -> ".join(str(servo) for servo in self._segments)

    def __repr__(self):
        return f"Manipulator({self._segments})"


def main():
    # создаём манипулятор
    manipulator = Manipulator()

    # добавляем 6 синхронных сервоприводов
    for i in range(6):
        manipulator.add_servo(SynchronousServo(angle=0, power=10 + i, rotation_axis="Z", frequency=60))

    print("🤖 Текущее состояние манипулятора:")
    print(manipulator)

    # двигаем все звенья на разные углы
    angles = [10, 15, -5, 20, -10, 30]
    manipulator + angles

    print("\n📈 После движения:")
    print(manipulator)

    # сравнение сервоприводов по мощности
    s1 = SynchronousServo(power=50)
    s2 = SynchronousServo(power=60)
    print("\n⚖️ Сравнение мощности сервоприводов:")
    print(s1 == s2)  # False
    print(s1 < s2)   # True


if __name__ == "__main__":
    main()
