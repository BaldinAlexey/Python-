from abc import ABC, abstractmethod


class Vehicle(ABC):
    """Абстрактный базовый класс для всех транспортных средств."""

    def __init__(self, vehicle_type):
        self._vehicle_type = vehicle_type

    @abstractmethod
    def get_max_speed(self):
        """Вернуть максимальную скорость ТС."""
        pass

    @abstractmethod
    def get_vehicle_type(self):
        """Вернуть тип транспортного средства."""
        pass

    def __str__(self):
        return f"Транспорт: {self._vehicle_type}"

    def __repr__(self):
        return f"Vehicle({self._vehicle_type})"


class RoadVehicle(Vehicle):
    """Абстрактный подкласс для наземных транспортных средств."""

    def __init__(self, vehicle_type):
        super().__init__(vehicle_type)

    @abstractmethod
    def get_engine_type(self):
        """Вернуть тип двигателя."""
        pass


class Car(RoadVehicle):
    """Класс для автомобиля."""

    def __init__(self, car_type, engine_type, max_speed=180):
        super().__init__("Автомобиль")
        self._car_type = car_type
        self._engine_type = engine_type
        self._max_speed = max_speed

    def get_max_speed(self):
        return self._max_speed

    def get_vehicle_type(self):
        return self._vehicle_type

    def get_engine_type(self):
        return self._engine_type

    def __str__(self):
        return f"{self._vehicle_type} ({self._car_type}), двигатель: {self._engine_type}, макс. скорость: {self._max_speed} км/ч"

    def __repr__(self):
        return f"Car({self._car_type}, {self._engine_type}, {self._max_speed})"


class Bicycle(RoadVehicle):
    """Класс для велосипеда."""

    def __init__(self, bicycle_type, max_speed=40):
        super().__init__("Велосипед")
        self._bicycle_type = bicycle_type
        self._engine_type = "мускульная сила"
        self._max_speed = max_speed

    def get_max_speed(self):
        return self._max_speed

    def get_vehicle_type(self):
        return self._vehicle_type

    def get_engine_type(self):
        return self._engine_type

    def __str__(self):
        return f"{self._vehicle_type} ({self._bicycle_type}), двигатель: {self._engine_type}, макс. скорость: {self._max_speed} км/ч"

    def __repr__(self):
        return f"Bicycle({self._bicycle_type}, {self._max_speed})"


def main():
    car = Car("седан", "электрический", 200)
    bicycle = Bicycle("горный", 35)

    print("🚘 Автомобиль:")
    print(car)
    print("Тип ТС:", car.get_vehicle_type())
    print("Тип двигателя:", car.get_engine_type())
    print("Макс. скорость:", car.get_max_speed())

    print("\n🚲 Велосипед:")
    print(bicycle)
    print("Тип ТС:", bicycle.get_vehicle_type())
    print("Тип двигателя:", bicycle.get_engine_type())
    print("Макс. скорость:", bicycle.get_max_speed())


if __name__ == "__main__":
    main()
