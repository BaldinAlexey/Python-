# Глобальный реестр плагинов
PluginRegistry = {}


class Plugin:
    """Базовый класс для всех плагинов."""

    name = None  # уникальное имя плагина

    def __init_subclass__(cls, **kwargs):
        """
        Срабатывает автоматически при создании подкласса.
        Регистрирует новый плагин в реестре PluginRegistry.
        """
        super().__init_subclass__(**kwargs)
        if cls.name is None:
            raise ValueError(f"Класс {cls.__name__} должен иметь атрибут name")
        if cls.name in PluginRegistry:
            raise ValueError(f"Плагин с именем '{cls.name}' уже зарегистрирован")

        PluginRegistry[cls.name] = cls

    def execute(self, data):
        """Абстрактный метод: все плагины обязаны его реализовать."""
        raise NotImplementedError("Метод execute() должен быть переопределён в подклассе")


class UpperCasePlugin(Plugin):
    """Плагин: переводит текст в верхний регистр."""

    name = "upper"

    def execute(self, data):
        return data.upper()


class ReversePlugin(Plugin):
    """Плагин: переворачивает строку задом наперёд."""

    name = "reverse"

    def execute(self, data):
        return data[::-1]


class ReplaceSpacesPlugin(Plugin):
    """Плагин: заменяет пробелы на подчёркивания."""

    name = "replace_spaces"

    def execute(self, data):
        return data.replace(" ", "_")


def main():
    print("🔌 Доступные плагины:", list(PluginRegistry.keys()))

    text = input("Введите текст для обработки: ").strip()

    plugin_name = input("Введите имя плагина (upper / reverse / replace_spaces): ").strip()
    if plugin_name not in PluginRegistry:
        print(f"Плагин '{plugin_name}' не найден.")
        return

    plugin_class = PluginRegistry[plugin_name]
    plugin_instance = plugin_class()

    result = plugin_instance.execute(text)
    print("Результат:", result)


if __name__ == "__main__":
    main()
