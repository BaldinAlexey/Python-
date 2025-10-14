import json
import os

class Task:
    """Класс для представления одной задачи."""
    def __init__(self, description, category, done=False):
        self.description = description
        self.category = category
        self.done = done

    def mark_done(self):
        """Отметить задачу как выполненную."""
        self.done = True

    def to_dict(self):
        """Преобразовать объект в словарь (для сохранения в JSON)."""
        return {
            "description": self.description,
            "category": self.category,
            "done": self.done
        }

    @staticmethod
    def from_dict(data):
        """Создать объект Task из словаря (при загрузке из JSON)."""
        return Task(
            data.get("description", ""),
            data.get("category", ""),
            data.get("done", False)
        )

    def __str__(self):
        """Человекочитаемое представление задачи."""
        status = "[x]" if self.done else "[ ]"
        return f"{status} {self.description} #{self.category}"

    def __repr__(self):
        """Официальное представление (для отладки)."""
        return f"Task(description={self.description!r}, category={self.category!r}, done={self.done})"


class TaskTracker:
    """Класс для управления списком задач и их сохранения."""
    def __init__(self, filename="tasks.json"):
        self.filename = filename
        self.tasks = []
        self.load_tasks()

    def add_task(self, description, category):
        """Добавить новую задачу."""
        task = Task(description, category)
        self.tasks.append(task)
        print(" Задача добавлена!")

    def mark_task_done(self, index):
        """Отметить задачу как выполненную по индексу."""
        if 0 <= index < len(self.tasks):
            self.tasks[index].mark_done()
            print(" Задача отмечена как выполненная.")
        else:
            print(" Неверный индекс.")

    def list_tasks(self):
        """Вывести список всех задач."""
        if not self.tasks:
            print(" Список задач пуст.")
        else:
            for i, task in enumerate(self.tasks):
                print(f"{i}: {task}")

    def search_by_category(self, category):
        """Вывести задачи по категории."""
        found = [t for t in self.tasks if t.category == category]
        if found:
            print(f" Задачи в категории '{category}':")
            for t in found:
                print(t)
        else:
            print(" Ничего не найдено.")

    def save_tasks(self):
        """Сохранить задачи в JSON-файл."""
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self.tasks], f, ensure_ascii=False, indent=2)

    def load_tasks(self):
        """Загрузить задачи из JSON-файла (если он существует)."""
        if os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.tasks = [Task.from_dict(d) for d in data]
        else:
            self.tasks = []


def main():
    tracker = TaskTracker()

    while True:
        print("\n Меню:")
        print("1 — Добавить задачу")
        print("2 — Отметить задачу как выполненную")
        print("3 — Показать все задачи")
        print("4 — Поиск по категории")
        print("5 — Выйти")

        choice = input("Выберите действие: ")

        if choice == "1":
            desc = input("Описание задачи: ")
            cat = input("Категория: ")
            tracker.add_task(desc, cat)

        elif choice == "2":
            tracker.list_tasks()
            try:
                idx = int(input("Введите индекс задачи: "))
                tracker.mark_task_done(idx)
            except ValueError:
                print(" Введите корректный номер!")

        elif choice == "3":
            tracker.list_tasks()

        elif choice == "4":
            cat = input("Введите категорию: ")
            tracker.search_by_category(cat)

        elif choice == "5":
            tracker.save_tasks()
            print(" Задачи сохранены. Выход.")
            break

        else:
            print(" Неверный выбор, попробуйте снова.")


if __name__ == "__main__":
    main()
