import json
import os

class Transaction:
    """Класс для представления одной финансовой операции (доход или расход)."""

    def __init__(self, description, amount, t_type, category):
        self.description = description
        self.amount = amount
        self.t_type = t_type  # "доход" или "расход"
        self.category = category

    def to_dict(self):
        """Преобразовать объект в словарь для сохранения в JSON."""
        return {
            "description": self.description,
            "amount": self.amount,
            "t_type": self.t_type,
            "category": self.category
        }

    @staticmethod
    def from_dict(data):
        """Создать объект Transaction из словаря (при загрузке из JSON)."""
        return Transaction(
            data.get("description", ""),
            data.get("amount", 0),
            data.get("t_type", "расход"),
            data.get("category", "")
        )

    def __str__(self):
        """Человекочитаемое представление транзакции."""
        sign = "+" if self.t_type == "доход" else "-"
        return f"{sign}{self.amount} | {self.description} #{self.category}"

    def __repr__(self):
        """Техническое представление объекта для отладки."""
        return (f"Transaction(description={self.description!r}, "
                f"amount={self.amount}, t_type={self.t_type!r}, "
                f"category={self.category!r})")


class BudgetTracker:
    """Класс для хранения всех операций и управления балансом."""

    def __init__(self, filename="transactions.json"):
        self.filename = filename
        self.transactions = []
        self.load_transactions()

    def add_transaction(self, description, amount, t_type, category):
        """Добавить новую операцию (доход или расход)."""
        transaction = Transaction(description, amount, t_type, category)
        self.transactions.append(transaction)
        print(" Операция добавлена!")

    def get_balance(self):
        """Посчитать текущий баланс."""
        balance = 0
        for t in self.transactions:
            if t.t_type == "доход":
                balance += t.amount
            else:
                balance -= t.amount
        return balance

    def list_transactions(self):
        """Вывести список всех операций."""
        if not self.transactions:
            print(" Список операций пуст.")
        else:
            for i, t in enumerate(self.transactions):
                print(f"{i}: {t}")
            print(f"\n Текущий баланс: {self.get_balance()}")

    def save_transactions(self):
        """Сохранить операции в JSON-файл."""
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self.transactions], f, ensure_ascii=False, indent=2)

    def load_transactions(self):
        """Загрузить операции из JSON-файла (если он существует)."""
        if os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.transactions = [Transaction.from_dict(d) for d in data]
        else:
            self.transactions = []


def main():
    tracker = BudgetTracker()

    while True:
        print("\n Меню:")
        print("1 — Добавить операцию")
        print("2 — Показать все операции")
        print("3 — Показать баланс")
        print("4 — Выйти")

        choice = input("Выберите действие: ")

        if choice == "1":
            desc = input("Описание операции: ")
            try:
                amount = float(input("Сумма: "))
            except ValueError:
                print(" Ошибка: сумма должна быть числом.")
                continue
            t_type = input("Тип (доход/расход): ").strip().lower()
            if t_type not in ("доход", "расход"):
                print(" Некорректный тип операции.")
                continue
            cat = input("Категория: ")
            tracker.add_transaction(desc, amount, t_type, cat)

        elif choice == "2":
            tracker.list_transactions()

        elif choice == "3":
            print(f" Текущий баланс: {tracker.get_balance()}")

        elif choice == "4":
            tracker.save_transactions()
            print(" Операции сохранены. Выход.")
            break

        else:
            print(" Неверный выбор, попробуйте снова.")


if __name__ == "__main__":
    main()
