
class Queue:
    """Класс Очередь (FIFO — First In, First Out)."""

    def __init__(self):
        self._items = []

    def enqueue(self, item):
        """Добавить элемент в конец очереди."""
        self._items.append(item)

    def dequeue(self):
        """Удалить и вернуть элемент из начала очереди."""
        if not self._items:
            print(" Очередь пуста")
            return None
        return self._items.pop(0)

    def peek(self):
        """Посмотреть первый элемент без удаления."""
        if not self._items:
            print(" Очередь пуста")
            return None
        return self._items[0]

    def is_empty(self):
        """Проверить, пуста ли очередь."""
        return len(self._items) == 0

    def __str__(self):
        return f"Очередь: {self._items}"


class Stack:
    """Класс Стек (LIFO — Last In, First Out)."""

    def __init__(self):
        self._items = []

    def push(self, item):
        """Добавить элемент на вершину стека."""
        self._items.append(item)

    def pop(self):
        """Удалить и вернуть элемент с вершины стека."""
        if not self._items:
            print(" Стек пуст")
            return None
        return self._items.pop()

    def peek(self):
        """Посмотреть верхний элемент без удаления."""
        if not self._items:
            print(" Стек пуст")
            return None
        return self._items[-1]

    def is_empty(self):
        """Проверить, пуст ли стек."""
        return len(self._items) == 0

    def __str__(self):
        return f"Стек: {self._items}"


def main():
    queue = Queue()
    stack = Stack()

    while True:
        print("\n Меню:")
        print("1 — Добавить в очередь")
        print("2 — Удалить из очереди")
        print("3 — Показать очередь")
        print("4 — Добавить в стек")
        print("5 — Удалить из стека")
        print("6 — Показать стек")
        print("7 — Выйти")

        choice = input("Выберите действие: ")

        if choice == "1":
            value = input("Введите элемент: ")
            queue.enqueue(value)

        elif choice == "2":
            removed = queue.dequeue()
            if removed is not None:
                print(f"Удалено из очереди: {removed}")

        elif choice == "3":
            print(queue)

        elif choice == "4":
            value = input("Введите элемент: ")
            stack.push(value)

        elif choice == "5":
            removed = stack.pop()
            if removed is not None:
                print(f"Удалено из стека: {removed}")

        elif choice == "6":
            print(stack)

        elif choice == "7":
            print(" Выход.")
            break

        else:
            print(" Неверный выбор, попробуйте снова.")


if __name__ == "__main__":
    main()
