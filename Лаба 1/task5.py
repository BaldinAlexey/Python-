import string
from collections import Counter
from pathlib import Path


def read_text_from_file(min_length: int = 100) -> str:
    """
    Считывает текст из файла
    """
    while True:
        file_path = input("Введите путь к текстовому файлу (.txt): ").strip()
        path = Path(file_path)

        if not path.is_file():
            print("Ошибка: файл не найден. Попробуйте ещё раз.")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            print("Ошибка: файл не в кодировке UTF-8.")
            continue

        if len(text) < min_length:
            print(f"Ошибка: текст слишком короткий (< {min_length} символов).")
            continue

        return text


def clean_text(text: str) -> list[str]:
    """Удаляет знаки препинания, приводит к нижнему регистру, возвращает список слов."""
    text = text.lower()

    # список лишних символов, которые хотим убрать
    remove_chars = string.punctuation + "—«»…"

    # заменяем каждый знак препинания на пробел
    for ch in remove_chars:
        text = text.replace(ch, " ")

    return text.split()


def count_characters(text: str):
    """Подсчёт количества символов (с пробелами и без)."""
    total_with_spaces = len(text)
    total_without_spaces = len(text.replace(" ", ""))
    return total_with_spaces, total_without_spaces


def get_statistics(word_list: list[str]):
    """
    Возвращает статистику:
    - общее количество слов
    - 5 самых частых слов
    - 5 самых длинных слов
    - среднюю длину слова
    """
    total_words = len(word_list)
    counter = Counter(word_list)

    most_common = counter.most_common(5)

    longest_words = sorted(word_list, key=len, reverse=True)
    unique_longest = []
    for word in longest_words:
        if word not in unique_longest:
            unique_longest.append(word)
        if len(unique_longest) == 5:
            break

    avg_length = (
        sum(len(word) for word in word_list) / total_words
        if total_words > 0
        else 0
    )

    return total_words, most_common, unique_longest, avg_length


def main():
    text = read_text_from_file()
    word_list = clean_text(text)

    total_with_spaces, total_without_spaces = count_characters(text)
    total_words, most_common, longest_words, avg_length = get_statistics(word_list)

    print("\n Результаты анализа:")
    print(f"Общее количество символов: {total_with_spaces} (без пробелов: {total_without_spaces})")
    print(f"Количество словоформ: {total_words}")

    print("Самые частые словоформы:")
    for word, freq in most_common:
        print(f" - '{word}': {freq} раз(а)")

    print("Самые длинные словоформы:")
    for word in longest_words:
        print(f" - '{word}' ({len(word)} букв)")

    print(f"Средняя длина словоформы: {avg_length:.1f} символа") # форматировать с одним знаком после запятой (1f)


if __name__ == "__main__":
    main()
