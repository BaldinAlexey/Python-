# task1.py
import json
from functools import reduce
from collections import defaultdict


# -------------------------------
# 1. Загрузка данных
# -------------------------------
def load_countries(filename):
    """
    Загружает JSON. Ожидается:
    - для countries.json: список строк (названий стран)
    - для countries-data.json: список объектов стран
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return []
    except json.JSONDecodeError:
        print(f"Файл {filename} не является корректным JSON.")
        return []


# -------------------------------
# 2. map / filter / reduce варианты
# -------------------------------
def to_upper_case_map(countries):
    """map: все названия в верхний регистр"""
    return list(map(str.upper, countries))


def filter_contains_land_map(countries):
    """filter: содержит 'LAND' (работаем с UPPER)"""
    return list(filter(lambda c: "LAND" in c, countries))


def filter_exact_length_6_map(countries):
    """filter: ровно 6 символов"""
    return list(filter(lambda c: len(c) == 6, countries))


def filter_length_6_plus_map(countries):
    """filter: длина >= 6"""
    return list(filter(lambda c: len(c) >= 6, countries))


def filter_start_with_e_map(countries):
    """filter: начинаются с 'E' (UPPER)"""
    return list(filter(lambda c: c.startswith("E"), countries))


def north_european_sentence_reduce():
    """reduce: объединить список Северной Европы в одну фразу"""
    nordic = ["Финляндия", "Швеция", "Дания", "Норвегия", "Исландия"]
    # reduce объединяет первые n-1 через запятую, затем добавляем "и X являются..."
    if not nordic:
        return ""
    if len(nordic) == 1:
        return f"{nordic[0]} является страной Северной Европы"
    # создаём "A, B, C" из первых n-1
    prefix = reduce(lambda a, b: f"{a}, {b}", nordic[:-1])
    return f"{prefix} и {nordic[-1]} являются странами Северной Европы"


# -------------------------------
# 3. Реализация без map/filter/reduce (генераторы/циклы)
# -------------------------------
def to_upper_case_manual(countries):
    """Ручная реализация перевода в верхний регистр."""
    result = []
    for c in countries:
        result.append(c.upper())
    return result


def filter_contains_land_manual(countries):
    """Ручной фильтр по 'LAND' (работаем с UPPER)."""
    out = []
    for c in countries:
        if "LAND" in c:
            out.append(c)
    return out


def filter_exact_length_6_manual(countries):
    out = []
    for c in countries:
        if len(c) == 6:
            out.append(c)
    return out


def filter_length_6_plus_manual(countries):
    out = []
    for c in countries:
        if len(c) >= 6:
            out.append(c)
    return out


def filter_start_with_e_manual(countries):
    out = []
    for c in countries:
        if c.startswith("E"):
            out.append(c)
    return out


def north_european_sentence_manual():
    """Тот же reduce, но реализован вручную через цикл."""
    nordic = ["Финляндия", "Швеция", "Дания", "Норвегия", "Исландия"]
    if not nordic:
        return ""
    if len(nordic) == 1:
        return f"{nordic[0]} является страной Северной Европы"
    s = ""
    for i, name in enumerate(nordic):
        if i == 0:
            s += name
        elif i < len(nordic) - 1:
            s += ", " + name
        else:
            s += " и " + name
    s += " являются странами Северной Европы"
    return s


# -------------------------------
# 4. Каррирование и замыкание
# -------------------------------
# каррированная версия через lambda
categorize_curried = lambda pattern: (lambda countries: [c for c in countries if pattern.lower() in c.lower()])

# замыкание (обычная функция возвращает функцию)
def categorize_closure(pattern):
    def inner(countries):
        return [c for c in countries if pattern.lower() in c.lower()]
    return inner


# -------------------------------
# 5. Работа с countries-data.json
# -------------------------------
def sort_by_name(data):
    """Сортировка по полю 'name' (по возрастанию)."""
    return sorted(data, key=lambda x: x.get("name", ""))


def sort_by_capital(data):
    """Сортировка по полю 'capital' (пустые строки в конец)."""
    return sorted(data, key=lambda x: x.get("capital") or "")


def sort_by_population(data, descending=True):
    """Сортировка по полю population. По умолчанию — убывание (большие страны первыми)."""
    return sorted(data, key=lambda x: x.get("population", 0), reverse=descending)


def get_top_languages_with_countries(data, top_n=10):
    """
    Возвращает список топ-N языков с количеством стран и списком стран, где язык используется.
    Формат: [(language, count, [country1, country2, ...]), ...]
    """
    lang_map = defaultdict(set)  # language -> set of country names
    for country in data:
        name = country.get("name", "")
        for lang in country.get("languages", []):
            lang_map[lang].add(name)
    # соберём список (lang, count, [countries])
    lang_list = [(lang, len(countries), sorted(list(countries))) for lang, countries in lang_map.items()]
    # отсортируем по count desc
    lang_list.sort(key=lambda x: x[1], reverse=True)
    return lang_list[:top_n]


def get_top_countries_by_population(data, top_n=10):
    """Возвращает список топ-N стран по населению (объекты)."""
    return sort_by_population(data, descending=True)[:top_n]


# -------------------------------
# 6. main — демонстрация
# -------------------------------
def main():
    # --- countries.json (список строк) ---
    countries = load_countries("countries.json")
    if not countries:
        print("Нет данных countries.json или он пуст.")
    else:
        # приводим все к верхнему регистру (map)
        upper_map = to_upper_case_map(countries)
        print("\nПервые 10 стран (UPPER via map):")
        print(upper_map[:10])

        # фильтры (на UPPER)
        print("\nФильтры через filter (на UPPER):")
        print("contains 'LAND':", filter_contains_land_map(upper_map))
        print("exact length == 6:", filter_exact_length_6_map(upper_map))
        print("length >= 6:", filter_length_6_plus_map(upper_map))
        print("start with 'E':", filter_start_with_e_map(upper_map))

        # reduce
        print("\nReduce sentence (Северная Европа):")
        print(north_european_sentence_reduce())

        # manual implementations
        print("\nРучные реализации (без map/filter/reduce):")
        upper_manual = to_upper_case_manual(countries)
        print("UPPER manual first 10:", upper_manual[:10])
        print("contains 'LAND' manual:", filter_contains_land_manual(upper_manual))
        print("exact length == 6 manual:", filter_exact_length_6_manual(upper_manual))
        print("length >= 6 manual:", filter_length_6_plus_manual(upper_manual))
        print("start with 'E' manual:", filter_start_with_e_manual(upper_manual))
        print("Reduce manual:", north_european_sentence_manual())

        # каррирование и замыкание
        print("\nКаррирование / замыкание:")
        print("land (curried):", categorize_curried("land")(countries)[:10])
        ia_filter = categorize_closure("ia")
        print("ia (closure):", ia_filter(countries)[:10])

    # --- countries-data.json (объекты стран) ---
    data = load_countries("countries-data.json")
    if not data:
        print("\nНет данных countries-data.json или он пуст.")
    else:
        print("\n--- Сортировка (по name) — первые 5: ---")
        print([c.get("name") for c in sort_by_name(data)[:5]])

        print("\n--- Сортировка (по capital) — первые 5: ---")
        print([c.get("capital") for c in sort_by_capital(data)[:5]])

        print("\n--- Топ 10 языков и страны, где на них говорят: ---")
        top_langs = get_top_languages_with_countries(data, top_n=10)
        for lang, count, countries_list in top_langs:
            print(f"{lang}: {count} стран. Примеры: {countries_list[:5]}")

        print("\n--- Топ 10 самых населённых стран: ---")
        top_countries = get_top_countries_by_population(data, top_n=10)
        for c in top_countries:
            print(c.get("name"), "-", c.get("population"))


if __name__ == "__main__":
    main()
