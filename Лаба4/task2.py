"""
Задание 2: Анализ и визуализация данных Titanic (pandas + seaborn)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_titanic():
    """Загружает датасет Titanic из seaborn."""
    df = sns.load_dataset('titanic')
    print("Данные успешно загружены.")
    return df


def analyze_structure(df):
    """Выводит типы данных и количество пропусков по столбцам."""
    print("\n--- Анализ структуры данных ---")
    print(df.info())
    print("\nКоличество пропусков:")
    print(df.isnull().sum())
    print("-" * 50)


def create_features(df):
    """Добавляет новые признаки age_group и family_size."""
    df = df.copy()
    # Обработка пропусков в возрасте
    df['age'] = df['age'].fillna(df['age'].median())

    # Группы по возрасту
    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 18, 30, 50, 100],
        labels=['child', 'young', 'adult', 'senior']
    )

    # Размер семьи
    df['family_size'] = df['sibsp'].fillna(0) + df['parch'].fillna(0)
    print("Добавлены признаки: 'age_group', 'family_size'")
    return df


def group_survival(df):
    """Считает выживаемость по полу и классу."""
    grouped = df.groupby(['sex', 'class'])['survived'].mean().reset_index()
    grouped.rename(columns={'survived': 'survival_rate'}, inplace=True)
    print("\n--- Средняя выживаемость по полу и классу ---")
    print(grouped)
    return grouped


def visualize_titanic(df):
    """Создаёт несколько графиков для анализа данных Titanic."""
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Распределение возрастов по полу
    sns.histplot(data=df, x='age', hue='sex', multiple='stack', ax=axes[0, 0])
    axes[0, 0].set_title("Распределение возрастов по полу")

    # Выживаемость по классам каюты
    sns.barplot(data=df, x='class', y='survived', ax=axes[0, 1])
    axes[0, 1].set_title("Выживаемость по классам каюты")

    # Корреляционная матрица
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title("Корреляция числовых признаков")

    # Стоимость билета по классам (ящик с усами)
    sns.boxplot(data=df, x='class', y='fare', ax=axes[1, 1])
    axes[1, 1].set_title("Стоимость билета по классам")

    plt.tight_layout()
    plt.show()


def compare_types_performance(df):
    """Сравнивает скорость группировки с обычными и категориальными типами."""
    import time

    # Обычные типы
    t0 = time.time()
    df.groupby('class')['survived'].mean()
    normal_time = time.time() - t0

    # Категориальные типы
    df_cat = df.copy()
    df_cat['class'] = df_cat['class'].astype('category')
    t0 = time.time()
    df_cat.groupby('class')['survived'].mean()
    cat_time = time.time() - t0

    print(f"\nСкорость группировки:")
    print(f"Обычные типы: {normal_time:.6f} сек")
    print(f"Категориальные типы: {cat_time:.6f} сек")
    print(f"Категориальные быстрее в {normal_time / cat_time:.2f} раз.")


def main():
    # 1. Загрузка данных
    df = load_titanic()

    # 2. Анализ структуры
    analyze_structure(df)

    # 3. Создание новых признаков
    df = create_features(df)

    # 4. Группировка по полу и классу
    group_survival(df)

    # 5. Сравнение типов
    compare_types_performance(df)

    # 6. Визуализация
    visualize_titanic(df)


if __name__ == "__main__":
    main()
