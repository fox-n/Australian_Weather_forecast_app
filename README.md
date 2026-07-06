# 🌦️ Rain in Australia — прогноз дощу за допомогою Streamlit

Навчальний проєкт з розділу **MLOps**: розгортання моделі машинного навчання
у вигляді веб-застосунку за допомогою [Streamlit](https://streamlit.io/).

Застосунок передбачає, чи піде дощ завтра (`RainTomorrow`), на основі
сьогоднішніх погодних показників для обраної локації в Австралії
(датасет [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)).

🔗 **Задеплоєний застосунок:** _додати посилання після деплою на Streamlit Cloud_

> Зверни увагу: застосунки на безкоштовному Streamlit Cloud "засинають"
> після періоду неактивності — перший запуск після паузи може зайняти
> кілька секунд.

## Структура проєкту

```
├── data/               # дані, використані для навчання моделі
├── images/             # зображення для застосунку
├── models/             # навчена ML-модель (pipeline: imputer, scaler, encoder, model)
├── app.py              # основний файл Streamlit-застосунку
├── requirements.txt    # залежності проєкту
└── README.md
```

## Модель

Модель — логістична регресія (`LogisticRegression`), навчена на датасеті
Rain in Australia. Разом з моделлю збережено весь пайплайн препроцесингу:

- `imputer` — заповнення пропусків у числових колонках (`SimpleImputer`)
- `scaler` — нормалізація числових колонок (`MinMaxScaler`)
- `encoder` — кодування категоріальних колонок (`OneHotEncoder`)

## Як запустити локально

1. Клонувати репозиторій:
   ```bash
   git clone https://github.com/<твій-юзернейм>/<назва-репо>.git
   cd <назва-репо>
   ```

2. Створити та активувати віртуальне середовище:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Встановити залежності:
   ```bash
   pip install -r requirements.txt
   ```

4. Запустити застосунок:
   ```bash
   streamlit run app.py
   ```

## Технології

- Python 3.13
- scikit-learn 1.5.2
- Streamlit
- pandas / numpy
