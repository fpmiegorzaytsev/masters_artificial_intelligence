# Masters Artificial Intelligence — Вступительное задание

Репозиторий содержит решение тестового задания в магистратуру по направлению **Artificial Intelligence**, команда **Alignment**.

Проект включает:
- обучение reward-моделей для двух уровней;
- обучение основной модели;
- комментарии к решению (в `src/README.md` и ноутбуках).

---

## Установка

Активировав venv или conda environment, выполните:

```bash
git clone https://github.com/fpmiegorzaytsev/masters_artificial_intelligence.git
cd masters_artificial_intelligence
python -m pip install -r requirements.txt
```

## Запуск

### Уровень 1

#### Обучение reward-модели:

Откройте и выполните все ячейки в `reward_model_level_1.ipynb` Jupyter-ноутбуке:

#### Alignment основной модели:

```bash
cd src
CUDA_VISIBLE_DEVICES=<доступные GPU устройства> accelerate launch train.py
```

### Уровень 2

#### Обучение reward-модели:

Откройте и выполните все ячейки в `reward_model_level_2.ipynb` Jupyter-ноутбуке 

Запускать проект следует в указанном порядке
