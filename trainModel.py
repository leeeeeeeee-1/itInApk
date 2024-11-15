from ultralytics import YOLO
import matplotlib.pyplot as plt
import yaml

# Путь к вашему конфигурационному файлу
data_path = "C:/ITinAPK/data.yaml"

# Создаем объект модели YOLO
model = YOLO('yolov8n.pt')  # Загружаем предобученную модель (можно использовать yolov8s.pt, yolov8m.pt и т.д.)

# Обучение модели
model.train(
    data=data_path,  # Путь к файлу .yaml
    epochs=50,  # Количество эпох
    batch=16,  # Размер батча
    imgsz=640,  # Размер изображений
    project='runs/detect',  # Папка для сохранения результатов
    name='train_experiment'  # Имя эксперимента
)

metrics_train = model.train()
print("Train Metrics:", metrics_train)


metrics_test = model.val()
print("Test Metrics:", metrics_test)


