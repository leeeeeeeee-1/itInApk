from ultralytics import YOLO

# Создаем объект модели YOLO
model = YOLO('runs/detect/train_experiment32/weights/best.pt')  # Путь к обученной модели

# Запускаем проверку модели на валидационном наборе данных
metrics_test = model.val(
    data="C:/ITinAPK/data.yaml",  # Путь к файлу конфигурации
    batch=16,  # Размер батча для проверки
    imgsz=640,  # Размер изображений
    save_dir='runs/val_results'  # Папка для сохранения результатов валидации
)

print("Test Metrics:", metrics_test)
