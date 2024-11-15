import cv2
from ultralytics import YOLO  # Используем правильный импорт

# Загрузка модели YOLO
model = YOLO('C:/ITinAPK/runs/detect/train_experiment32/weights/best.pt')  # Укажите путь к файлу вашей обученной модели       

# Открытие видео
cap = cv2.VideoCapture('C:/ITinAPK/video_2024-11-06_18-48-50.mp4')

# Проверка, что видео открылось
if not cap.isOpened():
    print("Ошибка при открытии видео")
    exit()

# Обработка кадров
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Применение модели к кадру
    results = model(frame)  # Получаем результаты

    # Рендеринг результатов
    annotated_frame = results[0].plot()  # Возвращает изображение с аннотациями

    # Получаем координаты для объектов Truck и InTruck
    truck_box = None
    grain_box = None
    for obj in results[0].boxes:  # Итерируемся по объектам боксов
        class_id = int(obj.cls.item())  # Получаем ID класса из `obj.cls`
        if class_id == 0:  # Truck
            truck_box = obj.xywh[0]  # Преобразуем в массив [x_center, y_center, width, height]
        elif class_id == 1:  # InTruck
            grain_box = obj.xywh[0]

    if truck_box is not None and grain_box is not None:
        # Вычисляем координаты границ кузова
        truck_top = truck_box[1] - truck_box[3] / 2  # Верхняя граница кузова

        # Вычисляем верхнюю границу потока зерна
        grain_top = grain_box[1] - grain_box[3] / 2

        # Определяем порог для "границы" — момент, когда зерно почти выходит за кузов
        threshold = 10  # Порог в пикселях, можно настроить

        # Проверяем, находится ли верхняя граница потока зерна близко к верхней границе кузова
        if abs(grain_top - truck_top) < threshold:
            print("Зерно почти высыпается за кузов!")

    # Отображаем аннотированный кадр
    cv2.imshow('Frame', annotated_frame)

    # Прервать цикл при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрыть видео и окна
cap.release()
cv2.destroyAllWindows()
