import cv2
from ultralytics import YOLO  # Используем правильный импорт

# Загрузка модели YOLO
model = YOLO('C:/ITinAPK/runs/detect/train_experiment32/weights/best.pt')  # Укажите путь к файлу вашей обученной модели       

# Открытие видео
cap = cv2.VideoCapture('C:/ITinAPK/video_2024-11-06_18-48-50.mp4')
#cap = cv2.VideoCapture('C:/ITinAPK/IMG_9217.mp4')

# Проверка, что видео открылось
if not cap.isOpened():
    print("Ошибка при открытии видео")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  

    
    annotated_frame = results[0].plot()  

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
        # Вычисляем границы кузова
        truck_left = truck_box[0] - truck_box[2] / 2
        truck_right = truck_box[0] + truck_box[2] / 2
        truck_top = truck_box[1] - truck_box[3] / 2
        truck_bottom = truck_box[1] + truck_box[3] / 2

        # Вычисляем границы потока зерна
        grain_left = grain_box[0] - grain_box[2] / 2
        grain_right = grain_box[0] + grain_box[2] / 2
        grain_top = grain_box[1] - grain_box[3] / 2
        grain_bottom = grain_box[1] + grain_box[3] / 2

        # Порог в пикселях для определения близости границ
        threshold = 10

        # Флаг для приостановки воспроизведения
        pause = False

        # Проверяем, находится ли зерно близко к любой из границ кузова
        if abs(grain_top - truck_top) < threshold:
            print("Зерно почти высыпается за верхнюю границу кузова!")
            pause = True
        if abs(grain_top - truck_bottom) < threshold:
            print("Зерно почти высыпается за нижнюю границу кузова!")
            pause = True
        if abs(grain_left - truck_left) < threshold:
            print("Зерно почти высыпается за левую границу кузова!")
            pause = True
        if abs(grain_right - truck_right) < threshold:
            print("Зерно почти высыпается за правую границу кузова!")
            pause = True

        # Приостановка видео при обнаружении границы
        if pause:
            print("Видео приостановлено. Нажмите любую клавишу, чтобы продолжить...")
            while True:
                # Ожидание нажатия любой клавиши для продолжения
                if cv2.waitKey(0) & 0xFF != 255:
                    break

    # Отображаем аннотированный кадр
    cv2.imshow('Frame', annotated_frame)

    # Прервать цикл при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрыть видео и окна
cap.release()
cv2.destroyAllWindows()
