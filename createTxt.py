import os
import xml.etree.ElementTree as ET

# Определяем классы
classes = ["Truck", "InTruck"]

def convert_annotation(xml_file, output_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Извлекаем размеры изображения
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)
    
    # Создаем строки для YOLO формата
    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in classes:
            class_id = classes.index(class_name)
            
            # Извлекаем координаты ограничивающей рамки
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)
            
            # Преобразуем в формат YOLO
            x_center = (xmin + xmax) / 2 / image_width
            y_center = (ymin + ymax) / 2 / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            # Форматируем строку
            yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    # Сохраняем результат в текстовом файле
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(xml_file))[0] + '.txt')
    with open(output_file, 'w') as f:
        f.write("\n".join(yolo_lines))

# Основная функция для обработки всех XML файлов
def main(xml_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            full_xml_path = os.path.join(xml_folder, xml_file)
            convert_annotation(full_xml_path, output_folder)

# Укажите путь к вашим файлам XML и папку, куда сохранять YOLO аннотации
xml_folder = "C:/Users/RVler/Downloads/video_2024-11-06_18-48-50_000"
output_folder = "C:/Users/RVler/OneDrive/Рабочий стол/ITinAPK/dataset"
main(xml_folder, output_folder)


