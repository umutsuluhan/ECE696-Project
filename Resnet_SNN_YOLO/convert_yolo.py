import ast
import cv2
import os

category_map = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    7: 6,  # tricycle
    8: 7,  # awning-tricycle
    9: 8,  # bus
    10: 9  # motor
}

def convert_annotation_file(original_annotations_file, img_h, img_w, yolo_annotations_per_frame):
    file_handler = open(original_annotations_file, "r")
    original_annotations = file_handler.readlines()
    file_handler.close()

    original_annotations_tuple = []
    for item in original_annotations:
        original_annotations_tuple.append(ast.literal_eval(item))

    original_annotations_dict = {}

    for item in original_annotations_tuple:
        if int(item[0]) not in original_annotations_dict:
            original_annotations_dict[int(item[0])] = []

        if int(item[7]) == 0:
            continue

        original_annotations_dict[int(item[0])].append((int(item[7]), int(item[2]), int(item[3]), int(item[4]), int(item[5])))

    for key, value in original_annotations_dict.items():
        if key not in yolo_annotations_per_frame:
            yolo_annotations_per_frame[key] = []
        
        for annotation in value:
            label = annotation[0]
            yolo_class = category_map.get(label, -1)
            if yolo_class == -1:
                continue

            x, y = int(annotation[1]), int(annotation[2])
            w, h = int(annotation[3]), int(annotation[4])

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w = w / img_w
            h = h / img_h

            yolo_annotations_per_frame[key].append((yolo_class, x_center, y_center, w, h))
    return yolo_annotations_per_frame

def convert_whole_folder(folder_path, task):
    yolo_annotations_per_frame = {}

    # Extract heigth and width for the image
    sample_image_path = "./datasets/original_Visdrone/VisDrone2019-VID-" + str(task) + "/sequences/" + folder_path +"/0000001.jpg"
    img = cv2.imread(sample_image_path)
    img_h, img_w, _ = img.shape 
    # Convert the image
    yolo_annotations_per_frame = convert_annotation_file('./datasets/original_Visdrone/VisDrone2019-VID-' + task + '/annotations/' + folder_path + '.txt',  img_h, img_w, yolo_annotations_per_frame)

    # Defined save paths 
    annotated_save_path = './datasets/VisDrone_YOLO/annotated/' + task + '/'
    original_save_path = './datasets/VisDrone_YOLO/images/' + task + '/'
    label_save_path = './datasets/VisDrone_YOLO/labels/' + task + '/'

    # Save each frame, annotated framge (for debugging purposes) and labels
    for key, value in yolo_annotations_per_frame.items():
        original_image_id = str(key).zfill(7)
        save_label_id = folder_path + original_image_id
        save_label_path = label_save_path + save_label_id + ".txt"
        save_label_file_handler = open(save_label_path, "w")

        image_path = "./datasets/original_Visdrone/VisDrone2019-VID-" + task + "/sequences/"+ folder_path + "/" + original_image_id + ".jpg"
        img = cv2.imread(image_path)
        h, w, _ = img.shape 
        annotated_img = img.copy()
        
        for annotation in value:
            class_id, x_center, y_center, width, height = annotation

            save_label_file_handler.write(str(class_id) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) + "\n")

            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            color = (0, 255, 0)
            thickness = 2

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
            label = f"Class {class_id}"
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        save_image_id = folder_path + original_image_id + ".jpg"
        original_merged_save_path = os.path.join(original_save_path, save_image_id)
        annotated_merged_save_path = os.path.join(annotated_save_path, save_image_id)
        save_label_file_handler.close()
        cv2.imwrite(original_merged_save_path, img)
        cv2.imwrite(annotated_merged_save_path, annotated_img)

# Task can "train", "test" or "val"
task = "train"
folder_list = sorted(os.listdir("./datasets/original_Visdrone/VisDrone2019-VID-" + str(task) + "/sequences"))

# Create folders for YOLO formatted data
if not os.path.exists('./datasets/VisDrone_YOLO/annotated/'):
    os.mkdir('./datasets/VisDrone_YOLO/annotated/')

if not os.path.exists('./datasets/VisDrone_YOLO/images/'):
    os.mkdir('./datasets/VisDrone_YOLO/images/')

if not os.path.exists('./datasets/VisDrone_YOLO/labels/'):
    os.mkdir('./datasets/VisDrone_YOLO/labels/')

if not os.path.exists('./datasets/VisDrone_YOLO/annotated/' + str(task) + '/'):
    os.mkdir('./datasets/VisDrone_YOLO/annotated/' + str(task) + '/')

if not os.path.exists('./datasets/VisDrone_YOLO/images/' + str(task) + '/'):
    os.mkdir('./datasets/VisDrone_YOLO/images/' + str(task) + '/')

if not os.path.exists('./datasets/VisDrone_YOLO/labels/' + str(task) + '/'):
    os.mkdir('./datasets/VisDrone_YOLO/labels/' + str(task) + '/')

for folder in folder_list:
    # Convert each folder to YOLO format
    convert_whole_folder(folder, task)