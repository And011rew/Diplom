import pixellib
import torch
import colorsys
import random
from collections import Counter
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.instance import instance_segmentation
import os
import cv2
import math
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.messagebox import showerror, showwarning, showinfo
from functools import partial
from datetime import timedelta
import time
from PIL import Image, ImageTk
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model



#video = cv2.VideoCapture("C:/Users/AndreyBayborodin/Desktop/dplm.mp4")

#ret, frame1 = video.read()
#ret, frame2 = video.read()

#while video.isOpened():
#    if ret:
#        difference = cv2.absdiff(frame1, frame2)
#        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
#        blur = cv2.GaussianBlur(gray, (5,5), 0)
#        _, threshold = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
#        dilate = cv2.dilate(threshold, None, iterations=3)
#        contour, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        max=0
#        sel_countour=None
#        for countour in contour:
#            if countour.shape[0]>max and countour.shape[0] > 1600:
#                sel_countour=countour
#                max=countour.shape[0]
#            print(countour.shape[0])
#        cv2.drawContours(frame1, sel_countour, -1, (0, 0, 255), 2)
#        cv2.imshow("image", frame1)
#        frame1 = frame2
#        ret, frame2 = video.read()
#        if cv2.waitKey(40) == ord('q'):
#            break

#video.release()



## Чтение видеофайла
#video_path = "C:/Users/AndreyBayborodin/Desktop/test_2.mp4"

#cap = cv2.VideoCapture(video_path)

## Получение первого кадра для инициализации
#ret, frame = cap.read()
#prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

## Цикл обработки каждого кадра видео
#while True:
#    # Чтение текущего кадра
#    ret, frame = cap.read()
#    if not ret:
#        break

#    # Предобработка текущего кадра
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    gray = cv2.GaussianBlur(gray, (5, 5), 0)

#    # Вычисление разности между текущим и предыдущим кадром
#    frame_diff = cv2.absdiff(prev_gray, gray)

#    # Применение порогового значения для выделения движущихся объектов
#    _, thresh = cv2.threshold(frame_diff, 55, 255, cv2.THRESH_BINARY)

#    # Поиск контуров на пороговом изображении
#    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE )


#    # Отрисовка контуров на исходном кадре
#    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

#    # Отображение кадра с выделенными контурами
#    cv2.imshow("Motion Detection", frame)

#    # Обновление предыдущего кадра
#    prev_gray = gray

#    # Задержка в 30 миллисекунд
#    time.sleep(0.03)

#    # Прерывание цикла по нажатию клавиши "q"
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

## Освобождение ресурсов
#cap.release()
#cv2.destroyAllWindows()


#video = cv2.VideoCapture("C:/Users/AndreyBayborodin/Desktop/test_short_2.mp4")

#ret, frame1 = video.read()
#ret, frame2 = video.read()

#while True:
#    if ret:
#        difference = cv2.absdiff(frame1, frame2)
#        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
#        blur = cv2.GaussianBlur(gray, (5,5), 0)
#        _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#        dilate = cv2.dilate(threshold, None, iterations=3)
#        contour, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#        max=0
#        sel_countour=None
#        for countour in contour:            
#            if countour.shape[0]>max and countour.shape[0]>1600:
#                sel_countour=countour
#                max=countour.shape[0]
#            print(max)
#        cv2.drawContours(frame1, sel_countour, -1, (0, 0, 255), 2)
#        cv2.imshow("image", frame1)
#        frame1 = frame2
#        ret, frame2 = video.read()
#    if cv2.waitKey(40) == ord('q'):
#        break

#video.release()
#video.destroyAllWindows()



def compare_images(image1_path, image2_path):
    # Загрузка предобученной модели VGG16
    model = VGG16(weights='imagenet', include_top=False)

    # Загрузка и предобработка изображений
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    x1 = np.expand_dims(img1, axis=0)
    x2 = np.expand_dims(img2, axis=0)
    x1 = preprocess_input(x1)
    x2 = preprocess_input(x2)

    # Извлечение признаков изображений с помощью модели VGG16
    features1 = model.predict(x1)
    features2 = model.predict(x2)

    # Вычисление косинусного расстояния между признаками
    dot_product = np.dot(features1.flatten(), features2.flatten())
    norm_product = np.linalg.norm(features1.flatten()) * np.linalg.norm(features2.flatten())
    similarity = dot_product / norm_product

    # Вывод процента схожести
    similarity = similarity * 100
    print("Процент схожести: ", similarity)
    return similarity

##Путь к изображениям
#image1_path = "C:/Users/AndreyBayborodin/Desktop/test_det/extract_img/El(1)_frame(30)_id(1).jpg"
#image2_path = "C:/Users/AndreyBayborodin/Desktop/test_det/extract_img/El(1)_frame(89)_id(1).jpg"
##/////////////////////////////////////////////////////////////////////////////////////


## Вызов функции для сравнения изображений
#compare_images(image1_path, image2_path)



#/////////////////////////////////
#сравнение изображений
#/////////////////////////////////
#def compare_images(image1_path, image2_path):
#    # Загрузка предобученной модели VGG16
#    model = VGG16(weights='imagenet', include_top=False)

#    # Загрузка и предобработка изображений
#    img1 = image.load_img(image1_path, target_size=(300, 300))
#    img2 = image.load_img(image2_path, target_size=(300, 300))
#    x1 = image.img_to_array(img1)
#    x1 = np.expand_dims(x1, axis=0)
#    x1 = preprocess_input(x1)
#    x2 = image.img_to_array(img2)
#    x2 = np.expand_dims(x2, axis=0)
#    x2 = preprocess_input(x2)

#    # Извлечение признаков изображений с помощью модели VGG16
#    features1 = model.predict(x1)
#    features2 = model.predict(x2)

#    # Вычисление косинусного расстояния между признаками
#    dot_product = np.dot(features1.flatten(), features2.flatten())
#    norm_product = np.linalg.norm(features1.flatten()) * np.linalg.norm(features2.flatten())
#    similarity = dot_product / norm_product

#    # Вывод процента схожести
#    similarity = similarity * 100
#    print("Процент схожести: ", similarity)
#    return similarity

# Путь к изображениям
#image1_path = "C:/Users/AndreyBayborodin/Desktop/testirou/extract_img/El(1)_frame(9)_id(1).jpg"
#image2_path = "C:/Users/AndreyBayborodin/Desktop/testirou/extract_img/El(1)_frame(10).jpg"
#/////////////////////////////////////////////////////////////////////////////////////


## Вызов функции для сравнения изображений
#compare_images(image1_path, image2_path)

#def assign_object_ids(video_path):
#    cap = cv2.VideoCapture(video_path)
#    ret, frame1 = cap.read()

#    segment_image = instance_segmentation(infer_speed = "rapid")
#    segment_image.load_model("mask_rcnn_coco.h5", confidence= 0.8)  # Загрузка модели Mask R-CNN

#    # Задаем начальные значения идентификаторов объектов
#    object_ids = {}
#    object_id_counter = 1

#    frame_id = 1

#    while ret:
#        ret, frame2 = cap.read()
#        if not ret:
#            break

#        # Выполняем сегментацию объектов на кадре
#        result = segment_image.segmentFrame(frame2, show_bboxes=False)

#        # Получаем маску сегментации объектов
#        masks = result[0]['masks']

#        # Получаем боксы объектов
#        boxes = result[0]['rois']

#        for i in range(masks.shape[2]):
#            # Вычисляем процент площади маски объекта
#            mask_area = np.sum(masks[:, :, i])
#            total_area = masks.shape[0] * masks.shape[1]
#            percent_area = (mask_area / total_area) * 100

#            if percent_area >= 70:
#                # Если объект уже имеет идентификатор, используем его
#                if frame_id in object_ids:
#                    object_id = object_ids[frame_id]
#                else:
#                    # Если объект еще не имеет идентификатора, присваиваем новый
#                    object_id = object_id_counter
#                    object_id_counter += 1
#                    object_ids[frame_id] = object_id

#                # Выводим идентификатор объекта на кадр видео
#                cv2.putText(frame2, str(object_id), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#        cv2.imshow("Video", frame2)
#        cv2.waitKey(1)

#        frame_id += 1

#    cap.release()
#    cv2.destroyAllWindows()

## Пример использования
#video_path = "C:/Users/AndreyBayborodin/Desktop/fishing.mp4"
##assign_object_ids(video_path)



#/////////////////////////////////
#сравнение изображений
#/////////////////////////////////
def compare_objects(image1, image2):
    img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    detector = cv2.SIFT_create()

    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Вычисление процента совпадения
    percent_match = len(good_matches) / len(keypoints1) * 100

    #print("Количество совпадений:", len(good_matches))
    print("Процент совпадения:", percent_match)
    return percent_match

    #img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)

    #cv2.imshow("Совпадения", img_matches)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Пример использования
#image1_path = "C:/Users/AndreyBayborodin/Desktop/test_det/extract_img/El(1)_frame(70)_id(1).jpg"
#image2_path = "C:/Users/AndreyBayborodin/Desktop/test_det/extract_img/El(1)_frame(82)_id(1).jpg"
##image1_path = "C:/Users/AndreyBayborodin/Desktop/test_det/extract_img/box_extract_1_6.jpg"
##image2_path = "C:/Users/AndreyBayborodin/Desktop/test_det/extract_img/box_extract_1_7.jpg"
#compare_objects(image1_path, image2_path)


#/////////////////////////////////
#присвоение идентификаторов
#/////////////////////////////////
#video_path = "fishing.mp4"
#output_path = "output_fishing.mp4"
#min_score = 0.8
#id_counter = 0
#id_dict = {}

#segment_video = instance_segmentation()
#segment_video.load_model("mask_rcnn_coco.h5")

## Открытие видеофайла
#cap = cv2.VideoCapture(video_path)
#fps = cap.get(cv2.CAP_PROP_FPS)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

## Создание объекта VideoWriter для записи результата
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#while True:
#    # Считывание кадра видео
#    ret, frame = cap.read()
#    if not ret:
#        break

#    # Применение модели для получения списка объектов
#    result = segment_video.segmentFrame(frame, show_bboxes=False)

#    # Присвоение идентификаторов объектам
#    for obj in result["rois"][result["scores"] > min_score]:
#        obj_coords = tuple(map(int, obj))
#        if obj_coords in id_dict:
#            obj_id = id_dict[obj_coords]\
#        else:
#            id_counter += 1
#            obj_id = id_counter
#            id_dict[obj_coords] = obj_id
#        cv2.putText(frame, f"ID: {obj_id}", (obj_coords[1], obj_coords[0]), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#    # Запись кадра с присвоенными идентификаторами
#    out.write(frame)

#cap.release()
#out.release()



#/////////////////////////////////
#сохранение фрагмента видео
#/////////////////////////////////
# открываем исходное видеофайл
#cap = cv2.VideoCapture('fishing.mp4')

## устанавливаем позицию в начальный кадр
#cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

## создаем объект VideoWriter для сохранения фрагмента видео
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output_fishing.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

## проходим по кадрам и сохраняем их до конечного кадра
#for i in range(100, 201):
#    # считываем кадр
#    ret, frame = cap.read()

#    # проверяем, что кадр считался успешно
#    if not ret:
#        break

#    # записываем кадр в выходной файл
#    out.write(frame)

## освобождаем ресурсы
#cap.release()
#out.release()

#/////////////////////////////////////////
#указание минимального размера в пикселях
#/////////////////////////////////////////
#image_path = "path/to/image.jpg"
#min_size = 100
#image = cv2.imread(image_path)

#segment_image = instance_segmentation()
#segment_image.load_model("mask_rcnn_coco.h5")

## Применение модели для получения списка объектов
#result = segment_image.segment(image)

## Фильтрация объектов по минимальному размеру
#filtered_result = result["rois"][result["scores"] > 0.9]
#filtered_result = [obj for obj in filtered_result if (obj[2] - obj[0]) >= min_size 
#                   and (obj[3] - obj[1]) >= min_size]

## Отображение изображения с выделенными объектами
#for obj in filtered_result:
#    cv2.rectangle(image, (obj[1], obj[0]), (obj[3], obj[2]), (0, 255, 0), 2)

#cv2.imshow("Result", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#/////////////////////////////////////////
#время и количество объектов на скриншоте
#/////////////////////////////////////////
def draw_object_count(image_to_process, time, objects_count=None):

    start = (10, 70)
    start2 = (10, 120)
    font_size = 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Time: " + str(time)
    
    # Text output with a stroke
    # (so that it can be seen in different lighting conditions of the picture)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)
    if objects_count is not None:
        text2 = "ID: " + str(objects_count)
        final_image = cv2.putText(final_image, text2, start2, font, font_size,
                                  black_outline_color, width * 3, cv2.LINE_AA)
        final_image = cv2.putText(final_image, text2, start2, font, font_size,
                                  white_color, width, cv2.LINE_AA)


    return final_image

#/////////////////////////////////////////
#функция для сигментации фреймов
#/////////////////////////////////////////
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
def display_box_instances(image, boxes, masks, class_ids, class_name, scores, show_bboxes,text_size,
box_thickness, text_thickness):
    
    n_instances = boxes.shape[0]
    colors = random_colors(0)
    if boxes.shape[0]==0:
        return image 
    txt_color = (255,255,255)
    
    for i, color in enumerate(colors):
        mask = masks[:,:,i]
        
        image = apply_mask(image, mask, color)
        if not np.any(boxes[i]):
            continue

        if show_bboxes == True:    
            x1, y1, x2, y2 = boxes[i]
            label = class_name[class_ids[i]]
        
            score = scores[i] if scores is not None else None
        
            caption = '{} {:.2f}'.format(label, score) if score else label
        
        
            color_rec = [int(c) for c in np.array(colors[i]) * 255]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color_rec, box_thickness)
            image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_size,  txt_color, text_thickness)
        
    return image  


#/////////////////////////////////////////
#сигментация фреймов
#/////////////////////////////////////////
def segmentFrame(self, frame, path_for_pic, counter, segment_target_classes = None, show_bboxes = False, extract_segmented_objects = False,  extract_from_box = False,
    save_extracted_objects = False,mask_points_values = False, output_image_name = None,  text_thickness = 1,text_size = 0.6, box_thickness = 2, verbose = None):
        path_for_pic = path_for_pic + f"/extract_img"
        if not os.path.exists(path_for_pic):
            os.mkdir(path_for_pic)

        if extract_segmented_objects == True:
            original_frame = frame.copy()
            if extract_from_box == False:
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
            
        outputs = self.predictor.segment(frame)

        masks = outputs["instances"].pred_masks
        scores = outputs["instances"].scores
        class_ids = outputs["instances"].pred_classes

        boxes = outputs["instances"].pred_boxes.tensor
        boxes = torch.as_tensor(boxes, dtype = torch.int64)
        boxes = boxes.cpu().numpy()

        if torch.cuda.is_available() == False:
            class_ids = class_ids.numpy()
            masks = masks.numpy() 
            scores = scores.numpy()

        else:
            class_ids = class_ids.cpu().numpy()
            masks = masks.cpu().numpy() 
            scores = scores.cpu().numpy()

        names = []
        for _, a in enumerate(class_ids):
            name = self.class_names[a]
            names.append(name)
      
        scores = scores * 100
        scores = torch.as_tensor(scores, dtype = torch.int64)

        object_counts = Counter(names)
        
        r = {"boxes":boxes, "class_ids":class_ids, "class_names":names, "object_counts":object_counts,
         "scores":scores,"masks":masks, "extracted_objects":[]}

        # Run detection
        if verbose is not None:
            print("Processing image...")
        

        """Filter unused detections and detect specific classes """
        
        if segment_target_classes is not None:
            r = self.filter_objects(r, segment_target_classes) 
            object_counts = Counter(r["class_names"])
            r["object_counts"] = object_counts
        
        
        else:
            if len(r["masks"]) != 0:
                r["masks"] = np.stack(masks, axis = 2)
               
        output = display_box_instances(frame, r['boxes'], r['masks'], r["class_ids"], self.class_names, r['scores'], show_bboxes = show_bboxes,
        text_size = text_size, box_thickness=box_thickness, text_thickness=text_thickness)   
       
        if output_image_name is not None:

            cv2.imwrite(output_image_name, output)

        if extract_segmented_objects == False:
                
            if mask_points_values == True:
                mask = r['masks']
                contain_val = []
                if len(mask) != 0:  
                    for a in range(mask.shape[2]):
                    
                        m = mask[:,:,a]
                        mask_values = Mask(m).polygons()
                        val = mask_values.points
                        contain_val.append(val)

                    contain_val = np.asarray(contain_val, dtype = object)
                r['masks'] = contain_val
             

            return r, output

        elif extract_segmented_objects == True:

            """ Code to extract and crop out each of the objects segmented in an image """
            if extract_from_box == True:
                N = len(boxes)
                extracts = []
                m = 0
                for i in range(N):
                    m += 1    
                    ori_frame = original_frame
                    
                    box = boxes[i]
                    xmin, ymin, xmax, ymax = box
                    extracted_objects = ori_frame[int(ymin):int(ymax), int(xmin):int(xmax)] 
                           
                    if save_extracted_objects == True:
                        save_path = os.path.join(path_for_pic + f"/El({m})_frame({counter}).jpg")
                        
                        cv2.imwrite(save_path, extracted_objects)   

                    extracts.append(extracted_objects)
                
                if mask_points_values == True:
                    mask = r["masks"]    
                    contain_val = []
                    if len(mask) != 0:
                        for a in range(mask.shape[2]):
                            m = mask[:,:,a]
                            mask_values = Mask(m).polygons()
                            val = mask_values.points
                
                            contain_val.append(val)

                        contain_val = np.asarray(contain_val, dtype = object)    
                    r['masks'] = contain_val
   
                r["extracted_objects"] = extracts
                return r,  output, m
            
            else:

                mask = r['masks']
                m = 0
                ex = []
                if len(mask != 0):
                   
                    for a in range(mask.shape[2]):
                        ori_frame = original_frame
                        img = cv2.cvtColor(ori_frame, cv2.COLOR_RGB2BGR)
                    
                        for b in range(img.shape[2]):
       
                            img[:,:,b] = img[:,:,b] * mask[:,:,a]
                        m+=1
                        extracted_objects = img[np.ix_(mask[:,:,a].any(1), mask[:,:,a].any(0))]
                        ex.append(extracted_objects)
                        if save_extracted_objects == True:

                            save_path = os.path.join(path_for_pic + f"/El({m})_frame({counter}).jpg")
                            cv2.imwrite(save_path, extracted_objects)
                        extracted_objects = ex
                    

                    if mask_points_values == True:
                        mask = r["masks"]
                        contain_val = []
                        if len(mask) != 0:
                            for a in range(mask.shape[2]):
                                m = mask[:,:,a]
                                mask_values = Mask(m).polygons()
                                val = mask_values.points
                
                                contain_val.append(val)

                            contain_val = np.asarray(contain_val, dtype = object)    
                        r['masks'] = contain_val



                    """ The mask values of each of the extracted cropped object in the image
                    is added to the dictionary containing an array of output values:
                    """ 

                    r["extracted_objects"] = extracted_objects
                
                return r, output, m



#/////////////////////////////////////////
#распознавание объектов
#/////////////////////////////////////////
def object_detection_by_frame():
    
    filepath = filedialog.askopenfilename(title="Выбор файла")
    capture = cv2.VideoCapture(filepath)
    ## Получение первого кадра для инициализации
    ret, frame = capture.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    
    segment_frame = instanceSegmentation()
    segment_frame.load_model("pointrend_resnet50.pkl", confidence = 0.85, detection_speed = "rapid")
    

    object = object_var.get()
   
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    frames_per_second= 30
    
    class_name = {}



    path = enter.get()    
    #output_video_name=filedialog.asksaveasfilename(filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    path_for_pic = filedialog.askdirectory() + f"/{path}"
    output_video_name = path_for_pic + "/out_file.mp4"
    #print(path_for_pic)
    if not os.path.exists(path_for_pic):
        os.mkdir(path_for_pic)
    if frames_per_second is not None:
        save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
    counter = 0
    count_obj = 0
    count_pic = 0
    frames_obj =[]
    id_arr=[]
    ig_obj_help =[]
    rec_arr = [[], []]

    # Инициализируем переменные для хранения предыдущего кадра и прямоугольников объектов
    previous_frame = None
    previous_boxes = {}


    
    if object=="люди":
        target_classes = segment_frame.select_target_classes(person=True)
    elif object=="транспорт":
        target_classes = segment_frame.select_target_classes(car = True, airplane = True, bus = True, train = True, truck = True)
    elif object=="велосипеды и мотоциклы":
        target_classes = segment_frame.select_target_classes(bicycle = True, motorcycle = True)
    elif object=="животные":
         target_classes = segment_frame.select_target_classes(bird = True, cat = True, dog = True, horse = True, 
               sheep = True, cow = True, elephant = True, bear = True, zebra = True, giraffe = True)
    elif object=="спортинвертарь":
        target_classes = segment_frame.select_target_classes(frisbee = True, skis = True, snowboard = True, sports_ball = True, 
                                                             skateboard = True, surfboard = True)
    else:
        target_classes = segment_frame.select_target_classes(person=True, bicycle = True, car = True, motorcycle = True, airplane = True,
                bus = True, train = True, truck = True, boat = True, bird = True, cat = True, dog = True, horse = True, 
                sheep = True, cow = True, elephant = True, bear = True, zebra = True, giraffe = True, frisbee = True, 
                skis = True, snowboard = True, sports_ball = True, skateboard = True, surfboard = True)
    start = time.time()           
    while True:   
        ret, frame = capture.read()
        if ret: 
            
            obj_id = 0
            
            fps = capture.get(cv2.CAP_PROP_FPS)
            #difference = cv2.absdiff(frame1, frame)
            #gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
            #blur = cv2.GaussianBlur(gray, (5,5), 0)
            #_, threshold = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
            #dilate = cv2.dilate(threshold, None, iterations=15)
            #contour, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #max=0
            #sel_countour=None
            #for countour in contour:
            #    if countour.shape[0]>max and countour.shape[0] > 1600:
            #        sel_countour=countour
            #        max=countour.shape[0]

                #print(countour.shape[0])
            

            counter +=1
            percent_count=0
            frame_duration = counter / fps
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #gray = cv2.GaussianBlur(gray, (5, 5), 0)
            ## Вычисление разности между текущим и предыдущим кадром
            #frame_diff = cv2.absdiff(prev_gray, gray)
            ## Применение порогового значения для выделения движущихся объектов
            #_, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
            #dilate = cv2.dilate(thresh, None, iterations=15)
            ## Поиск контуров на пороговом изображении
            #contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
            ## Отрисовка контуров на исходном кадре
            #maxim=0
            #sel_countour=None
            #for countour in contours:
            #    if countour.shape[0]>maxim and countour.shape[0] > 1600:
            #        sel_countour=countour
            #        maxim=countour.shape[0]
            seg, output, m = segmentFrame(segment_frame, frame, path_for_pic, counter, show_bboxes=False, segment_target_classes= target_classes, extract_segmented_objects = True, extract_from_box = False, save_extracted_objects = True)
            


            boxes = seg['boxes']            
            print("No. of frames:", counter) 
            
            id_obj =[]
            for j in range(1, len(seg['boxes'])+1): #заполнение массива с id
                id_obj.append(j)
           
            if counter == 1:
                for i in range(len(seg['boxes'])): #присваивание идентификаторов
                    #obj_id += 1                
                    cv2.putText(frame, f"ID: {id_obj[i]}", (seg['boxes'][i][0], seg['boxes'][i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    os.rename(path_for_pic + f"/extract_img/El({id_obj[i]})_frame({counter}).jpg", path_for_pic + f"/extract_img/El({id_obj[i]})_frame({counter})_id({id_obj[i]}).jpg")                
            else:                
                if len(ig_obj_help)>len(id_obj):                                      
                    for l in id_arr:
                        for j in id_obj:
                            if os.path.exists(path_for_pic + f"/extract_img/El({l})_frame({counter-1})_id({l}).jpg")==False or os.path.exists(path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg")==False:
                                continue
                            frame_befor = path_for_pic + f"/extract_img/El({l})_frame({counter-1})_id({l}).jpg"
                            frame_now = path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg"

                            percent = compare_images(frame_befor, frame_now) 
                            if percent > 45:
                                #os.rename(frame_now_new, path_for_pic + f"/extract_img/El({id_obj[i-1]})_frame({counter})_id({id_obj[i-1]}).jpg")
                                percent_count+=1
                                os.rename(frame_now, path_for_pic + f"/extract_img/El({l if l!=j else j})_frame({counter})_id({l}).jpg")                             
                                im = Image.open(path_for_pic + f"/extract_img/El({l if l!=j else j})_frame({counter})_id({l}).jpg") 
                                #for box in seg['boxes']:
                                    #if im.size[0]==(box[2]-box[0]) and im.size[1]==(box[3]-box[1]):                                    
                                cv2.putText(frame, f"ID: {l}", (seg['boxes'][j-1][2]-im.size[0], seg['boxes'][j-1][3]-im.size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                                #x1, y1, x2, y2 = boxes[j-1]
                                #prev_x1, prev_y1, prev_x2, prev_y2 = previous_boxes[l]
                                #diff = abs(x1 - prev_x1) + abs(y1 - prev_y1) + abs(x2 - prev_x2) + abs(y2 - prev_y2)
                                #if diff > 35:
                                #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                                ##print('Коробка: ', boxes)
                                ##print('Предыдущая коробка: ', previous_boxes)
                                #previous_boxes[l] = (x1, y1, x2, y2)

                    ig_obj_help = id_obj
                elif len(ig_obj_help) < len(id_obj):
                    none_obj=0
                    
                    count_none_obj = id_obj
                    for k in id_obj:
                        percent=0
                        for l in id_obj:
                            mean_per=[]
                            if os.path.exists(path_for_pic + f"/extract_img/El({k})_frame({counter}).jpg"):
                                img = Image.open(path_for_pic + f"/extract_img/El({k})_frame({counter}).jpg")
                            
                            if img.size[0] < 190:
                                none_obj+=1
                                if k in count_none_obj:
                                    count_none_obj.pop(k-1) 
                            if os.path.exists(path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg") == False:
                                continue
                            if Image.open(path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg").size[0] < 190:
                                continue

                            for m in range(1, counter, 5):
                                frame_befor = path_for_pic + f"/extract_img/El({k})_frame({m})_id({k}).jpg"
                                frame_now = path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg"
                                if os.path.exists(path_for_pic + f"/extract_img/El({k})_frame({m})_id({k}).jpg")==False:
                                    continue
                                percent = compare_images(frame_befor, frame_now)
                                #mean_per.append(percent)
                                if percent > 43:
                                    break
                                
                                    #ig_obj_help = count_none_obj
                                
                                
                            #print(f"Среднее id({l}): ", np.mean(mean_per))
                            if percent > 43:
                                img.close()
                                os.rename(path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg", path_for_pic + f"/extract_img/El({k if k!=i else i})_frame({counter})_id({k}).jpg")
                                im = Image.open(path_for_pic + f"/extract_img/El({k if k!=i else i})_frame({counter})_id({k}).jpg")
                                cv2.putText(frame, f"ID: {k}", (seg['boxes'][l-1][2]-im.size[0], seg['boxes'][l-1][3]-im.size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                                #x1, y1, x2, y2 = boxes[l-1]
                                #prev_x1, prev_y1, prev_x2, prev_y2 = previous_boxes[k]
                                #diff = abs(x1 - prev_x1) + abs(y1 - prev_y1) + abs(x2 - prev_x2) + abs(y2 - prev_y2)
                                #if diff > 35:
                                #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                                ##print('Коробка: ', boxes)
                                ##print('Предыдущая коробка: ', previous_boxes)
                                #previous_boxes[k] = (x1, y1, x2, y2)
                                
                                if none_obj > 0:
                                    ig_obj_help = count_none_obj
                                else:
                                    ig_obj_help=id_obj                                
                                break
                    #if ig_obj_help!=id_obj:                        
                        #for i in id_obj:
                    #elif np.mean(mean_per) > 0 and np.mean(mean_per) <= 40:
                        #if np.mean(mean_per) < 35 or ig_obj_help!=id_obj or os.path.exists(path_for_pic + f"/extract_img/El({k})_frame({counter-1})_id({k}).jpg")==False:
                    for i in id_obj:
                        if os.path.exists(path_for_pic + f"/extract_img/El({i})_frame({counter})_id({i}).jpg"):
                            continue
                        else:
                            if none_obj > 0:
                                ig_obj_help = count_none_obj
                                continue
                            #prev_img_size[len(id_arr)] = img.size[0]
                            img.close()
                            os.rename(path_for_pic + f"/extract_img/El({i})_frame({counter}).jpg", path_for_pic + f"/extract_img/El({len(id_arr)+1})_frame({counter})_id({len(id_arr)+1}).jpg")
                            im = Image.open(path_for_pic + f"/extract_img/El({len(id_arr)+1})_frame({counter})_id({len(id_arr)+1}).jpg")
                            cv2.putText(frame, f"ID: {len(id_arr)+1}", (seg['boxes'][i-1][2]-im.size[0], seg['boxes'][i-1][3]-im.size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            id_arr.append(len(id_arr)+1) 
                            #if none_obj > 0:
                                #ig_obj_help = count_none_obj
                            #else:
                            ig_obj_help=id_obj                            
                            
                                
                            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
                            cv2.imwrite(path_for_pic + f"/Detect_new_obj_{count_pic}.jpg", draw_object_count(frame, frame_duration_formatted, len(id_arr)))
                            count_pic+=1
                            x1, y1, x2, y2 = boxes[i-1]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                            previous_boxes[len(id_arr)] = (x1, y1, x2, y2)
                else:
                    
                    for i in range(1, (len(seg['boxes'])+1)):                    
                        #obj_id += 1
                        percent_count_arr = []

                        for j in range(1, (len(seg['boxes'])+1)):     
                            if percent_count == len(seg['boxes']): #ограничение на повторные проверки
                                break
                            frame_befor = path_for_pic + f"/extract_img/El({i})_frame({counter-1})_id({id_obj[i-1]}).jpg"
                            frame_now = path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg"
                            #if os.path.exists(path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg")==False:
                            #        continue

                            if os.path.exists(path_for_pic + f"/extract_img/El({i})_frame({counter-1})_id({id_obj[i-1]}).jpg")==False:
                                for k in id_arr:
                                    if os.path.exists(path_for_pic + f"/extract_img/El({k})_frame({counter-1})_id({k}).jpg"):
                                        frame_befor = path_for_pic + f"/extract_img/El({k})_frame({counter-1})_id({k}).jpg"
                                        percent = compare_objects(frame_befor, frame_now)

                                        if percent > 25:
                                            #os.rename(frame_now_new, path_for_pic + f"/extract_img/El({id_obj[i-1]})_frame({counter})_id({id_obj[i-1]}).jpg")
                                            percent_count+=1
                                            os.rename(frame_now, path_for_pic + f"/extract_img/El({k if k!=j else j})_frame({counter})_id({k}).jpg")                             
                                            im = Image.open(path_for_pic + f"/extract_img/El({k if k!=j else j})_frame({counter})_id({k}).jpg") 
                                            #for box in seg['boxes']:
                                                #if im.size[0]==(box[2]-box[0]) and im.size[1]==(box[3]-box[1]):                                    
                                            cv2.putText(frame, f"ID: {k}", (seg['boxes'][j-1][2]-im.size[0], seg['boxes'][j-1][3]-im.size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
                                            x1, y1, x2, y2 = boxes[j-1]
                                            prev_x1, prev_y1, prev_x2, prev_y2 = previous_boxes[k]
                                            diff = abs(x1 - prev_x1) + abs(y1 - prev_y1) + abs(x2 - prev_x2) + abs(y2 - prev_y2)
                                            if diff > 25:
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                                            previous_boxes[k] = (x1, y1, x2, y2)
                                break                            
                            if os.path.exists(frame_now)==False:
                                percent_count_arr.append(0)
                                continue
                            #elif os.path.exists(path_for_pic + f"/extract_img/El({j})_frame({counter})_id({id_obj[j-1]}).jpg" )==False:
                            #    continue
                            #else:
                            #    percent = compare_objects(frame_befor, frame_now)
                            percent = compare_objects(frame_befor, frame_now)
                            percent_count_arr.append(percent)
                        if len(percent_count_arr)!=0:
                            percent_max = max(percent_count_arr)
                            percent_index = percent_count_arr.index(percent_max)
                            #percent_count+=1
                            os.rename(path_for_pic + f"/extract_img/El({(percent_index+1)})_frame({counter}).jpg", path_for_pic + f"/extract_img/El({id_obj[i-1] if id_obj[i-1]!=(percent_index+1) else (percent_index+1)})_frame({counter})_id({id_obj[i-1]}).jpg")                             
                            im = Image.open(path_for_pic + f"/extract_img/El({id_obj[i-1] if id_obj[i-1]!=(percent_index+1) else (percent_index+1)})_frame({counter})_id({id_obj[i-1]}).jpg") 
                            #for box in seg['boxes']:
                                #if im.size[0]==(box[2]-box[0]) and im.size[1]==(box[3]-box[1]):                                    
                            #for n in id_obj:
                            #    if os.path.exists(path_for_pic + f"/extract_img/El({n})_frame({counter})_id({n}).jpg"):
                            cv2.putText(frame, f"ID: {i}", (seg['boxes'][percent_index][2]-im.size[0], seg['boxes'][percent_index][3]-im.size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            class_name[percent_index+1] = seg['class_names'][percent_index]
                            
                            x1, y1, x2, y2 = boxes[percent_index]
                            prev_x1, prev_y1, prev_x2, prev_y2 = previous_boxes[i]
                            diff = abs(x1 - prev_x1) + abs(y1 - prev_y1) + abs(x2 - prev_x2) + abs(y2 - prev_y2)
                            if diff > 25:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                            #print('Коробка: ', boxes)
                            #print('Предыдущая коробка: ', previous_boxes)
                            previous_boxes[i] = (x1, y1, x2, y2)
            
            
            #cv2.drawContours(frame, sel_countour, -1, (0, 0, 255), 2)           
            # Обновление предыдущего кадра
            #prev_gray = gray

            #cv2.drawContours(frame1, sel_countour, -1, (0, 0, 255), 2)            
            #frame1 = frame
            print(previous_boxes)




            ## Преобразуем кадр в оттенки серого
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            ## Применяем разницу кадров между текущим и предыдущим кадром
            #if previous_frame is None:
            #    previous_frame = gray
            #    continue

            #frame_diff = cv2.absdiff(previous_frame, gray)

            ## Применяем пороговую обработку для выделения движущихся областей
            #_, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

            ## Производим морфологические операции для устранения шума
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            ## Находим контуры объектов
            #contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ## Очищаем предыдущие прямоугольники
            #previous_boxes.clear()

            ## Обходим все найденные контуры
            #for contour in contours:
            #    # Вычисляем площадь контура
            #    area = cv2.contourArea(contour)

            #    # Игнорируем слишком маленькие контуры
            #    if area < 500:
            #        continue

            #    # Вычисляем координаты прямоугольника, охватывающего контур
            #    x, y, w, h = cv2.boundingRect(contour)

            #    # Обводим объект прямоугольником
            #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            #    # Сохраняем координаты прямоугольника для следующего кадра
            #    previous_boxes[(x, y, w, h)] = True

            ## Обновляем предыдущий кадр
            #previous_frame = gray            
                        
            ##добавление прямоугольника для отслеживания движения 
            #boxes = seg['boxes']
            #class_ids = seg['class_ids']

            ## Обходим все найденные объекты на текущем кадре
            #for i, box in enumerate(boxes):
            #    class_id = class_ids[i]

            #    # Получаем координаты прямоугольника
            #    x1, y1, x2, y2 = box

            #    # Проверяем, был ли объект на предыдущем кадре
            #    if class_id in previous_boxes:
            #        # Получаем предыдущие координаты прямоугольника
            #        prev_x1, prev_y1, prev_x2, prev_y2 = previous_boxes[class_id]

            #        # Вычисляем разницу между текущими и предыдущими координатами
            #        diff = abs(x1 - prev_x1) + abs(y1 - prev_y1) + abs(x2 - prev_x2) + abs(y2 - prev_y2)

            #        # Если объект движется, обводим его прямоугольником
            #        if diff > 30:
            #            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            #    else:
            #        # Если объект появился на текущем кадре, обводим его прямоугольником
            #        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

            #    # Сохраняем текущие координаты прямоугольника для следующего кадра
            #    previous_boxes[class_id] = (x1, y1, x2, y2)
            #print(previous_boxes)
            ##добавление прямоугольника для отслеживания движения                  
            #if id_obj is not None:
            #    for i in range(len(id_arr)):
            #        rec_arr[i].append(seg['boxes'][i].tolist())
    
            #    for boxes in rec_arr:
            #        #print(boxes)
            #        if len(boxes)==2:
            #            #r1 = r1 + (boxes[couter][0]-boxes[counter-1][0])
            #            #r2 = r2 + (boxes[couter][1]-boxes[counter-1][1])
            #            #r3 = r3 + (boxes[couter][2]-boxes[counter-1][2])
            #            #r4 = r4 + (boxes[couter][3]-boxes[counter-1][3])
            #            if (boxes[1][0]-boxes[0][0])>5 or (boxes[1][1]-boxes[0][1])>5 or (boxes[1][2]-boxes[0][2])>5 or (boxes[1][3]-boxes[0][3])>5:
            #                cv2.rectangle(frame, (boxes[1][0], boxes[1][1]), (boxes[1][2],boxes[1][3]), (255, 0, 255))
            #            boxes = []

            #        #if counter%5==0 and r1 > 20 or r2 > 20 or r3 > 20 or r4 > 20:
            #        #    cv2.rectangle(frame, (boxes[couter][0], boxes[couter][1]), (boxes[couter][2],boxes[couter][3]), (255, 0, 255))
            #        #    r1 = 0
            #        #    r2 = 0
            #        #    r3 = 0
            #        #    r4 = 0
            #            #for box in rec_arr[boxes]:
            #            #    if box[]
                

                #for i in range(len(id_arr)):
                #    frames_obj[i]=counter
                #print("Последний фрейм: ", frames_obj)
            #cv2.imshow("Frame", frame)

            #output = cv2.resize(output, (width,height), interpolation=cv2.INTER_AREA)
            #if len(seg["scores"])>count_obj:
            #    #if not os.path.exists(path_for_pic + f"/Detect{len(seg['scores'])}obj.jpg"):
            #    frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            #    cv2.imwrite(path_for_pic + f"/Detect_new_obj_{count_pic}.jpg", draw_object_count(frame, frame_duration_formatted, len(seg["scores"])))
            #    print(len(seg['scores']))
            #    count_obj = len(seg["scores"])
            #    count_pic+=1
            #elif len(seg["scores"])<count_obj:
            #    #cv2.imwrite(path_for_pic + f"/Obj_leave_{count_pic}.jpg", draw_object_count(frame, len(seg["scores"]), frame_duration_formatted))
            #    count_obj = len(seg["scores"])
            #    #count_pic+=1
            


            

            if output_video_name is not None:
                save_video.write(output)                  
        else:
            break    
        #lb.set(f"No. of frames: {counter}")
    end = time.time() 
    #???????????????????????????????????????????????????????????????????????????????????????
    face_cascade= cv2.CascadeClassifier('C:/Users/AndreyBayborodin/source/repos/test_rasp_pr/test_rasp_pr/env1/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    for i in id_arr:
        prev_img_size = 0
        for j in range(1, counter+1):
            if os.path.exists(path_for_pic + f"/extract_img/El({i})_frame({j})_id({i}).jpg")==False:
                continue
            img = Image.open(path_for_pic + f"/extract_img/El({i})_frame({j})_id({i}).jpg") 
            image = cv2.imread(path_for_pic + f"/extract_img/El({i})_frame({j})_id({i}).jpg")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.5, 8)
            if class_name[i]=='person':
                if img.size[0]> prev_img_size and len(faces)>0:                
                    destination_path = os.path.join(path_for_pic, f"Best_photo_{i}.jpg")
                    cv2.imwrite(destination_path, image)
                    #cv2.imwrite(path_for_pic + f"/Best_photo_{i}.jpg", path_for_pic + f"/extract_img/El({i})_frame({counter}).jpg")
                    prev_img_size=img.size[0]
                    img.close()
            else:
                if img.size[0]> prev_img_size:
                    destination_path = os.path.join(path_for_pic, f"Best_photo_{i}.jpg")
                    cv2.imwrite(destination_path, image)
                    #cv2.imwrite(path_for_pic + f"/Best_photo_{i}.jpg", path_for_pic + f"/extract_img/El({i})_frame({counter}).jpg")
                    prev_img_size=img.size[0]
                    img.close()
    #???????????????????????????????????????????????????????????????????????????????????????


    lb.set(f"Обработано {counter} фреймов за {end-start:.1f} секунд")
    print(f"Processed {counter} frames in {end-start:.1f} seconds")             
    capture.release()
    if output_video_name is not None:
        save_video.release()  
    #lb.set(f"Выявление объектов выполнено! \nВидео и скриншоты находятся в данной папке {path_for_pic}")
    
    
    label_com = ttk.Label(text="Выберите ID объекта", background = "white", font=("Arial", 10))
    label_com.place(x=170, y=220, height=30, width = 300)
    #object = ["люди", "транспорт", "велосипеды и мотоциклы", "животные", "спортинвертарь"]
    
    combobox_id = ttk.Combobox(textvariable=id_count, values=id_arr)
    combobox_id.place(x=170, y=250, height=30, width = 200)

    btn_id = ttk.Button(text="Информация", command=partial(out_info, class_name, path_for_pic)) # создаем кнопку из пакета ttk
    btn_id.place(x=400, y=250, height=30, width = 100)    # размещаем кнопку в окне

    
    label_out = ttk.Label(textvariable=lbl_out, background = "white", font=("Arial", 10))
    label_out.place(x=520, y=250, height=30, width = 250)
    
    #enter = ttk.Entry()
    #enter.place(x=400, y=150, height=30, width = 200)
    
    #return seg, output

def out_info(class_name, path_for_pic):
        ident = id_count.get()
        #for i in range(1, (int(ident)+1)):
        #    if i == int(ident):
        lbl_out.set(f'Пердварительный класс объекта: {class_name[int(ident)]}')
        os.startfile(path_for_pic+ f'/Best_photo_{ident}.jpg')
    
#def live_object_detection():
#    capture = cv2.VideoCapture(0)

#    segment_video = instance_segmentation()
#    segment_video.load_model("mask_rcnn_coco.h5")
#    segment_video.process_camera(capture, frames_per_second= 15, output_video_name="output_video_live.mp4", show_frames= True, frame_name= "frame")

#/////////////////////////////////////////
#время скриншота
#/////////////////////////////////////////
def format_timedelta(td):
    """Служебная функция для классного форматирования объектов timedelta (например, 00:00:20.05)
    исключая микросекунды и сохраняя миллисекунды"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")
    
#/////////////////////////////////////////
#скриншоты объектов
#/////////////////////////////////////////
def take_screenshot_from_video():    
    #filepath = opened_file.get()
    filepath = filedialog.askopenfilename(title="Выбор файла")    
    cap = cv2.VideoCapture(filepath)
    count = 0
    speed = speed_var.get()
    path = enter_2.get()
    path_for_pic = filedialog.askdirectory() + f"/{path}"
    if not os.path.exists(path_for_pic):
        os.mkdir(path_for_pic)

    #saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    # получить список длительностей для сохранения
    #saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    
    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 3
        # print(fps)

        if ret:
            frame_id = int(round(cap.get(1)))
            # print(frame_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(speed)

            #if frame_id % multiplier == 0:
            #    cv2.imwrite(f"dataset_from_video/{count}.jpg", frame)
            #    print(f"Take a screenshot {count}")
            #    count += 1
            frame_duration = count / fps
            if k == ord(" "):
                #cv2.imwrite(f"dataset_from_video/{count}_extra_scr.jpg", frame)
                #print(f"Take an extra screenshot {count}")
                frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
                cv2.imwrite(path_for_pic + f"/frame{frame_duration_formatted}.jpg", frame)#draw_object_count(frame, frame_duration_formatted))
                
            elif k == ord("q"):
                #print("Q pressed, closing the app")
                break
            count += 1
        else:
            #print("[Error] Can't get the frame...")
            break

        

    lbl.set(f"Выявление объектов выполнено! \nВидео и скриншоты находятся в данной папке {path_for_pic}")
    cap.release()
    cv2.destroyAllWindows()



#/////////////////////////////////////////
#TKINTER
#/////////////////////////////////////////
def open_info(): 
    showinfo(title="Информация", message="В данной папке будет сохранены файлы. После нажатия на кнопку необходимо выбрать файл для распознавания и каталог, в котором будет создана новая папка. \nВыполнение программы занимает некоторое время, дождитесь сообщения об успешном выполнении.")

root = Tk()     # создаем корневой объект - окно

root.title("Приложение на Tkinter")     # устанавливаем заголовок окна
root.geometry("1000x500")    # устанавливаем размеры окна
root.resizable(False, False) 
root["bg"] = "white"

#logo = PhotoImage(file="logo.png")
image=Image.open('logo.png')
logo=image.resize((150, 80))
my_img=ImageTk.PhotoImage(logo)
label_logo = ttk.Label(image=my_img, text = "Выявление подвижных объектов", compound="left", background = "white", font=("Arial", 25))
label_logo.place(height= 60, width=1000)

label_object = ttk.Label(text="Выберите объект при необходимости", background = "white", font=("Arial", 10))
label_object.place(x=170, y=120, height=30, width = 300)
object = ["люди", "транспорт", "велосипеды и мотоциклы", "животные", "спортинвертарь"]
object_var = StringVar()
combobox_object = ttk.Combobox(textvariable=object_var, values=object)
combobox_object.place(x=170, y=150, height=30, width = 200)
label_folder = ttk.Label(text = "Введите название папки", background = "white", font=("Arial", 10))
label_folder.place(x=400, y=120, height=30, width = 150)
def open_info(): 
    showinfo(title="Информация", message="В данной папке будет сохранены файлы. После нажатия на кнопку необходимо выбрать файл для распознавания и каталог, в котором будет создана новая папка. \nВыполнение программы занимает некоторое время, дождитесь сообщения об успешном выполнении.")
image=Image.open('info.jpg')
info=image.resize((15, 15))
info_img=ImageTk.PhotoImage(info)
btn_info = ttk.Button(root, image=info_img, command = open_info)
btn_info.place(x=560, y=120, height=22, width = 22) 
enter = ttk.Entry()
enter.place(x=400, y=150, height=30, width = 200)
btn_detection = ttk.Button(text="Распознать", command=object_detection_by_frame) # создаем кнопку из пакета ttk
btn_detection.place(x=630, y=150, height=30, width = 100)    # размещаем кнопку в окне
lb = StringVar()
label_out = ttk.Label(textvariable=lb, background = "white", font=("Arial", 10))
label_out.place(x=170, y=180, height=30, width = 300)
id_count = StringVar()
lbl_out = StringVar()

#label_speed = ttk.Label(text = "Выберите скорость воспроизведения видео", background = "white", font=("Arial", 10))
#label_speed.place(x=600, y=120, height=30, width = 270)
#speed = [1, 20, 40]
#speed_var = IntVar(value=speed[1])
#combobox = ttk.Combobox(textvariable=speed_var, values=speed)
#combobox.place(x=600, y=150, height=30, width = 200)
##command=partial(take_screenshot_from_video, selected)
#label_folder_2 = ttk.Label(text = "Введите название папки", background = "white", font=("Arial", 10))
#label_folder_2.place(x=600, y=190, height=30, width = 150)
#def open_info_2(): 
#    showinfo(title="Информация", message="В данной папке будет сохранены файлы. После нажатия на кнопку необходимо выбрать файл для обработки и каталог, в котором будет создана новая папка. \nПри выполнении программы на экране будет воспроизвдено видео, при нажатии на Space можно сделать скриншот, при нажатии на Q закончить выполнение программы.")
#image=Image.open('info.jpg')
#info_2=image.resize((15, 15))
#info_img_2=ImageTk.PhotoImage(info_2)
#btn_info_2 = ttk.Button(root, image=info_img_2, command = open_info_2)
#btn_info_2.place(x=760, y=190, height=22, width = 22) 

#enter_2 = ttk.Entry()
#enter_2.place(x=600, y=220, height=30, width = 200)
#btn_screen = ttk.Button(text="Screen", command=take_screenshot_from_video) # создаем кнопку из пакета ttk
#btn_screen.place(x=600, y=260, height=30, width = 100)    # размещаем кнопку в окне
#lbl = StringVar()
#label_out_2 = ttk.Label(textvariable=lbl, background = "white", font=("Arial", 10))
#label_out_2.place(x=600, y=300, height=70, width = 300)


root.mainloop()


#from tracker import *

## Create tracker object
#tracker = EuclideanDistTracker()

#cap = cv2.VideoCapture("fishing.mp4")

## Object detection from Stable camera
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

#while True:
#    ret, frame = cap.read()
#    #height, width, _ = frame.shape

#    # Extract Region of interest
#    roi = frame[50: 1000,700: 2000]

#    # 1. Object Detection
#    mask = object_detector.apply(roi)
#    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
#    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    detections = []
#    for cnt in contours:
#        # Calculate area and remove small elements
#        area = cv2.contourArea(cnt)
#        if area > 300:
#            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
#            x, y, w, h = cv2.boundingRect(cnt)


#            detections.append([x, y, w, h])

#    # 2. Object Tracking
#    boxes_ids = tracker.update(detections)
#    for box_id in boxes_ids:
#        x, y, w, h, id = box_id
#        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
#        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

#    cv2.imshow("roi", roi)
#    cv2.imshow("Frame", frame)
#    cv2.imshow("Mask", mask)

#    key = cv2.waitKey(30)
#    if key == 27:
#        break

#cap.release()
#cv2.destroyAllWindows()
