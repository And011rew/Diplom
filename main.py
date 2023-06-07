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


# /////////////////////////////////
# сравнение изображений
# /////////////////////////////////
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

    print("Процент совпадения:", percent_match)
    return percent_match



# /////////////////////////////////////////
# время и количество объектов на скриншоте
# /////////////////////////////////////////
def draw_object_count(image_to_process, time, objects_count=None):
    start = (10, 70)
    start2 = (10, 120)
    font_size = 1.2
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Time: " + str(time)


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


# /////////////////////////////////////////
# функция для сигментации фреймов
# /////////////////////////////////////////
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


def display_box_instances(image, boxes, masks, class_ids, class_name, scores, show_bboxes, text_size,
                          box_thickness, text_thickness):
    n_instances = boxes.shape[0]
    colors = random_colors(0)
    if boxes.shape[0] == 0:
        return image
    txt_color = (255, 255, 255)

    for i, color in enumerate(colors):
        mask = masks[:, :, i]

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
            image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, text_size, txt_color,
                                text_thickness)

    return image


# /////////////////////////////////////////
# сигментация фреймов
# /////////////////////////////////////////
def segmentFrame(self, frame, path_for_pic, counter, segment_target_classes=None, show_bboxes=False,
                 extract_segmented_objects=False, extract_from_box=False,
                 save_extracted_objects=False, mask_points_values=False, output_image_name=None, text_thickness=1,
                 text_size=0.6, box_thickness=2, verbose=None):
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
    boxes = torch.as_tensor(boxes, dtype=torch.int64)
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
    scores = torch.as_tensor(scores, dtype=torch.int64)

    object_counts = Counter(names)

    r = {"boxes": boxes, "class_ids": class_ids, "class_names": names, "object_counts": object_counts,
         "scores": scores, "masks": masks, "extracted_objects": []}

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
            r["masks"] = np.stack(masks, axis=2)

    output = display_box_instances(frame, r['boxes'], r['masks'], r["class_ids"], self.class_names, r['scores'],
                                   show_bboxes=show_bboxes,
                                   text_size=text_size, box_thickness=box_thickness, text_thickness=text_thickness)

    if output_image_name is not None:
        cv2.imwrite(output_image_name, output)

    if extract_segmented_objects == False:

        if mask_points_values == True:
            mask = r['masks']
            contain_val = []
            if len(mask) != 0:
                for a in range(mask.shape[2]):
                    m = mask[:, :, a]
                    # mask_values = Mask(m).polygons()
                    # val = mask_values.points
                    # contain_val.append(val)

                contain_val = np.asarray(contain_val, dtype=object)
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
                        m = mask[:, :, a]
                        # mask_values = Mask(m).polygons()
                        # val = mask_values.points
                        #
                        # contain_val.append(val)

                    contain_val = np.asarray(contain_val, dtype=object)
                r['masks'] = contain_val

            r["extracted_objects"] = extracts
            return r, output, m

        else:

            mask = r['masks']
            m = 0
            ex = []
            if len(mask != 0):

                for a in range(mask.shape[2]):
                    ori_frame = original_frame
                    img = cv2.cvtColor(ori_frame, cv2.COLOR_RGB2BGR)

                    for b in range(img.shape[2]):
                        img[:, :, b] = img[:, :, b] * mask[:, :, a]
                    m += 1
                    extracted_objects = img[np.ix_(mask[:, :, a].any(1), mask[:, :, a].any(0))]
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
                            m = mask[:, :, a]
                            # mask_values = Mask(m).polygons()
                            # val = mask_values.points
                            #
                            # contain_val.append(val)

                        contain_val = np.asarray(contain_val, dtype=object)
                    r['masks'] = contain_val

                """ The mask values of each of the extracted cropped object in the image
                is added to the dictionary containing an array of output values:
                """

                r["extracted_objects"] = extracted_objects

            return r, output, m


# /////////////////////////////////////////
# распознавание объектов
# /////////////////////////////////////////
def object_detection_by_frame():
    filepath = filedialog.askopenfilename(title="Выбор файла")
    capture = cv2.VideoCapture(filepath)
    ## Получение первого кадра для инициализации
    ret, frame = capture.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    segment_frame = instanceSegmentation()
    segment_frame.load_model("pointrend_resnet50.pkl", confidence=0.85, detection_speed="rapid")

    object = object_var.get()

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    frames_per_second = 30

    class_name = {}

    path = enter.get()
    # output_video_name=filedialog.asksaveasfilename(filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    path_for_pic = filedialog.askdirectory() + f"/{path}"
    output_video_name = path_for_pic + "/out_file.mp4"
    # print(path_for_pic)
    if not os.path.exists(path_for_pic):
        os.mkdir(path_for_pic)
    if frames_per_second is not None:
        save_video = cv2.VideoWriter(output_video_name, codec, frames_per_second, (width, height))
    counter = 0
    count_obj = 0
    count_pic = 0
    frames_obj = []
    id_arr = []
    ig_obj_help = []
    rec_arr = [[], []]

    # Инициализируем переменные для хранения предыдущего кадра и прямоугольников объектов
    previous_frame = None
    previous_boxes = {}

    if object == "люди":
        target_classes = segment_frame.select_target_classes(person=True)
    elif object == "транспорт":
        target_classes = segment_frame.select_target_classes(car=True, airplane=True, bus=True, train=True, truck=True)
    elif object == "велосипеды и мотоциклы":
        target_classes = segment_frame.select_target_classes(bicycle=True, motorcycle=True)
    elif object == "животные":
        target_classes = segment_frame.select_target_classes(bird=True, cat=True, dog=True, horse=True,
                                                             sheep=True, cow=True, elephant=True, bear=True, zebra=True,
                                                             giraffe=True)
    elif object == "спортинвертарь":
        target_classes = segment_frame.select_target_classes(frisbee=True, skis=True, snowboard=True, sports_ball=True,
                                                             skateboard=True, surfboard=True)
    else:
        target_classes = segment_frame.select_target_classes(person=True, bicycle=True, car=True, motorcycle=True,
                                                             airplane=True,
                                                             bus=True, train=True, truck=True, boat=True, bird=True,
                                                             cat=True, dog=True, horse=True,
                                                             sheep=True, cow=True, elephant=True, bear=True, zebra=True,
                                                             giraffe=True, frisbee=True,
                                                             skis=True, snowboard=True, sports_ball=True,
                                                             skateboard=True, surfboard=True)
    start = time.time()
    while True:
        ret, frame = capture.read()
        if ret:

            obj_id = 0

            fps = capture.get(cv2.CAP_PROP_FPS)
            counter += 1
            percent_count = 0
            frame_duration = counter / fps

            seg, output, m = segmentFrame(segment_frame, frame, path_for_pic, counter, show_bboxes=False,
                                          segment_target_classes=target_classes, extract_segmented_objects=True,
                                          extract_from_box=False, save_extracted_objects=True)

            boxes = seg['boxes']
            print("No. of frames:", counter)

            id_obj = []
            for j in range(1, len(seg['boxes']) + 1):  # заполнение массива с id
                id_obj.append(j)

            if counter == 1:
                for i in range(len(seg['boxes'])):  # присваивание идентификаторов
                    # obj_id += 1
                    cv2.putText(frame, f"ID: {id_obj[i]}", (seg['boxes'][i][0], seg['boxes'][i][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    os.rename(path_for_pic + f"/extract_img/El({id_obj[i]})_frame({counter}).jpg",
                              path_for_pic + f"/extract_img/El({id_obj[i]})_frame({counter})_id({id_obj[i]}).jpg")
            else:
                if len(ig_obj_help) > len(id_obj):
                    for l in id_arr:
                        for j in id_obj:
                            if os.path.exists(
                                    path_for_pic + f"/extract_img/El({l})_frame({counter - 1})_id({l}).jpg") == False or os.path.exists(
                                    path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg") == False:
                                continue
                            frame_befor = path_for_pic + f"/extract_img/El({l})_frame({counter - 1})_id({l}).jpg"
                            frame_now = path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg"

                            percent = compare_images(frame_befor, frame_now)
                            if percent > 45:
                                # os.rename(frame_now_new, path_for_pic + f"/extract_img/El({id_obj[i-1]})_frame({counter})_id({id_obj[i-1]}).jpg")
                                percent_count += 1
                                os.rename(frame_now,
                                          path_for_pic + f"/extract_img/El({l if l != j else j})_frame({counter})_id({l}).jpg")
                                im = Image.open(
                                    path_for_pic + f"/extract_img/El({l if l != j else j})_frame({counter})_id({l}).jpg")
                                # for box in seg['boxes']:
                                # if im.size[0]==(box[2]-box[0]) and im.size[1]==(box[3]-box[1]):
                                cv2.putText(frame, f"ID: {l}",
                                            (seg['boxes'][j - 1][2] - im.size[0], seg['boxes'][j - 1][3] - im.size[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    ig_obj_help = id_obj
                elif len(ig_obj_help) < len(id_obj):
                    none_obj = 0

                    count_none_obj = id_obj
                    for k in id_obj:
                        percent = 0
                        for l in id_obj:
                            mean_per = []
                            if os.path.exists(path_for_pic + f"/extract_img/El({k})_frame({counter}).jpg"):
                                img = Image.open(path_for_pic + f"/extract_img/El({k})_frame({counter}).jpg")

                            if img.size[0] < 190:
                                none_obj += 1
                                if k in count_none_obj:
                                    count_none_obj.pop(k - 1)
                            if os.path.exists(path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg") == False:
                                continue
                            if Image.open(path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg").size[0] < 190:
                                continue

                            for m in range(1, counter, 5):
                                frame_befor = path_for_pic + f"/extract_img/El({k})_frame({m})_id({k}).jpg"
                                frame_now = path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg"
                                if os.path.exists(
                                        path_for_pic + f"/extract_img/El({k})_frame({m})_id({k}).jpg") == False:
                                    continue
                                percent = compare_images(frame_befor, frame_now)
                                mean_per.append(percent)
                                # if percent > 43:
                                #    break

                                # ig_obj_help = count_none_obj

                            print(f"Среднее id({l}): ", np.mean(mean_per))
                            if np.mean(mean_per) > 35:
                                img.close()
                                os.rename(path_for_pic + f"/extract_img/El({l})_frame({counter}).jpg",
                                          path_for_pic + f"/extract_img/El({k if k != i else i})_frame({counter})_id({k}).jpg")
                                im = Image.open(
                                    path_for_pic + f"/extract_img/El({k if k != i else i})_frame({counter})_id({k}).jpg")
                                cv2.putText(frame, f"ID: {k}",
                                            (seg['boxes'][l - 1][2] - im.size[0], seg['boxes'][l - 1][3] - im.size[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                                if none_obj > 0:
                                    ig_obj_help = count_none_obj
                                else:
                                    ig_obj_help = id_obj
                                break

                    for i in id_obj:
                        if os.path.exists(path_for_pic + f"/extract_img/El({i})_frame({counter})_id({i}).jpg"):
                            continue
                        else:
                            if none_obj > 0:
                                ig_obj_help = count_none_obj
                                continue
                            # prev_img_size[len(id_arr)] = img.size[0]
                            img.close()
                            os.rename(path_for_pic + f"/extract_img/El({i})_frame({counter}).jpg",
                                      path_for_pic + f"/extract_img/El({len(id_arr) + 1})_frame({counter})_id({len(id_arr) + 1}).jpg")
                            im = Image.open(
                                path_for_pic + f"/extract_img/El({len(id_arr) + 1})_frame({counter})_id({len(id_arr) + 1}).jpg")
                            cv2.putText(frame, f"ID: {len(id_arr) + 1}",
                                        (seg['boxes'][i - 1][2] - im.size[0], seg['boxes'][i - 1][3] - im.size[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            id_arr.append(len(id_arr) + 1)
                            ig_obj_help = id_obj
                            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
                            cv2.imwrite(path_for_pic + f"/Detect_new_obj_{count_pic}.jpg",
                                        draw_object_count(frame, frame_duration_formatted, len(id_arr)))
                            count_pic += 1
                            x1, y1, x2, y2 = boxes[i - 1]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                            previous_boxes[len(id_arr)] = (x1, y1, x2, y2)
                else:

                    for i in range(1, (len(seg['boxes']) + 1)):
                        # obj_id += 1
                        percent_count_arr = []

                        for j in range(1, (len(seg['boxes']) + 1)):
                            if percent_count == len(seg['boxes']):  # ограничение на повторные проверки
                                break
                            frame_befor = path_for_pic + f"/extract_img/El({i})_frame({counter - 1})_id({id_obj[i - 1]}).jpg"
                            frame_now = path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg"
                            # if os.path.exists(path_for_pic + f"/extract_img/El({j})_frame({counter}).jpg")==False:
                            #        continue

                            if os.path.exists(
                                    path_for_pic + f"/extract_img/El({i})_frame({counter - 1})_id({id_obj[i - 1]}).jpg") == False:
                                for k in id_arr:
                                    if os.path.exists(
                                            path_for_pic + f"/extract_img/El({k})_frame({counter - 1})_id({k}).jpg"):
                                        frame_befor = path_for_pic + f"/extract_img/El({k})_frame({counter - 1})_id({k}).jpg"
                                        percent = compare_objects(frame_befor, frame_now)

                                        if percent > 25:
                                            # os.rename(frame_now_new, path_for_pic + f"/extract_img/El({id_obj[i-1]})_frame({counter})_id({id_obj[i-1]}).jpg")
                                            percent_count += 1
                                            os.rename(frame_now,
                                                      path_for_pic + f"/extract_img/El({k if k != j else j})_frame({counter})_id({k}).jpg")
                                            im = Image.open(
                                                path_for_pic + f"/extract_img/El({k if k != j else j})_frame({counter})_id({k}).jpg")
                                            # for box in seg['boxes']:
                                            # if im.size[0]==(box[2]-box[0]) and im.size[1]==(box[3]-box[1]):
                                            cv2.putText(frame, f"ID: {k}", (
                                            seg['boxes'][j - 1][2] - im.size[0], seg['boxes'][j - 1][3] - im.size[1]),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                            x1, y1, x2, y2 = boxes[j - 1]
                                            prev_x1, prev_y1, prev_x2, prev_y2 = previous_boxes[k]
                                            diff = abs(x1 - prev_x1) + abs(y1 - prev_y1) + abs(x2 - prev_x2) + abs(
                                                y2 - prev_y2)
                                            if diff > 25 and diff < 70:
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                                            previous_boxes[k] = (x1, y1, x2, y2)
                                            print(f"Объект {k}", diff)
                                break
                            if os.path.exists(frame_now) == False:
                                percent_count_arr.append(0)
                                continue

                            percent = compare_objects(frame_befor, frame_now)
                            percent_count_arr.append(percent)
                        if len(percent_count_arr) != 0:
                            percent_max = max(percent_count_arr)
                            percent_index = percent_count_arr.index(percent_max)
                            # percent_count+=1
                            os.rename(path_for_pic + f"/extract_img/El({(percent_index + 1)})_frame({counter}).jpg",
                                      path_for_pic + f"/extract_img/El({id_obj[i - 1] if id_obj[i - 1] != (percent_index + 1) else (percent_index + 1)})_frame({counter})_id({id_obj[i - 1]}).jpg")
                            im = Image.open(
                                path_for_pic + f"/extract_img/El({id_obj[i - 1] if id_obj[i - 1] != (percent_index + 1) else (percent_index + 1)})_frame({counter})_id({id_obj[i - 1]}).jpg")

                            cv2.putText(frame, f"ID: {i}", (
                            seg['boxes'][percent_index][2] - im.size[0], seg['boxes'][percent_index][3] - im.size[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            class_name[percent_index + 1] = seg['class_names'][percent_index]

                            x1, y1, x2, y2 = boxes[percent_index]
                            prev_x1, prev_y1, prev_x2, prev_y2 = previous_boxes[i]
                            diff = abs(x1 - prev_x1) + abs(y1 - prev_y1) + abs(x2 - prev_x2) + abs(y2 - prev_y2)
                            if diff > 25 and diff < 70:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                            previous_boxes[i] = (x1, y1, x2, y2)
                            print(f"Объект {percent_index + 1}", diff)

            if output_video_name is not None:
                save_video.write(output)
        else:
            break

    end = time.time()

    face_cascade = cv2.CascadeClassifier(
        'C:/Users/AndreyBayborodin/source/repos/test_rasp_pr/test_rasp_pr/env1/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    for i in id_arr:
        prev_img_size = 0
        for j in range(1, counter + 1):
            if os.path.exists(path_for_pic + f"/extract_img/El({i})_frame({j})_id({i}).jpg") == False:
                continue
            img = Image.open(path_for_pic + f"/extract_img/El({i})_frame({j})_id({i}).jpg")
            image = cv2.imread(path_for_pic + f"/extract_img/El({i})_frame({j})_id({i}).jpg")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.5, 8)
            if class_name[i] == 'person':
                if img.size[0] > prev_img_size and len(faces) > 0:
                    destination_path = os.path.join(path_for_pic, f"Best_photo_{i}.jpg")
                    cv2.imwrite(destination_path, image)
                    # cv2.imwrite(path_for_pic + f"/Best_photo_{i}.jpg", path_for_pic + f"/extract_img/El({i})_frame({counter}).jpg")
                    prev_img_size = img.size[0]
                    img.close()
            else:
                if img.size[0] > prev_img_size:
                    destination_path = os.path.join(path_for_pic, f"Best_photo_{i}.jpg")
                    cv2.imwrite(destination_path, image)
                    # cv2.imwrite(path_for_pic + f"/Best_photo_{i}.jpg", path_for_pic + f"/extract_img/El({i})_frame({counter}).jpg")
                    prev_img_size = img.size[0]
                    img.close()

    lb.set(f"Обработано {counter} фреймов за {end - start:.1f} секунд")
    print(f"Processed {counter} frames in {end - start:.1f} seconds")
    capture.release()
    if output_video_name is not None:
        save_video.release()
        # lb.set(f"Выявление объектов выполнено! \nВидео и скриншоты находятся в данной папке {path_for_pic}")

    label_com = ttk.Label(text="Выберите ID объекта", background="white", font=("Arial", 10))
    label_com.place(x=170, y=220, height=30, width=300)
    # object = ["люди", "транспорт", "велосипеды и мотоциклы", "животные", "спортинвертарь"]

    combobox_id = ttk.Combobox(textvariable=id_count, values=id_arr)
    combobox_id.place(x=170, y=250, height=30, width=200)
    style = ttk.Style()
    style.configure("TButton", justify="center")
    btn_id = ttk.Button(text="Информация",
                        command=partial(out_info, class_name, path_for_pic, counter))  # создаем кнопку из пакета ttk
    btn_id.place(x=400, y=250, height=40, width=100)  # размещаем кнопку в окне

    btn_best = ttk.Button(text="Наилучшее \nфото",
                          command=partial(out_best_photo, path_for_pic))  # создаем кнопку из пакета ttk
    btn_best.place(x=510, y=250, height=40, width=100)  # размещаем кнопку в окне

    btn_first = ttk.Button(text="Первое \nпоявление",
                           command=partial(out_first_detect, path_for_pic))  # создаем кнопку из пакета ttk
    btn_first.place(x=620, y=250, height=40, width=100)  # размещаем кнопку в окне

    label_out = ttk.Label(textvariable=lbl_out, background="white", font=("Arial", 10))
    label_out.place(x=170, y=290, height=100, width=400)
def out_best_photo(path_for_pic):
    ident = id_count.get()
    os.startfile(path_for_pic + f'/Best_photo_{ident}.jpg')


def out_first_detect(path_for_pic):
    ident = id_count.get()
    ident = int(ident)
    os.startfile(path_for_pic + f'/Detect_new_obj_{ident - 1}.jpg')
def out_info(class_name, path_for_pic, counter):
    ident = id_count.get()
    prev_size = [0, 0]
    prev_size_sm = [math.inf, math.inf]
    for i in range(counter + 1):
        if os.path.exists(path_for_pic + f"/extract_img/El({ident})_frame({i})_id({ident}).jpg") == False:
            continue
        imge = Image.open(path_for_pic + f"/extract_img/El({ident})_frame({i})_id({ident}).jpg")
        if (imge.size[0] * imge.size[1]) > (prev_size[0] * prev_size[1]):
            prev_size = imge.size
        imge.close()
    for j in range(counter + 1):
        if os.path.exists(path_for_pic + f"/extract_img/El({ident})_frame({j}).jpg"):
            imge = Image.open(path_for_pic + f"/extract_img/El({ident})_frame({j}).jpg")
            if (imge.size[0] * imge.size[1]) < (prev_size_sm[0] * prev_size_sm[1]):
                prev_size_sm = imge.size
            imge.close()
        if os.path.exists(path_for_pic + f"/extract_img/El({ident})_frame({j})_id({ident}).jpg"):
            imge = Image.open(path_for_pic + f"/extract_img/El({ident})_frame({j})_id({ident}).jpg")
            if (imge.size[0] * imge.size[1]) < (prev_size_sm[0] * prev_size_sm[1]):
                prev_size_sm = imge.size
            imge.close()
    ident = int(ident)
    lbl_out.set(
        f'Информация о выбранном объекте: \nПердварительный класс объекта: {class_name[ident]} \nНаибольший размер объекта: {prev_size[0]}x{prev_size[1]} (в пикселях: {prev_size[0] * prev_size[1]}) \nНаименьший размер объекта: {prev_size_sm[0]}x{prev_size_sm[1]} (в пикселях: {prev_size_sm[0] * prev_size_sm[1]})')

# /////////////////////////////////////////
# время скриншота
# /////////////////////////////////////////
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

# /////////////////////////////////////////
# TKINTER
# /////////////////////////////////////////
def open_info():
    showinfo(title="Информация",
             message="В данной папке будет сохранены файлы. После нажатия на кнопку необходимо выбрать файл для распознавания и каталог, в котором будет создана новая папка. \nВыполнение программы занимает некоторое время, дождитесь сообщения об успешном выполнении.")


root = Tk()  # создаем корневой объект - окно

root.title("Приложение на Tkinter")  # устанавливаем заголовок окна
root.geometry("1000x500")  # устанавливаем размеры окна
root.resizable(False, False)
root["bg"] = "white"

# logo = PhotoImage(file="logo.png")
image = Image.open('logo.png')
logo = image.resize((150, 80))
my_img = ImageTk.PhotoImage(logo)
label_logo = ttk.Label(image=my_img, text="Выявление подвижных объектов", compound="left", background="white",
                       font=("Arial", 25))
label_logo.place(height=60, width=1000)

label_object = ttk.Label(text="Выберите объект при необходимости", background="white", font=("Arial", 10))
label_object.place(x=170, y=120, height=30, width=300)
object = ["люди", "транспорт", "велосипеды и мотоциклы", "животные", "спортинвертарь"]
object_var = StringVar()
combobox_object = ttk.Combobox(textvariable=object_var, values=object)
combobox_object.place(x=170, y=150, height=30, width=200)
label_folder = ttk.Label(text="Введите название папки", background="white", font=("Arial", 10))
label_folder.place(x=400, y=120, height=30, width=150)


def open_info():
    showinfo(title="Информация",
             message="В данной папке будет сохранены файлы. После нажатия на кнопку необходимо выбрать файл для распознавания и каталог, в котором будет создана новая папка. \nВыполнение программы занимает некоторое время, дождитесь сообщения об успешном выполнении.")


image = Image.open('info.jpg')
info = image.resize((15, 15))
info_img = ImageTk.PhotoImage(info)
btn_info = ttk.Button(root, image=info_img, command=open_info)
btn_info.place(x=560, y=120, height=22, width=22)
enter = ttk.Entry()
enter.place(x=400, y=150, height=30, width=200)
btn_detection = ttk.Button(text="Распознать", command=object_detection_by_frame)  # создаем кнопку из пакета ttk
btn_detection.place(x=630, y=150, height=30, width=100)  # размещаем кнопку в окне
lb = StringVar()
label_out = ttk.Label(textvariable=lb, background="white", font=("Arial", 10))
label_out.place(x=170, y=180, height=30, width=300)
id_count = StringVar()
lbl_out = StringVar()

root.mainloop()