import cv2
from rembg import remove
from function.func_crop import crop_rect

def inputProcessing(input_path):
    # + READ IMAGE
    input = cv2.imread(input_path)
    input = cv2.resize(input, (600, 800))

    # + REMOVE BACKGROUND
    output = remove(input)

    # + GET THE OBJECT'S BOUNDING BOX
    output = cv2.cvtColor(output, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, 1, 2)
    max_area = 0
    best_cnt = None
    for counter in contours:
        area = cv2.contourArea(counter)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = counter

    rect = cv2.minAreaRect(best_cnt)

    # + CROP TO THE OBJECT
    img_crop = crop_rect(output, rect)
    img = crop_rect(input, rect)

    # + GET FEATURES
    feature = cv2.resize(img_crop, (224, 224))
    feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB) / 255
    feature = feature.ravel()
    
    return feature, img