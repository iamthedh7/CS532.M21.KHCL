import os
import cv2
import pandas as pd
from rembg import remove
from function.func_crop import crop_rect


# CREATE A DATAFRAME
df = pd.DataFrame(columns=['label', 'file', 'features'])

# PROCESSING
folders = os.listdir('D:\\CC\\data')
label = 0
for i in sorted(folders):

    files = os.listdir('D:\\CC\\data\\' + i)

    for j in files:

        print('\n\n\n--- SOLVING', i, '[', j, '] ---\n\n\n')

        # + READ IMAGE
        input_path = 'D:\\CC\\data\\' + i + '\\' + j
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

        # + GET FEATURES
        feature = cv2.resize(img_crop, (224, 224))
        feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB) / 255
        feature = feature.ravel()

        # + SAVE TO DATABASE
        temp_df = pd.DataFrame(columns=['label', 'file', 'features'])
        temp_df['label'] = [label]
        temp_df['file'] = [input_path]
        temp_df['features'] = temp_df.apply(lambda x: feature, axis = 1)

        df = pd.concat([df, temp_df])
        os.system('cls')

    label += 1

df.to_pickle('D:\\CC\\model\\model.pickle')
print('SAVED!')