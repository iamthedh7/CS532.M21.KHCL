import os
import cv2
import random
import easygui
import pandas as pd
import matplotlib.pyplot as plt
from function.func_getCosineSimilarity import getCosineSimilarity
from function.func_inputProcess import inputProcessing


# READ A DATAFRAME
df = pd.read_pickle('./model/model.pickle')

while (1):

    print('Choose your image!')

    file = easygui.fileopenbox()

    feature, img = inputProcessing(file)

    df['similarity'] = df.apply(lambda x: getCosineSimilarity(feature, x['features']), axis=1)
    sorted_df = df.sort_values(by='similarity', ascending=False)  
    top = sorted_df.head(3)

    name = list(top.file)
    sililarity = list(top.similarity)
    label = list(top.label)

    name.insert(0, '*')
    sililarity.insert(0, '*')
    label.insert(0, '*')

    plt.figure(figsize=(15, 5))
    for i in range(len(name)):
        if (i == 0):
            plt.subplot(1, len(name), i+1)
            plt.title('YOUR PRODUCT:')
            plt.imshow(img[:, :, ::-1])
            plt.axis('off')
        else:   
            plt.subplot(1, len(name), i+1)
            plt.title('Label: ' + str(label[i]) + ' (' + str(round(sililarity[i], 5)) + ')')
            plt.imshow(cv2.imread(name[i])[:, :, ::-1])
            plt.axis('off')

    name = './history/' + str(random.randint(1000, 9999)) + '.jpg'
    plt.savefig(name)
    result = cv2.imread(name)
    cv2.imshow('result', result)
    cv2.waitKey()

    os.system('cls')
    print('Press [Y] to CONTINUE: ', end='')
    gonext = input()
    if (gonext != 'y') and (gonext != 'Y'):
        os.system('cls')
        break
    else:
        os.system('cls')
        continue