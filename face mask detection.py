import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score, classification_report
import cv2
import seaborn as sns 


input_data_path = '/kaggle/input/face-mask-detection/images'
annotations_path = "/kaggle/input/face-mask-detection/annotations"
images = [*os.listdir("/kaggle/input/face-mask-detection/images")]
output_data_path =  '.'
def parse_annotation(path):
    tree = ET.parse(path)
    root = tree.getroot()
    constants = {}
    objects = [child for child in root if child.tag == 'object']
    for element in tree.iter():
        if element.tag == 'filename':
            constants['file'] = element.text[0:-4]
        if element.tag == 'size':
            for dim in list(element):
                if dim.tag == 'width':
                    constants['width'] = int(dim.text)
                if dim.tag == 'height':
                    constants['height'] = int(dim.text)
                if dim.tag == 'depth':
                    constants['depth'] = int(dim.text)
    object_params = [parse_annotation_object(obj) for obj in objects]
    #print(constants)
    full_result = [merge(constants,ob) for ob in object_params]
    return full_result   


def parse_annotation_object(annotation_object):
    params = {}
    for param in list(annotation_object):
        if param.tag == 'name':
            params['name'] = param.text
        if param.tag == 'bndbox':
            for coord in list(param):
                if coord.tag == 'xmin':
                    params['xmin'] = int(coord.text)              
                if coord.tag == 'ymin':
                    params['ymin'] = int(coord.text)
                if coord.tag == 'xmax':
                    params['xmax'] = int(coord.text)
                if coord.tag == 'ymax':
                    params['ymax'] = int(coord.text)
            
    return params       
 
def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res
dataset = [parse_annotation(anno) for anno in glob.glob(annotations_path+"/*.xml") ]

full_dataset = sum(dataset, [])

df = pd.DataFrame(full_dataset)
df.shape
(4072, 9)
df.head()
file	width	height	depth	name	xmin	ymin	xmax	ymax
0	maksssksksss737	400	226	3	with_mask	28	55	46	71
1	maksssksksss737	400	226	3	with_mask	98	62	111	78
2	maksssksksss737	400	226	3	mask_weared_incorrect	159	50	193	90
3	maksssksksss737	400	226	3	with_mask	293	59	313	80
4	maksssksksss737	400	226	3	with_mask	352	51	372	72
final_test_image = 'maksssksksss0'
df_final_test = df.loc[df["file"] == final_test_image]
images.remove(f'{final_test_image}.png')
df = df.loc[df["file"] != final_test_image]
df
file	width	height	depth	name	xmin	ymin	xmax	ymax
0	maksssksksss737	400	226	3	with_mask	28	55	46	71
1	maksssksksss737	400	226	3	with_mask	98	62	111	78
2	maksssksksss737	400	226	3	mask_weared_incorrect	159	50	193	90
3	maksssksksss737	400	226	3	with_mask	293	59	313	80
4	maksssksksss737	400	226	3	with_mask	352	51	372	72
...	...	...	...	...	...	...	...	...	...
4067	maksssksksss13	400	226	3	with_mask	229	53	241	72
4068	maksssksksss138	400	267	3	with_mask	51	144	128	239
4069	maksssksksss138	400	267	3	with_mask	147	169	217	233
4070	maksssksksss138	400	267	3	with_mask	224	92	309	186
4071	maksssksksss212	267	400	3	with_mask	115	75	169	135
4069 rows × 9 columns

df["name"].value_counts()
with_mask                3231
without_mask              715
mask_weared_incorrect     123
Name: name, dtype: int64
df["name"].value_counts().plot(kind='barh')
plt.xlabel('Count', fontsize = 10, fontweight = 'bold')
plt.ylabel('name', fontsize = 10, fontweight = 'bold')
Text(0, 0.5, 'name')

labels = df['name'].unique()
directory = ['train', 'test', 'val']
output_data_path =  '.'

import os
for label in labels:
    for d in directory:
        path = os.path.join(output_data_path, d, label)
        if not os.path.exists(path):
            os.makedirs(path)
def crop_img(image_path, x_min, y_min, x_max, y_max):
    x_shift = (x_max - x_min) * 0.1
    y_shift = (y_max - y_min) * 0.1
    img = Image.open(image_path)
    cropped = img.crop((x_min - x_shift, y_min - y_shift, x_max + x_shift, y_max + y_shift))
    return cropped
def extract_faces(image_name, image_info):
    faces = []
    df_one_img = image_info[image_info['file'] == image_name[:-4]][['xmin', 'ymin', 'xmax', 'ymax', 'name']]
    for row_num in range(len(df_one_img)):
        x_min, y_min, x_max, y_max, label = df_one_img.iloc[row_num] 
        image_path = os.path.join(input_data_path, image_name)
        faces.append((crop_img(image_path, x_min, y_min, x_max, y_max), label,f'{image_name[:-4]}_{(x_min, y_min)}'))
    return faces
cropped_faces = [extract_faces(img, df) for img in images]
flat_cropped_faces = sum(cropped_faces, [])
with_mask = [(img, image_name) for img, label,image_name in flat_cropped_faces if label == "with_mask"]
mask_weared_incorrect = [(img, image_name) for img, label,image_name in flat_cropped_faces if label == "mask_weared_incorrect"]
without_mask = [(img, image_name) for img, label,image_name in flat_cropped_faces if label == "without_mask"]
print(len(with_mask))
print(len(without_mask))
print(len(mask_weared_incorrect))
print(len(with_mask) + len(without_mask) + len(mask_weared_incorrect))
3231
715
123
4069
train_with_mask, test_with_mask = train_test_split(with_mask, test_size=0.20, random_state=42)
test_with_mask, val_with_mask = train_test_split(test_with_mask, test_size=0.7, random_state=42)

train_mask_weared_incorrect, test_mask_weared_incorrect = train_test_split(mask_weared_incorrect, test_size=0.20, random_state=42)
test_mask_weared_incorrect, val_mask_weared_incorrect = train_test_split(test_mask_weared_incorrect, test_size=0.7, random_state=42)

train_without_mask, test_without_mask = train_test_split(without_mask, test_size=0.20, random_state=42)
test_without_mask, val_without_mask = train_test_split(test_without_mask, test_size=0.7, random_state=42)
def save_image(image, image_name, output_data_path,  dataset_type, label):
    output_path = os.path.join(output_data_path, dataset_type, label ,f'{image_name}.png')
    image.save(output_path)  
for image, image_name in train_with_mask:
    save_image(image, image_name, output_data_path, 'train', 'with_mask')

for image, image_name in train_mask_weared_incorrect:
    save_image(image, image_name, output_data_path, 'train', 'mask_weared_incorrect')

for image, image_name in train_without_mask:
    save_image(image, image_name, output_data_path, 'train', 'without_mask')

for image, image_name in test_with_mask:
    save_image(image, image_name, output_data_path, 'test', 'with_mask')

for image, image_name in test_mask_weared_incorrect:
    save_image(image, image_name, output_data_path, 'test', 'mask_weared_incorrect')

for image, image_name in test_without_mask:
    save_image(image, image_name, output_data_path, 'test', 'without_mask')
        
for image, image_name in val_with_mask:
    save_image(image, image_name, output_data_path, 'val', 'with_mask')

for image, image_name in val_without_mask:
    save_image(image, image_name, output_data_path, 'val', 'without_mask')

for image, image_name in val_mask_weared_incorrect:
    save_image(image, image_name, output_data_path, 'val', 'mask_weared_incorrect')
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = 3,  padding='same', activation = 'relu', input_shape = (35,35,3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 32, kernel_size = 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(filters = 64, kernel_size = 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(units = 500, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 3, activation = 'softmax'))

model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 35, 35, 16)        448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 17, 17, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 17, 17, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 64)          18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
dropout (Dropout)            (None, 4, 4, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 500)               512500    
_________________________________________________________________
dropout_1 (Dropout)          (None, 500)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 1503      
=================================================================
Total params: 537,587
Trainable params: 537,587
Non-trainable params: 0
_________________________________________________________________
2022-10-10 03:20:29.985537: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
batch_size = 8
epochs = 50

datagen = ImageDataGenerator(
    rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1,
    height_shift_range=0.1, rotation_range=4, vertical_flip=False

)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)
 

train_generator = datagen.flow_from_directory(
    directory='/kaggle/working/train', 
    target_size = (35,35),
    class_mode="categorical", batch_size=batch_size, shuffle=True

)

# Validation data
val_generator = val_datagen.flow_from_directory(
    directory='/kaggle/working/val', 
    target_size = (35,35),
    class_mode="categorical", batch_size=batch_size, shuffle=True
)

# Test data
test_generator = val_datagen.flow_from_directory(
    directory='/kaggle/working/test', 
    target_size = (35,35),
    class_mode="categorical", batch_size=batch_size, shuffle=False
)
Found 3254 images belonging to 3 classes.
Found 572 images belonging to 3 classes.
Found 243 images belonging to 3 classes.
data_size = len(train_generator)

steps_per_epoch = int(data_size / batch_size)
print(f"steps_per_epoch: {steps_per_epoch}")

val_steps = int(len(val_generator) // batch_size)
print(f"val_steps: {val_steps}")
steps_per_epoch: 50
val_steps: 9
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuracy', 'Recall', 'Precision', 'AUC']
)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lrr = ReduceLROnPlateau(monitor='val_loss',patience=8,verbose=1,factor=0.5, min_lr=0.00001)
model_history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    shuffle=True,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[early_stopping, lrr]
)
/opt/conda/lib/python3.7/site-packages/keras/engine/training.py:1972: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
  warnings.warn('`Model.fit_generator` is deprecated and '
2022-10-10 03:20:30.737895: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/50
50/50 [==============================] - 3s 32ms/step - loss: 0.6627 - accuracy: 0.7825 - recall: 0.7425 - precision: 0.7984 - auc: 0.8839 - val_loss: 0.5597 - val_accuracy: 0.7500 - val_recall: 0.7500 - val_precision: 0.7500 - val_auc: 0.9383
Epoch 2/50
50/50 [==============================] - 1s 19ms/step - loss: 0.4503 - accuracy: 0.8100 - recall: 0.7925 - precision: 0.8087 - auc: 0.9476 - val_loss: 0.3292 - val_accuracy: 0.9167 - val_recall: 0.8750 - val_precision: 0.9403 - val_auc: 0.9774
Epoch 3/50
50/50 [==============================] - 1s 19ms/step - loss: 0.3610 - accuracy: 0.8900 - recall: 0.8600 - precision: 0.9053 - auc: 0.9621 - val_loss: 0.2114 - val_accuracy: 0.9306 - val_recall: 0.9306 - val_precision: 0.9437 - val_auc: 0.9887
Epoch 4/50
50/50 [==============================] - 1s 19ms/step - loss: 0.3051 - accuracy: 0.9175 - recall: 0.9125 - precision: 0.9194 - auc: 0.9726 - val_loss: 0.2479 - val_accuracy: 0.9028 - val_recall: 0.8889 - val_precision: 0.9014 - val_auc: 0.9774
Epoch 5/50
50/50 [==============================] - 1s 19ms/step - loss: 0.2247 - accuracy: 0.9350 - recall: 0.9300 - precision: 0.9370 - auc: 0.9821 - val_loss: 0.3233 - val_accuracy: 0.9167 - val_recall: 0.9028 - val_precision: 0.9286 - val_auc: 0.9691
Epoch 6/50
50/50 [==============================] - 1s 18ms/step - loss: 0.3051 - accuracy: 0.8950 - recall: 0.8925 - precision: 0.9061 - auc: 0.9730 - val_loss: 0.2947 - val_accuracy: 0.9167 - val_recall: 0.9167 - val_precision: 0.9167 - val_auc: 0.9783
Epoch 7/50
50/50 [==============================] - 1s 20ms/step - loss: 0.2895 - accuracy: 0.9095 - recall: 0.9020 - precision: 0.9135 - auc: 0.9745 - val_loss: 0.3259 - val_accuracy: 0.8750 - val_recall: 0.8750 - val_precision: 0.8873 - val_auc: 0.9769
Epoch 8/50
50/50 [==============================] - 1s 21ms/step - loss: 0.2588 - accuracy: 0.9425 - recall: 0.9350 - precision: 0.9468 - auc: 0.9773 - val_loss: 0.2764 - val_accuracy: 0.8889 - val_recall: 0.8889 - val_precision: 0.8889 - val_auc: 0.9810
Epoch 9/50
50/50 [==============================] - 1s 19ms/step - loss: 0.1944 - accuracy: 0.9425 - recall: 0.9300 - precision: 0.9442 - auc: 0.9874 - val_loss: 1.0626 - val_accuracy: 0.7778 - val_recall: 0.7778 - val_precision: 0.7778 - val_auc: 0.9029
Epoch 10/50
50/50 [==============================] - 1s 19ms/step - loss: 0.2848 - accuracy: 0.9175 - recall: 0.9100 - precision: 0.9262 - auc: 0.9743 - val_loss: 0.2790 - val_accuracy: 0.9028 - val_recall: 0.9028 - val_precision: 0.9155 - val_auc: 0.9752
Epoch 11/50
50/50 [==============================] - 2s 32ms/step - loss: 0.2354 - accuracy: 0.9375 - recall: 0.9350 - precision: 0.9444 - auc: 0.9811 - val_loss: 0.2187 - val_accuracy: 0.9167 - val_recall: 0.9167 - val_precision: 0.9167 - val_auc: 0.9898

Epoch 00011: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
model_loss, model_acc, recall, precision, auc = model.evaluate(test_generator)
print(f'{model_loss} || {model_acc*100} || {recall*100} || {precision*100} || {auc*100}')
31/31 [==============================] - 0s 7ms/step - loss: 0.2676 - accuracy: 0.9300 - recall: 0.9218 - precision: 0.9295 - auc: 0.9759
0.26758328080177307 || 93.00411343574524 || 92.18106865882874 || 92.94605851173401 || 97.58548736572266
predictions = model.predict(test_generator)
predictions
array([[8.03483091e-03, 9.27638635e-02, 8.99201334e-01],
       [6.02829754e-02, 9.31686163e-01, 8.03078152e-03],
       [4.92345057e-02, 9.43306863e-01, 7.45862769e-03],
       [1.10853739e-01, 7.90212452e-01, 9.89338160e-02],
       [7.38791153e-02, 9.19933438e-01, 6.18744222e-03],
       [7.86243007e-02, 8.98281991e-01, 2.30937731e-02],
       [1.70814656e-02, 9.82698679e-01, 2.19930749e-04],
       [4.02383804e-02, 8.35741699e-01, 1.24019854e-01],
       [2.62128003e-02, 8.47933531e-01, 1.25853658e-01],
       [1.07074678e-01, 8.42779875e-01, 5.01454137e-02],
       [5.26406318e-02, 9.13943052e-01, 3.34162712e-02],
       [9.84749720e-02, 8.60552132e-01, 4.09728363e-02],
       [6.03106245e-02, 9.35034335e-01, 4.65509528e-03],
       [3.83005999e-02, 9.60671544e-01, 1.02782110e-03],
       [7.62078464e-02, 9.17212665e-01, 6.57947967e-03],
       [6.49337471e-02, 9.30098534e-01, 4.96765226e-03],
       [5.36389202e-02, 7.22287357e-01, 2.24073693e-01],
       [7.00742006e-02, 9.21898723e-01, 8.02713260e-03],
       [5.59180751e-02, 9.40442920e-01, 3.63898394e-03],
       [3.30758505e-02, 9.66021359e-01, 9.02680040e-04],
       [3.87818031e-02, 9.56827700e-01, 4.39048372e-03],
       [6.53701425e-02, 9.27099645e-01, 7.53024686e-03],
       [2.23682933e-02, 9.77172554e-01, 4.59191389e-04],
       [9.57977325e-02, 8.85131299e-01, 1.90709475e-02],
       [5.10506928e-02, 9.47511792e-01, 1.43751246e-03],
       [5.43460362e-02, 9.11508560e-01, 3.41453999e-02],
       [1.62239242e-02, 9.83428061e-01, 3.47980007e-04],
       [3.71883437e-02, 9.61230338e-01, 1.58130436e-03],
       [3.50842625e-02, 9.64410007e-01, 5.05742326e-04],
       [6.38457090e-02, 9.26914513e-01, 9.23972018e-03],
       [5.27145602e-02, 9.40743387e-01, 6.54205214e-03],
       [5.68447523e-02, 9.40741956e-01, 2.41338369e-03],
       [1.57756079e-02, 9.83831167e-01, 3.93255905e-04],
       [3.97342779e-02, 9.58793163e-01, 1.47257478e-03],
       [4.49391007e-02, 9.20048237e-01, 3.50126438e-02],
       [3.92673686e-02, 9.59785283e-01, 9.47326364e-04],
       [2.34985631e-02, 9.76241648e-01, 2.59869470e-04],
       [1.33115938e-02, 9.86569822e-01, 1.18521784e-04],
       [1.30943075e-01, 7.30812252e-01, 1.38244659e-01],
       [8.77299756e-02, 8.67702425e-01, 4.45676260e-02],
       [2.78967656e-02, 9.71119165e-01, 9.84062557e-04],
       [7.11283013e-02, 9.02297080e-01, 2.65746582e-02],
       [2.64359266e-02, 9.72920477e-01, 6.43556239e-04],
       [6.75828904e-02, 9.25599337e-01, 6.81777205e-03],
       [4.14061472e-02, 9.53074157e-01, 5.51965879e-03],
       [7.06794485e-02, 9.16455925e-01, 1.28646186e-02],
       [1.51303457e-02, 9.84695971e-01, 1.73678505e-04],
       [1.04759596e-01, 8.15777540e-01, 7.94628412e-02],
       [3.55451070e-02, 9.63261724e-01, 1.19316461e-03],
       [9.13429782e-02, 7.53201902e-01, 1.55455053e-01],
       [3.04504111e-02, 9.65832412e-01, 3.71714728e-03],
       [3.83088626e-02, 9.59890544e-01, 1.80062128e-03],
       [6.25852272e-02, 9.19783771e-01, 1.76309999e-02],
       [5.43116070e-02, 9.43275034e-01, 2.41337833e-03],
       [6.94819838e-02, 8.07376802e-01, 1.23141207e-01],
       [3.21680158e-02, 9.67186987e-01, 6.45023421e-04],
       [2.31388565e-02, 9.76534247e-01, 3.26840585e-04],
       [2.90842988e-02, 9.70284820e-01, 6.30840950e-04],
       [3.40622552e-02, 9.64487731e-01, 1.44991733e-03],
       [1.00385949e-01, 8.89137030e-01, 1.04770139e-02],
       [6.30129650e-02, 9.31155741e-01, 5.83123742e-03],
       [5.00239134e-02, 9.48790193e-01, 1.18588074e-03],
       [4.02884781e-02, 9.58389044e-01, 1.32247899e-03],
       [2.88474876e-02, 9.70497549e-01, 6.54899981e-04],
       [5.67672811e-02, 9.26045179e-01, 1.71876065e-02],
       [7.03675374e-02, 9.27819490e-01, 1.81294093e-03],
       [6.14399798e-02, 9.19294596e-01, 1.92654226e-02],
       [2.24543661e-02, 9.75344658e-01, 2.20103655e-03],
       [2.05082744e-02, 9.79200184e-01, 2.91552511e-04],
       [4.14652452e-02, 9.58074510e-01, 4.60306590e-04],
       [1.45272210e-01, 8.32909107e-01, 2.18187105e-02],
       [1.62594393e-02, 9.83611286e-01, 1.29228531e-04],
       [9.06379372e-02, 9.02879953e-01, 6.48212293e-03],
       [6.44188970e-02, 9.28610444e-01, 6.97066495e-03],
       [7.74367377e-02, 9.04748678e-01, 1.78146604e-02],
       [2.18960382e-02, 9.77244914e-01, 8.58994725e-04],
       [1.31898493e-01, 8.09363782e-01, 5.87377623e-02],
       [9.65252072e-02, 7.78960168e-01, 1.24514692e-01],
       [4.45412807e-02, 9.45023060e-01, 1.04356008e-02],
       [2.62288600e-02, 9.73377943e-01, 3.93200869e-04],
       [4.11402322e-02, 9.54621017e-01, 4.23880247e-03],
       [7.61056617e-02, 8.77371430e-01, 4.65229563e-02],
       [1.18264928e-01, 7.06995904e-01, 1.74739212e-01],
       [4.26916480e-02, 9.52773869e-01, 4.53445176e-03],
       [3.56441550e-02, 9.62950885e-01, 1.40494609e-03],
       [4.02941070e-02, 9.58214462e-01, 1.49148470e-03],
       [9.35787782e-02, 8.41265380e-01, 6.51558116e-02],
       [3.59989218e-02, 9.63238716e-01, 7.62305397e-04],
       [1.71336960e-02, 9.82482553e-01, 3.83862585e-04],
       [8.09719488e-02, 8.97767425e-01, 2.12606527e-02],
       [1.10011259e-02, 9.88849699e-01, 1.49122425e-04],
       [4.65395711e-02, 9.50695157e-01, 2.76530092e-03],
       [4.95830961e-02, 9.49410975e-01, 1.00593618e-03],
       [1.16148461e-02, 9.88212407e-01, 1.72721455e-04],
       [1.85823645e-02, 9.80901182e-01, 5.16492990e-04],
       [2.22989097e-02, 9.76613045e-01, 1.08811154e-03],
       [2.54743025e-02, 9.74326789e-01, 1.98872236e-04],
       [7.32567683e-02, 9.23369229e-01, 3.37400427e-03],
       [1.29906982e-01, 8.38371217e-01, 3.17218713e-02],
       [3.96819934e-02, 9.53984499e-01, 6.33350667e-03],
       [9.59271789e-02, 8.29679966e-01, 7.43928328e-02],
       [2.33416129e-02, 9.76233900e-01, 4.24505764e-04],
       [3.49852443e-02, 9.63706374e-01, 1.30835758e-03],
       [6.86422959e-02, 9.22056973e-01, 9.30068083e-03],
       [2.80030910e-02, 9.71293688e-01, 7.03126483e-04],
       [1.59951136e-01, 6.19580686e-01, 2.20468208e-01],
       [6.25211671e-02, 9.32054281e-01, 5.42454701e-03],
       [4.14021499e-02, 9.56714332e-01, 1.88357860e-03],
       [6.67449534e-02, 9.15514946e-01, 1.77400950e-02],
       [2.18357611e-02, 9.24981177e-01, 5.31830937e-02],
       [6.63034394e-02, 9.29968297e-01, 3.72821745e-03],
       [6.54188320e-02, 9.26880658e-01, 7.70059414e-03],
       [3.48735675e-02, 9.63772774e-01, 1.35359389e-03],
       [4.64723557e-02, 9.52854514e-01, 6.73122297e-04],
       [5.71748614e-02, 9.37993884e-01, 4.83119767e-03],
       [1.60813015e-02, 9.82164860e-01, 1.75394223e-03],
       [1.81026831e-02, 9.80428934e-01, 1.46828278e-03],
       [9.35787559e-02, 8.89309108e-01, 1.71121731e-02],
       [2.22502705e-02, 9.77458656e-01, 2.91125296e-04],
       [7.98060670e-02, 7.56934106e-01, 1.63259804e-01],
       [6.74780086e-02, 8.51860166e-01, 8.06618258e-02],
       [3.57322991e-02, 9.52524185e-01, 1.17435539e-02],
       [1.45056844e-01, 6.65714860e-01, 1.89228311e-01],
       [1.05565310e-01, 8.81647050e-01, 1.27876541e-02],
       [7.74581879e-02, 9.16043818e-01, 6.49802154e-03],
       [6.40826747e-02, 9.27407444e-01, 8.50991439e-03],
       [5.90807721e-02, 8.94800901e-01, 4.61184010e-02],
       [5.46210222e-02, 9.41144586e-01, 4.23438940e-03],
       [1.00960784e-01, 8.70503426e-01, 2.85358075e-02],
       [9.00116265e-02, 8.99246931e-01, 1.07414229e-02],
       [3.05176843e-02, 9.65788722e-01, 3.69359553e-03],
       [3.88311706e-02, 9.51857507e-01, 9.31131188e-03],
       [4.23070677e-02, 8.02758992e-01, 1.54933974e-01],
       [1.15455844e-01, 4.60979283e-01, 4.23564792e-01],
       [1.46556765e-01, 5.93505263e-01, 2.59938061e-01],
       [8.76323283e-02, 8.94221067e-01, 1.81466099e-02],
       [3.65352817e-02, 9.62467730e-01, 9.96888615e-04],
       [1.05014548e-01, 8.78196657e-01, 1.67887490e-02],
       [3.43045704e-02, 9.60910261e-01, 4.78519686e-03],
       [2.51091179e-02, 9.74601090e-01, 2.89800955e-04],
       [9.58625823e-02, 8.77281547e-01, 2.68558376e-02],
       [3.64484638e-02, 9.62541521e-01, 1.01001363e-03],
       [3.82416882e-02, 9.56744969e-01, 5.01333131e-03],
       [6.40996918e-02, 9.14635539e-01, 2.12648269e-02],
       [8.05144385e-02, 9.07565236e-01, 1.19203655e-02],
       [5.35930693e-02, 9.39660192e-01, 6.74671633e-03],
       [4.77573574e-02, 4.06879187e-01, 5.45363486e-01],
       [7.51394555e-02, 5.57045758e-01, 3.67814809e-01],
       [6.97774664e-02, 9.13618147e-01, 1.66043099e-02],
       [3.80437709e-02, 9.58543837e-01, 3.41235101e-03],
       [7.28335753e-02, 9.11360562e-01, 1.58058535e-02],
       [4.22288626e-02, 9.18933749e-01, 3.88373099e-02],
       [4.24888358e-02, 9.52528834e-01, 4.98230755e-03],
       [3.85975987e-02, 9.51077282e-01, 1.03251748e-02],
       [6.29923269e-02, 9.34522629e-01, 2.48500099e-03],
       [9.56324860e-02, 8.71140361e-01, 3.32271568e-02],
       [1.37021571e-01, 7.81497180e-01, 8.14813003e-02],
       [4.95249825e-03, 9.95038092e-01, 9.41414328e-06],
       [3.61339301e-02, 9.62873101e-01, 9.92993126e-04],
       [9.22600254e-02, 8.98737073e-01, 9.00293980e-03],
       [3.52463871e-02, 6.78922832e-01, 2.85830826e-01],
       [8.83138254e-02, 3.59059989e-01, 5.52626133e-01],
       [1.59218103e-01, 7.90493369e-01, 5.02885692e-02],
       [2.93774921e-02, 9.70312655e-01, 3.09914263e-04],
       [1.21798679e-01, 7.38281310e-01, 1.39919907e-01],
       [9.49213430e-02, 8.96099806e-01, 8.97880830e-03],
       [1.39329672e-01, 7.94225097e-01, 6.64452761e-02],
       [1.22143179e-01, 8.33266914e-01, 4.45898250e-02],
       [8.16494450e-02, 9.11946893e-01, 6.40375447e-03],
       [9.94257480e-02, 8.84963334e-01, 1.56109333e-02],
       [7.66994581e-02, 9.11240041e-01, 1.20604495e-02],
       [1.72653254e-02, 9.82568204e-01, 1.66469792e-04],
       [7.19833523e-02, 8.71147335e-01, 5.68692498e-02],
       [4.11557266e-03, 8.33400115e-02, 9.12544370e-01],
       [1.53971344e-01, 7.85076022e-01, 6.09525517e-02],
       [4.28079329e-02, 9.50678468e-01, 6.51366264e-03],
       [2.88409125e-02, 9.65813756e-01, 5.34532499e-03],
       [4.03667102e-03, 9.95948255e-01, 1.50316919e-05],
       [7.40321353e-02, 9.19373274e-01, 6.59461366e-03],
       [4.42872494e-02, 9.52526987e-01, 3.18580284e-03],
       [1.05673738e-01, 8.82321000e-01, 1.20052649e-02],
       [4.00674790e-02, 9.55099106e-01, 4.83337510e-03],
       [7.51970261e-02, 9.17662323e-01, 7.14071793e-03],
       [1.79281179e-02, 9.81972396e-01, 9.95212758e-05],
       [4.29974794e-02, 3.14827025e-01, 6.42175436e-01],
       [3.98885831e-02, 9.57948208e-01, 2.16326932e-03],
       [6.96011484e-02, 8.52074087e-01, 7.83247575e-02],
       [1.16724245e-01, 7.91813314e-01, 9.14624482e-02],
       [1.05852606e-02, 2.37803757e-01, 7.51610935e-01],
       [3.54972892e-02, 9.23104227e-01, 4.13984247e-02],
       [8.76320805e-03, 9.91212428e-01, 2.43960094e-05],
       [1.31422475e-01, 7.78105557e-01, 9.04719308e-02],
       [3.30145843e-02, 8.23811471e-01, 1.43173933e-01],
       [3.55564468e-02, 9.50094461e-01, 1.43491030e-02],
       [2.60598809e-02, 9.73653138e-01, 2.86937982e-04],
       [1.29029825e-01, 8.07977080e-01, 6.29930571e-02],
       [2.41479762e-02, 9.75321114e-01, 5.30899037e-04],
       [3.95569205e-02, 9.58683610e-01, 1.75951771e-03],
       [1.38306588e-01, 7.68524468e-01, 9.31689367e-02],
       [5.34054078e-02, 9.34038043e-01, 1.25564793e-02],
       [2.47305669e-02, 9.71595526e-01, 3.67392623e-03],
       [2.73657404e-02, 1.90275952e-01, 7.82358289e-01],
       [3.73688387e-03, 7.00738803e-02, 9.26189244e-01],
       [2.61465018e-03, 6.87204376e-02, 9.28664923e-01],
       [2.85438448e-03, 5.70111088e-02, 9.40134525e-01],
       [2.12964360e-02, 1.98649213e-01, 7.80054390e-01],
       [1.21205986e-01, 6.40656650e-01, 2.38137320e-01],
       [3.64043675e-02, 2.02212378e-01, 7.61383295e-01],
       [6.39490560e-02, 3.61077487e-01, 5.74973404e-01],
       [4.12025787e-02, 2.29376927e-01, 7.29420424e-01],
       [3.65858413e-02, 1.99038073e-01, 7.64376104e-01],
       [8.49022996e-03, 1.01350807e-01, 8.90158951e-01],
       [8.25606734e-02, 4.13559079e-01, 5.03880262e-01],
       [8.78208131e-03, 1.18009701e-01, 8.73208165e-01],
       [9.08074528e-03, 9.21075642e-02, 8.98811698e-01],
       [1.01755075e-02, 1.16623007e-01, 8.73201549e-01],
       [6.10734858e-02, 2.29177430e-01, 7.09749043e-01],
       [1.90790772e-01, 6.17552400e-01, 1.91656798e-01],
       [6.09292788e-03, 1.00736834e-01, 8.93170297e-01],
       [2.54013874e-02, 2.79162526e-01, 6.95436060e-01],
       [1.00756073e-02, 1.20318279e-01, 8.69606078e-01],
       [1.16745709e-02, 1.30846843e-01, 8.57478559e-01],
       [8.32498446e-02, 3.24534476e-01, 5.92215717e-01],
       [1.53892590e-02, 1.45676374e-01, 8.38934362e-01],
       [6.80491850e-02, 2.51922160e-01, 6.80028677e-01],
       [7.25067547e-03, 8.14720914e-02, 9.11277294e-01],
       [8.58442038e-02, 4.27580982e-01, 4.86574799e-01],
       [3.58058363e-02, 2.09976405e-01, 7.54217744e-01],
       [8.68298300e-03, 1.83430731e-01, 8.07886302e-01],
       [2.04740558e-02, 1.37076601e-01, 8.42449307e-01],
       [6.07595593e-02, 5.16441464e-01, 4.22798991e-01],
       [7.34972162e-03, 8.12818632e-02, 9.11368430e-01],
       [8.52203965e-02, 3.46970260e-01, 5.67809284e-01],
       [1.13500319e-01, 5.25078237e-01, 3.61421466e-01],
       [3.50131094e-02, 1.70067221e-01, 7.94919729e-01],
       [1.08075894e-01, 7.81937182e-01, 1.09986939e-01],
       [1.02301594e-02, 9.99429822e-02, 8.89826894e-01],
       [3.77578065e-02, 1.81778386e-01, 7.80463755e-01],
       [2.14321278e-02, 2.08409473e-01, 7.70158350e-01],
       [3.17465775e-02, 2.57812917e-01, 7.10440576e-01],
       [2.58232411e-02, 1.44231170e-01, 8.29945505e-01],
       [1.64099466e-02, 1.81300536e-01, 8.02289546e-01],
       [5.32712787e-02, 3.07839483e-01, 6.38889253e-01]], dtype=float32)
def plot_loss_and_accuracy(history):
    history_df = pd.DataFrame(history)
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    
    history_df.loc[0:, ['loss', 'val_loss']].plot(ax=ax[0])
    ax[0].set(xlabel = 'epoch number', ylabel = 'loss')

    history_df.loc[0:, ['accuracy', 'val_accuracy']].plot(ax=ax[1])
    ax[1].set(xlabel = 'epoch number', ylabel = 'accuracy')
plot_loss_and_accuracy(model_history.history)

paths = test_generator.filenames
y_pred = model.predict(test_generator).argmax(axis=1)
classes = test_generator.class_indices

a_img_rand = np.random.randint(0,len(paths))
img = cv2.imread(os.path.join(output_data_path,'test', paths[a_img_rand]))      
colored_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

plt.imshow(colored_img)
true_label = paths[a_img_rand].split('/')[0]
predicted_label = list(classes)[y_pred[a_img_rand]]
print(f'{predicted_label} || {true_label}')
with_mask || with_mask

def evaluation(y, y_hat, title = 'Confusion Matrix'):
    cm = confusion_matrix(y, y_hat)
    sns.heatmap(cm,  cmap= 'PuBu', annot=True, fmt='g', annot_kws={'size':20})
    plt.xlabel('predicted', fontsize=18)
    plt.ylabel('actual', fontsize=18)
    plt.title(title, fontsize=18)
    
    plt.show()
y_true = test_generator.labels
y_pred = model.predict(test_generator).argmax(axis=1) # Predict prob and get Class Indices

evaluation(y_true, y_pred)

display(classes)
np.bincount(y_pred)
{'mask_weared_incorrect': 0, 'with_mask': 1, 'without_mask': 2}
array([  0, 200,  43])