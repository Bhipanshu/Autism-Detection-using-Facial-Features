import dlib 
import cv2
import os
import cv2
import dlib
from PIL import Image
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model_files.modeling import VisionTransformer, CONFIGS
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = CONFIGS['ViT-B_16']

model_eyes = VisionTransformer(config,256, zero_head=True, num_classes=2)
model_path = './model_checkpoint/eyes_checkpoint.bin'
state_dict = torch.load(model_path)
model_eyes.load_state_dict(state_dict)
model_eyes.to(device)

model_nose = VisionTransformer(config,256, zero_head=True, num_classes=2)
model_nose.to(device)
model_path = './model_checkpoint/nose_checkpoint.bin'
state_dict = torch.load(model_path)
model_nose.load_state_dict(state_dict)

model_lips = VisionTransformer(config,256, zero_head=True, num_classes=2)
model_lips.to(device)
model_path ='./model_checkpoint/lips_checkpoint.bin'
state_dict = torch.load(model_path)
model_lips.load_state_dict(state_dict)

predictor = dlib.shape_predictor("./model_checkpoint/shape_predictor_68_face_landmarks.dat")

transform_train = transforms.Compose([
    transforms.RandomResizedCrop((256, 256), scale=(0.05, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


detector = dlib.get_frontal_face_detector()

# for autistic image folder
input_folder = './Data/Faces/test/autistic'
output_folder = 'Results/autistic_output'

os.makedirs(output_folder, exist_ok=True)

# Loop through all the files in the input folder
for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
                # Construct the full path of the input image
                image_path = os.path.join(input_folder, filename)
                print(filename)
                image = cv2.imread(image_path)
                image_save = image.copy()
                faces = detector(image)
                for face in faces:
                        landmarks = predictor(image_save, face)
                        #for eyes
                        x_min_face = min(landmarks.part(17).x, landmarks.part(41).x + 6, landmarks.part(19).x, landmarks.part(24).x, landmarks.part(26).x, landmarks.part(47).x+ 6)
                        y_min_face = min(landmarks.part(17).y, landmarks.part(41).y + 6, landmarks.part(19).y, landmarks.part(24).y, landmarks.part(26).y, landmarks.part(47).y+ 6)
                        x_max_face = max(landmarks.part(17).x, landmarks.part(41).x + 6, landmarks.part(19).x, landmarks.part(24).x, landmarks.part(26).x, landmarks.part(47).x+ 6)
                        y_max_face = max(landmarks.part(17).y, landmarks.part(41).y + 6, landmarks.part(19).y, landmarks.part(24).y, landmarks.part(26).y, landmarks.part(47).y+ 6)
                        #for nose
                        x_min_nose = min(landmarks.part(39).x,landmarks.part(42).x, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(33).x)
                        y_min_nose = min(landmarks.part(39).y,landmarks.part(42).y, landmarks.part(31).y, landmarks.part(35).y, landmarks.part(33).y)
                        x_max_nose = max(landmarks.part(39).x,landmarks.part(42).x, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(33).x)
                        y_max_nose = max(landmarks.part(39).y,landmarks.part(42).y, landmarks.part(31).y, landmarks.part(35).y, landmarks.part(33).y)
                        #for lips
                        x_min_lips = min(landmarks.part(48).x, landmarks.part(54).x, landmarks.part(50).x, landmarks.part(52).x, landmarks.part(57).x)
                        y_min_lips = min(landmarks.part(48).y, landmarks.part(54).y, landmarks.part(50).y, landmarks.part(52).y, landmarks.part(57).y)
                        x_max_lips = max(landmarks.part(48).x, landmarks.part(54).x, landmarks.part(50).x, landmarks.part(52).x, landmarks.part(57).x)
                        y_max_lips = max(landmarks.part(48).y, landmarks.part(54).y, landmarks.part(50).y, landmarks.part(52).y, landmarks.part(57).y)
                        
                        crop_img_eyes = image[y_min_face:y_max_face, x_min_face:x_max_face]
                        crop_img_nose = image[y_min_nose:y_max_nose, x_min_nose:x_max_nose]
                        crop_img_lips = image[y_min_lips:y_max_lips, x_min_lips:x_max_lips]

                        crop_img_eyes_copy = crop_img_eyes.copy()
                        crop_img_nose_copy = crop_img_nose.copy()
                        crop_img_lips_copy = crop_img_lips.copy()
                        
                        cv2.rectangle(image_save, (x_min_face, y_min_face), (x_max_face, y_max_face), (0, 255, 0), 2)
                        cv2.rectangle(image_save, (x_min_nose, y_min_nose), (x_max_nose, y_max_nose), (0, 255, 255), 2)
                        cv2.rectangle(image_save, (x_min_lips, y_min_lips), (x_max_lips, y_max_lips), (255, 255, 0), 2)

                        #for eyes
                        image = cv2.cvtColor(crop_img_eyes_copy, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = transform_train(image).unsqueeze(0).to(device)
                        #passing through the model
                        logits_face = model_eyes(image)[0]
                        logits_face = F.softmax(logits_face, dim=-1)


                        #for nose
                        image = cv2.cvtColor(crop_img_nose_copy, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = transform_train(image).unsqueeze(0).to(device)
                        #passing through the model
                        logits_nose = model_nose(image)[0]
                        logits_nose = F.softmax(logits_nose, dim=-1)

                        #for lips
                        image = cv2.cvtColor(crop_img_lips_copy, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = transform_train(image).unsqueeze(0).to(device)
                        #passing through the model
                        logits_face = model_lips(image)[0]
                        logits_face = F.softmax(logits_face, dim=-1)

                        #for eyes
                        if logits_face[0][0] > 0.5:
                                cv2.putText(image_save, 'autistic', (x_min_face, y_min_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'autistic eyes')
                        else:
                                cv2.putText(image_save, 'non-autistic', (x_min_face, y_min_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'non-autistic eyes')

                        #for nose
                        if logits_nose[0][0] > 0.5:
                                cv2.putText(image_save, 'autistic', (x_min_nose, y_min_nose - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'autistic nose')
                        else:
                                cv2.putText(image_save, 'non-autistic', (x_min_nose, y_min_nose - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'non-autistic nose')

                        #for lips
                        if logits_face[0][0] > 0.5:
                                cv2.putText(image_save, 'autistic', (x_min_lips, y_min_lips - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'autistic lips')
                        else:
                                cv2.putText(image_save, 'non-autistic', (x_min_lips, y_min_lips - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'non-autistic lips')

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image_save)







# for non-autistic image folder
input_folder = './Data/Faces/test/non_autistic'
output_folder = 'Results/non-autistic_output'

os.makedirs(output_folder, exist_ok=True)

# Loop through all the files in the input folder
for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
                # Construct the full path of the input image
                image_path = os.path.join(input_folder, filename)
                print(filename)
                image = cv2.imread(image_path)
                image_save = image.copy()
                faces = detector(image)
                for face in faces:
                        landmarks = predictor(image_save, face)
                        #for eyes
                        x_min_face = min(landmarks.part(17).x, landmarks.part(41).x + 6, landmarks.part(19).x, landmarks.part(24).x, landmarks.part(26).x, landmarks.part(47).x+ 6)
                        y_min_face = min(landmarks.part(17).y, landmarks.part(41).y + 6, landmarks.part(19).y, landmarks.part(24).y, landmarks.part(26).y, landmarks.part(47).y+ 6)
                        x_max_face = max(landmarks.part(17).x, landmarks.part(41).x + 6, landmarks.part(19).x, landmarks.part(24).x, landmarks.part(26).x, landmarks.part(47).x+ 6)
                        y_max_face = max(landmarks.part(17).y, landmarks.part(41).y + 6, landmarks.part(19).y, landmarks.part(24).y, landmarks.part(26).y, landmarks.part(47).y+ 6)
                        #for nose
                        x_min_nose = min(landmarks.part(39).x,landmarks.part(42).x, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(33).x)
                        y_min_nose = min(landmarks.part(39).y,landmarks.part(42).y, landmarks.part(31).y, landmarks.part(35).y, landmarks.part(33).y)
                        x_max_nose = max(landmarks.part(39).x,landmarks.part(42).x, landmarks.part(31).x, landmarks.part(35).x, landmarks.part(33).x)
                        y_max_nose = max(landmarks.part(39).y,landmarks.part(42).y, landmarks.part(31).y, landmarks.part(35).y, landmarks.part(33).y)
                        #for lips
                        x_min_lips = min(landmarks.part(48).x, landmarks.part(54).x, landmarks.part(50).x, landmarks.part(52).x, landmarks.part(57).x)
                        y_min_lips = min(landmarks.part(48).y, landmarks.part(54).y, landmarks.part(50).y, landmarks.part(52).y, landmarks.part(57).y)
                        x_max_lips = max(landmarks.part(48).x, landmarks.part(54).x, landmarks.part(50).x, landmarks.part(52).x, landmarks.part(57).x)
                        y_max_lips = max(landmarks.part(48).y, landmarks.part(54).y, landmarks.part(50).y, landmarks.part(52).y, landmarks.part(57).y)
                        
                        crop_img_eyes = image[y_min_face:y_max_face, x_min_face:x_max_face]
                        crop_img_nose = image[y_min_nose:y_max_nose, x_min_nose:x_max_nose]
                        crop_img_lips = image[y_min_lips:y_max_lips, x_min_lips:x_max_lips]

                        crop_img_eyes_copy = crop_img_eyes.copy()
                        crop_img_nose_copy = crop_img_nose.copy()
                        crop_img_lips_copy = crop_img_lips.copy()
                        
                        cv2.rectangle(image_save, (x_min_face, y_min_face), (x_max_face, y_max_face), (0, 255, 0), 2)
                        cv2.rectangle(image_save, (x_min_nose, y_min_nose), (x_max_nose, y_max_nose), (0, 255, 255), 2)
                        cv2.rectangle(image_save, (x_min_lips, y_min_lips), (x_max_lips, y_max_lips), (255, 255, 0), 2)

                        #for eyes
                        image = cv2.cvtColor(crop_img_eyes_copy, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = transform_train(image).unsqueeze(0).to(device)
                        #passing through the model
                        logits_face = model_eyes(image)[0]
                        logits_face = F.softmax(logits_face, dim=-1)


                        #for nose
                        image = cv2.cvtColor(crop_img_nose_copy, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = transform_train(image).unsqueeze(0).to(device)
                        #passing through the model
                        logits_nose = model_nose(image)[0]
                        logits_nose = F.softmax(logits_nose, dim=-1)

                        #for lips
                        image = cv2.cvtColor(crop_img_lips_copy, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(image)
                        image = transform_train(image).unsqueeze(0).to(device)
                        #passing through the model
                        logits_face = model_lips(image)[0]
                        logits_face = F.softmax(logits_face, dim=-1)

                        #for eyes
                        if logits_face[0][0] > 0.5:
                                cv2.putText(image_save, 'autistic', (x_min_face, y_min_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'autistic eyes')
                        else:
                                cv2.putText(image_save, 'non-autistic', (x_min_face, y_min_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'non-autistic eyes')

                        #for nose
                        if logits_nose[0][0] > 0.5:
                                cv2.putText(image_save, 'autistic', (x_min_nose, y_min_nose - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'autistic nose')
                        else:
                                cv2.putText(image_save, 'non-autistic', (x_min_nose, y_min_nose - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'non-autistic nose')

                        #for lips
                        if logits_face[0][0] > 0.5:
                                cv2.putText(image_save, 'autistic', (x_min_lips, y_min_lips - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'autistic lips')
                        else:
                                cv2.putText(image_save, 'non-autistic', (x_min_lips, y_min_lips - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                print(filename+ 'non-autistic lips')

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, image_save)
