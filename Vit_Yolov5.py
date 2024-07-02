from keras import Input, Model
from keras.src.layers import Conv2D, BatchNormalization
import cv2 as cv
from vit_pytorch import vit
def create_vit_backbone(input_shape=(224, 224, 3)):
    vit_model = vit.vit_b16(
        image_size=input_shape[0],
        activation='softmax',  # Change this to 'linear' if you want to use YOLOv5 head
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )
    return vit_model

# Define YOLOv5 detection head
def create_yolov5_head(input_tensor):
    x = Conv2D(256, 3, padding='same', activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    yolov5_output = Conv2D(2, 3, padding='same', activation='sigmoid')(x)
    return yolov5_output

# Create the complete YOLOv5 model
def create_yolov5_model(input_shape=(224, 224, 3)):
    backbone = create_vit_backbone(input_shape)
    backbone.trainable = False  # Freeze Vision Transformer backbone
    input_tensor = Input(shape=input_shape)
    features = backbone(input_tensor)
    yolov5_output = create_yolov5_head(features)
    model = Model(inputs=input_tensor, outputs=yolov5_output)
    return model

def Vit_Yolov5(image):
    with open('Classes.txt', 'r') as f:
        class_labels = f.read().splitlines()
    yolov5_model = create_yolov5_model()
    yolov5_model.summary()
    results = yolov5_model(image)
    # Extract the bounding boxes, labels, and scores
    boxes = results.xyxy[0].numpy()
    class_ids = results.xyxy[0][:, 5].numpy().astype(int)
    confidences = results.xyxy[0][:, 4].numpy()
    # Set the confidence threshold
    confidence_threshold = 0.5
    # Iterate over the detected objects
    for i in range(len(boxes)):
        if confidences[i] > confidence_threshold:
            x1, y1, x2, y2 = boxes[i]
            label = class_labels[class_ids[i]]
            confidence = confidences[i]
            # Draw the bounding box and label
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f'{label}: {confidence:.2f}'
            cv.putText(image, text, (int(x1), int(y1) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

