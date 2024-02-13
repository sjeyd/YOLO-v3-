import cv2
import numpy as np
import os
def run_object_detection(image_path, config_path, weights_path, classes_path, output_path, count):
    Labels = []
    All_boxes = []
    global_bbox=[]

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        Labels.append(label)

    image = cv2.imread(image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_path, config_path)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id]:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                #print(boxes)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    #print(indices)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        All_boxes.append(box)

    items = [(label, values) for label, values in zip(Labels, All_boxes)]
    
    X_min=9999999
    Y_min=9999999
    X_max=0
    Y_max=0
    for i in range(len(All_boxes)):
        #print(All_boxes[i])
        x=round(All_boxes[i][0])
        y=round(All_boxes[i][1])
        w=round(All_boxes[i][2])
        h=round(All_boxes[i][3])
        xx=x-w/2
        yy=y-h/2
        xm=x+w/2
        ym=y+h/2
        X_min=min(X_min,x)
        if X_min <= 8:
            X_min=8 
            
        Y_min=min(Y_min,y)
        if Y_min <= 8:
            Y_min=8
            
        X_max=max(X_max,x+w)
        if X_max >= Width-8:
            X_max=Width-8
            
        Y_max=max(Y_max,y+h)
        if Y_max >= Height-8:
            Y_max=Height-8
       
        print(X_min,Y_min,X_max,Y_max)
        
    global_bbox=[X_min,Y_min,X_max,Y_max]
    #print(global_bbox)
    data_dict = {'items': global_bbox}
    #print(data_dict)

    # Generate a dynamic file name based on the input image
    image_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_path, f"object-detection_{count}_{os.path.splitext(image_filename)[0]}.jpg")
    
    #cv2.imwrite(output_image_path, image)

    # Save the dictionary in a .mat file
    cv2.imshow("object detection", image)
    cv2.waitKey()
    cv2.imwrite(output_image_path, image)
    cv2.destroyAllWindows()
    return global_bbox
    #return data_dict
