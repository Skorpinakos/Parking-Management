import cv2
import os
import numpy as np
from clustering_utils import *
from mqtt_publisher import *


######################################## YOLO STUFF
def build_model(is_cuda):
    net = cv2.dnn.readNet("edgeAI/resources/yolov5s.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
is_cuda = False
net = build_model(is_cuda)

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds



def load_classes():
    class_list = []
    with open("edgeAI/resources/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    
    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        
        if confidence >= 0.05:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
                

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
    centers=[]
    for box in result_boxes:
        centers.append((int(box[0]+box[2]/2),int(box[1]+box[3]/2)))
        
    return result_class_ids, result_confidences, result_boxes, centers

def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result
############################################

def closest_node(node, nodes):
    dist_2=[]
    for point in nodes:
        dist_2.append((node[0]-point[0])**2+(node[1]-point[1])**2)
    print('diff list',dist_2)
    print('minimum',min(dist_2))
    return dist_2.index(min(dist_2))

def evaluate_parking_spot(centers_occupied):
    old_centers=[]
    with open("edgeAI/results/points.csv",'r',encoding='utf-8') as file:
        for line in file.read().split('\n')[:-1]:
            old_centers.append((int(line.split(",")[0]),int(line.split(",")[1])))
    old_centers.extend(centers_occupied)
    print('current occupants',centers_occupied)
    occupied_parking_spots=[]
    unoccupied_parking_spots=[]
    if len(old_centers)>=10:
        test_space=range(2,8) #set possible parking spots multitude
        wcss=check_cluster_multitudes(test_space,old_centers) #check error for possible multitudes
        #plot_cluster_graph(test_space,wcss) #plot it
        n=find_elbow(wcss,test_space) #find the multitude on the elbow , how many parking spots there are
        #print(n)
        xs,ys,_=find_centroids(old_centers,n) #find centers for given multitude
        xs = [int(x) for x in xs] #make their coords ints
        ys = [int(y) for y in ys]
        parking_spots=[]
        for position in range(len(xs)): #use the centers to make a parking spot list
            parking_spots.append((ys[position],xs[position]))
        
        parking_spots=sorted(parking_spots, key=lambda k: [k[1], k[0]]) #sort on a reading-line-type priority
        print('parking spots',parking_spots)
        
        for occupant in centers_occupied:
            occupied_parking_spot=closest_node(occupant,parking_spots)
            occupied_parking_spots.append(occupied_parking_spot)
        unoccupied_parking_spots=list(range(0,n))
        unoccupied_parking_spots=list(set(unoccupied_parking_spots)-set(occupied_parking_spots))
        print('indexes',occupied_parking_spots)

    return occupied_parking_spots,unoccupied_parking_spots
        



        

    
        




#### MAIN


#client=connect_mqtt()

folder_dir = "edgeAI/dataset"
images_names=os.listdir(folder_dir) #get dir list of filenames
for image_name in images_names:
    # check if the image_name ends with PNG just to be sure
    if (image_name.endswith(".PNG")):
        #print(image_name)
        frame = cv2.imread(folder_dir+'/'+image_name)

    ### use model, we want the "boxes" and "centers"
    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)
    class_ids, confidences, boxes, centers = wrap_detection(inputImage, outs[0])

    # using older objects centers find parking spots and evaluate if they are occupied or not based on the current object centers
    occupied_parking_spots,unoccupied_parking_spots=evaluate_parking_spot(centers)
    #for spot in occupied_parking_spots:
        #publish(client,"/iot/workshop/team00/parking/{}".format(spot),1)
    #for spot in unoccupied_parking_spots:
        #publish(client,"/iot/workshop/team00/parking/{}".format(spot),0)

    for center in centers:
        cv2.circle(frame,center,20,(0,255,0),3) #draw circle at objects centers
        with open('edgeAI/results/points.csv','a',encoding='utf-8') as file:
            file.write("{},{}\n".format(center[0],center[1])) #save objects center coords



    for (classid, confidence, box) in zip(class_ids, confidences, boxes): #draw rectangle around objects
            color = (255,255,0)
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, "car", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

    

    cv2.imshow("output", frame) #show image
    cv2.waitKey(0) #wait 2 seconds
    


