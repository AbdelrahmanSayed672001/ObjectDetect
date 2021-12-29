import cv2

def webcamdetection():
    cap=cv2.VideoCapture(0)
    className=[]
    classFile='cocoNames.txt'
    
    with open(classFile,'r') as f:
        className=f.read().rstrip("\n").split("\n")
    
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath='frozen_inference_graph.pb'
    
    net=cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)
    
    while True:
        success,img=cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)
    
        if len(classIds) !=0:
            for classID, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                cv2.putText(img, className[classID - 1], (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255),2)
    
        cv2.imshow("web cam detection output", img)
        cv2.waitKey(1)


def main():
    webcamdetection()


if __name__ == "__main__":
    main()