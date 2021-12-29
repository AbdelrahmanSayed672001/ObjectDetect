
import os
from tkinter.font import BOLD
import PySimpleGUI as sg
import cv2


# Algorithm to detect Image
def detectImage(filename):
    
    im=cv2.imread(filename)
    img=cv2.resize(im,(500,500))
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

    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds,bbox)

    if len(classIds)!=0:
        for classID,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=3)
            cv2.putText(img,className[classID-1],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2  )


    cv2.imshow("Image detection output",img)
    cv2.waitKey(0)


# array of file type
file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

def main():
    layout = [
        [
            sg.Text("Image File",font=("normal",15,BOLD),pad=15,text_color='black',background_color='#dde3e7'),
            sg.Input(size=(35, 2), key="-FILE-"),
            sg.FileBrowse(file_types=file_types,size=(9,1),font=("normal",12,BOLD),pad=15,button_color=('white', '#1A10A5')),
            sg.Button("Detect Image",size=(12,1),font=("normal",12,BOLD),pad=15,button_color=('white', '#1A10A5')),
        ],
        

    ]

    window = sg.Window("Image Viewer", layout,size=(700,100),background_color='#dde3e7')

    emptyLayout = [
        [sg.Text("The image file is empty !!\n Please choose the image",
                size = (30,5),font=("normal",15,BOLD),justification='center',text_color='red',background_color='#dde3e7')]
    ]

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Detect Image":
            
            filename = values["-FILE-"]

            if(filename == ""):
                sg.Window("ERROR" , emptyLayout,background_color='#dde3e7').read()

            if os.path.exists(filename):
                detectImage(filename)


    window.close()
    

if __name__ == "__main__":
    main()


