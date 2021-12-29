from tkinter.font import BOLD
import PySimpleGUI as sg
import cv2
import ImageDetectionwithGUI as img 
import VideoDetectionwithGUI as vid
import WebCamDetection as webcam


def main():

    layout = [
        [

            [sg.Text("Object Detection", size=(200, 1) ,font=("normal",20,BOLD), justification="center",text_color='black',background_color='#dde3e7')],
            
            [sg.Button("Image Detection",size=(40,2),font=("normal",12,BOLD),pad=(135, 20),button_color=('white', '#1A10A5')),],
            
            [sg.Button("Video Detection",size=(40,2),font=("normal",12,BOLD),pad=(135, 20),button_color=('white', '#1A10A5'))],
            
            [sg.Button("Webcam Detection",size=(40,2),font=("normal",12,BOLD),pad=(135, 20),button_color=('white', '#1A10A5')),],
           
        ],
        
    ]

    window1 = sg.Window("Object Detection", layout,size=(650,330),background_color='#dde3e7')

   
    while True:
        event, values = window1.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == "Webcam Detection":
            webcam.main() 
            
        if event == "Image Detection":
            img.main() 
            
            
        if event == "Video Detection":
            vid.main()


    window1.close()


if __name__ == "__main__":
    main()
