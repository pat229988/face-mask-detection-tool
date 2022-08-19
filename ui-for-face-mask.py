import cv2
import PySimpleGUI as sg
import os
import PySimpleGUI as sg
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
model=load_model("./mask_detector_with_cudnn_3_layers_32+epoch.model")
haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}
rect_size = 4
cap = cv2.VideoCapture(0)

global recording
recording = False 

sg.theme('DarkAmber')

def landing_page():
    layout_landing = [
        [sg.Text('Welcome !!')],
        [sg.Text('Description: ')],
        [sg.Text('this tool can help you to detect the mask in real time using your web cam or a image from your gallery!!')],
        [sg.Text('select one option to continue: ')],
        [sg.Button('Live',key='-LIVE-'),sg.Button('Image',key='-IMAGE-'),sg.Button('Quit',key='-EXIT-')]
    ]
    window = sg.Window('Face Mask Detection Tool', layout_landing , resizable=True,finalize=True)
    return window

def load_image(path, window):
    try:
        image = Image.open(path)
        image.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(image)
        window["image"].update(data=photo_img)
    except:
        print(f"Unable to open {path}!")

def image_page():
    left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')]]

    # For now will only show the name of the file that was chosen
    images_col = [[sg.Text('You choose from the list:'),sg.Text(size=(40,1), key='-TOUT-')],
                [sg.Image(key='-IMAGE-file')]]

    # ----- Full layout -----
    layout_for_image = [
        [sg.Text('Welcome !!')],
        [sg.Text('Description: ')],
        [sg.Text('this tool can help you to detect wether the person has wear a mask or not from the image!!')],
        [sg.Column(left_col, element_justification='c'), sg.VSeperator(),sg.Column(images_col, element_justification='c')],
        [sg.Button('Back',key='-BACK-'),sg.Button('Quit',key='-EXIT-')]
        ]

    # --------------------------------- Create Window ---------------------------------
    window_for_image = sg.Window('Multiple Format Image Viewer', layout_for_image, resizable=True,finalize=True)
    return window_for_image

def get_img_data(f, maxsize=(700, 650), first=False):
    #Generate image data using PIL
    # img = Image.open(f)
    # img.thumbnail(maxsize)
    im = cv2.imread(f)
    im = cv2.resize(im, maxsize, interpolation = cv2.INTER_AREA)
    im=cv2.flip(im,1,1)
    rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        
        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(150,150))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
    
        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
    imgbytes = cv2.imencode('.png', im)[1].tobytes()

    return (imgbytes)

def webcam_page():
    layout_webcam = [[sg.Text('Welcome !!')],
        [sg.Text('Description: ')],
        [sg.Text('this tool can help you to detect wether the person has wear a mask or not from yourd device webcam feed!!')],
        [sg.Image(filename='',key='-IMAGE-live-')],
        [sg.Button('Back',key='-BACK-'),sg.Button('Quit',key='-EXIT-')]
        ]

    window = sg.Window('Face Mask Detection Tool', layout_webcam,resizable=True,finalize=True)
    return window

def main():
    window0 = landing_page()
    window = window0
    global recording
    while True:
        event, values = window.read(timeout=20)
        if recording:
            # print('.')
            ret, im = cap.read()
            im=cv2.flip(im,1,1)
            rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
            faces = haarcascade.detectMultiScale(rerect_size)
            for f in faces:
                (x, y, w, h) = [v * rect_size for v in f] 
                
                face_img = im[y:y+h, x:x+w]
                rerect_sized=cv2.resize(face_img,(150,150))
                normalized=rerect_sized/255.0
                reshaped=np.reshape(normalized,(1,150,150,3))
                reshaped = np.vstack([reshaped])
                result=model.predict(reshaped)

                label=np.argmax(result,axis=1)[0]
            
                cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
                cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
            imgbytes = cv2.imencode('.png', im)[1].tobytes()
            window3['-IMAGE-live-'].update(data=imgbytes)
            window['-IMAGE-live-'].update(data=imgbytes)
        
        if event == "-EXIT-" or event == sg.WIN_CLOSED:
            break
        elif event == "-IMAGE-":
            window.hide()
            window1 = image_page()
            window = window1 if window == window else window
            window.un_hide()
        elif event == "-FOLDER-": 
            folder = values["-FOLDER-"]
            try: 
                # Get list of files in folder 
                file_list = os.listdir(folder) 
                # file_list = [] 
                fnames = [ f for f in file_list if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".png",".jpeg",".jpg", ".gif")) ] 
                window["-FILE LIST-"].update(fnames)
            except: 
                pass
        elif event == "-FILE LIST-": 
            # A file was chosen from the listbox 
            try: 
                filename = os.path.join( values["-FOLDER-"], values["-FILE LIST-"][0]) 
                window["-TOUT-"].update(filename)
                # image_elem.update(data=get_img_data(filename, first=True))
                # window["-IMAGE-file"].update(filename=filename)
                window["-IMAGE-file"].update(data=get_img_data(filename, first=True))
            except: 
                pass 
        elif event == "-LIVE-":
            cap = cv2.VideoCapture(0)
            recording = True
            window.hide()
            window3 = webcam_page()
            window = window3
            
        elif event == '-BACK-':
            if recording:
                cap.release()
            recording = False
            print('going back')
            window.hide()
            window = window0
            window.un_hide()
        else:
            pass
    window.close()

if __name__ == "__main__":
    main()
