import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image
import os
import speech_recognition as sr
from PIL import Image
import time
import keras

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_draw = mp.solutions.drawing_utils



mp_holistic = mp.solutions.holistic  # HOLISTICS MODEL
mp_drawing = mp.solutions.drawing_utils # DRAWING UTILITIES

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR to RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make Prediction
    image.flags.writeable = True    # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB to BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Face Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Pose Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Left Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Right Hand Connections
    
def draw_styled_landmarks(image, results):
    # Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color =(80,110,10), thickness = 1, circle_radius = 1), # Dot Color
                             mp_drawing.DrawingSpec(color =(80,256,121), thickness = 1, circle_radius = 1) # Line Color
                             )
                              
   # Pose Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color =(80,22,10), thickness = 2, circle_radius = 4), # Dot Color
                             mp_drawing.DrawingSpec(color =(80,44,121), thickness =2, circle_radius = 2) # Line Color
                             )  
    # Left Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color =(121,22,76), thickness = 2, circle_radius = 4), # Dot Color
                             mp_drawing.DrawingSpec(color =(121,44,250), thickness = 2, circle_radius =2) # Line Color
                             ) 
     # Right Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color =(245,117,66), thickness = 2, circle_radius = 4), # Dot Color
                             mp_drawing.DrawingSpec(color =(245,66,230), thickness = 2, circle_radius = 2) # Line Color
                             ) 
   

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face  = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

actions = np.array(['Hello', 'Thanks', 'I Love You'])





DEMO_VIDEO = 'Studio_Project_V1.mp4'
DEMO_IMAGE = 'demo.jpg'

my_list = []




st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Sign Language Detection - SignO'fy")
st.sidebar.subheader('ASL')

# @st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)


    else:

        r = width / float(w)
        dim = (width, int(h * r))


    resized = cv2.resize(image, dim, interpolation=inter)


    return resized

app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Sign Language to Text','Speech to sign Language']
)

if app_mode =='About App':
    st.title("SignO'fy")
    st.markdown("Welcome to our website! We are a passionate and dedicated team of five freshers, driven by a shared love for technology and machine learning. Our goal is to harness the power of artificial intelligence and bring innovative solutions to real-world challenges. With a strong background in web development and a deep understanding of machine learning algorithms, we have combined our skills to create this platform. Our mission is to deliver intelligent and efficient solutions that make a positive impact on various industries. We believe in constant learning, collaboration, and pushing the boundaries of what's possible. Join us on this exciting journey as we explore the limitless potential of machine learning and its applications in the digital age.")
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.video(DEMO_VIDEO)

    st.title("Developers")
    st.markdown('Adwait Mali\n')
    st.markdown('Ankur Musmade\n')
    st.markdown('Rohit Pimple\n')
    st.markdown('Moin Palekar\n')
    st.markdown('Rohan Patil\n')

    
                 
                 
                
    

    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    
elif app_mode == 'Sign Language to Text':
    st.title('Sign Language to Text')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button('Go Live')
    stop = st.sidebar.button('Stop Video')
    record = st.sidebar.checkbox("Record Video")
    if record:
        st.checkbox("Recording", value=True)

    st.sidebar.markdown('---')
    sameer=""
    st.markdown(' ## Output')
    st.markdown(sameer)

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        elif stop:
            vid = cv2.VideoCapture(0)
            vid.release()
            cv2.destroyAllWindows()
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO

    else:
        tfflie.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tfflie.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    codec = cv2.VideoWriter_fourcc('V', 'P', '0', '9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.sidebar.markdown('---')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# finger_tips = [8, 12, 16, 20]
    # thumb_tip = 4
    
# 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    model = keras.models.load_model('Final3.h5')

    #cap = cv2.VideoCapture(0)
# Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while vid.isOpened() and stop == False:

        # Read feed
            ret, frame = vid.read()
        #time.sleep(3)

        # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
        
        # Draw landmarks
            draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            
                if res[np.argmax(res)] > threshold: 
                  
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 1: 
                    sentence = sentence[-1:]


            
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            frame = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            frame = image_resize(image=frame, width=640)
            stframe.image(frame, channels='BGR', use_column_width=True)
        # Show to screen
            #cv2.imshow('OpenCV Feed', image)

        # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    vid.release()
    cv2.destroyAllWindows()

    st.text('Video Processed')

    output_video = open('Studio_Project_V1.mp4', 'rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out.release()










else:
    st.title('Speech to Sign Language (The System use Indian Sign Language)')
    # initialize the speech recognition engine
    # initialize the speech recognition engine
    r = sr.Recognizer()


    # define function to display sign language images
    def display_images(text):
        # get the file path of the images directory
        img_dir = "images/"

        # initialize variable to track image position
        image_pos = st.empty()

        # iterate through the text and display sign language images
        for char in text:
            if char.isalpha():
                # display sign language image for the alphabet
                img_path = os.path.join(img_dir, f"{char}.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()
            elif char == ' ':
                # display space image for space character
                img_path = os.path.join(img_dir, "space.png")
                img = Image.open(img_path)

                # update the position of the image
                image_pos.image(img, width=300)

                # wait for 2 seconds before displaying the next image
                time.sleep(2)

                # remove the image
                image_pos.empty()

        # wait for 2 seconds before removing the last image
        time.sleep(2)
        image_pos.empty()


    # add start button to start recording audio
    if st.button("Start Talking"):
        # record audio for 5 seconds
        with sr.Microphone() as source:
            st.write("Say something!")
            audio = r.listen(source, phrase_time_limit=5)

            try:
                text = r.recognize_google(audio)
            except sr.UnknownValueError:
                st.write("Sorry, I did not understand what you said.")
            except sr.RequestError as e:
                st.write(f"Could not request results from Google Speech Recognition service; {e}")

        # convert text to lowercase
        text = text.lower()
        # display the final result
        st.write(f"You said: {text}", font_size=41)

        # display sign language images
        display_images(text)

