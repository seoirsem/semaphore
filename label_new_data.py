import cv2
from os import listdir
from pose import Pose
from process_poses import return_processed_frame
import os
import glob
from train_network import Model
import torch

"""
This is a simple helper function to allow me to extract
still images from videos and hand label the letter

HOW TO USE:
- Enter the video to load from, and the path to save images
- Step through the frames with any character. If the character
  is in the dict "semaphore_letters" that frame gets saved
  with the filename "*i" where i is an increment
  - for example, if there are files A1.png, and A2.png
    if you press "a" to save the file, it goes to A3
- To skip frames with no semaphore, hit e.g. any arrow key
- The program will end when the frames are exhausted

TO DELETE FRAMES:
- run "delete_files"
- arrow keys to scroll through the frames
- "d" to delete the current frame
- be careful with this as pose data carries frame to frame
  and is meant to be a video
"""

# dictionary of relevant character encodings:
semaphore_letters = {
    32: "_", # we call space underscore so the filename makes sense
    97: "A",
    98: "B",
    99: "C",
    100: "D",
    101: "E",
    102: "F",
    103: "G",
    104: "H",
    105: "I",
    106: "J",
    107: "K",
    108: "L",
    109: "M",
    110: "N",
    111: "O",
    112: "P",
    113: "Q",
    114: "R",
    115: "S",
    116: "T",
    117: "U",
    118: "V",
    119: "W",
    120: "X",
    121: "Y",
    122: "Z",
    27: "esc"
}

def save_frame(frame, label : str, folder : str):
    files = listdir(folder)
    i = 1
    while True:
        if (label + str(i) + '.png') in files:
            i += 1
        else:
            name = folder + label + str(i) + '.png'
            cv2.imwrite(name,frame)
            print("Saved file as '{}'".format(name))
            break
    
def delete_files(out_folder):
    """
    you shouldnt use this!
    the pose data carries from frame to frame and so is inconsistent
    """
    filepaths = glob.glob("{}/*.png".format(out_folder))
    pose1 = Pose(complexity=2)
    pose2 = Pose(complexity=2)
    i = 0
    model_path = "train_semaphore.pt"
    model = Model(38, 60, 27)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + model_path)
        model.eval()
    else:
        ValueError("No classifier model was found at {}".format(model_path))

    while i < len(filepaths):
        f = filepaths[i]
        print(f)
        frame = cv2.imread(f)
        if frame is None:
            pass
        else:
            if frame.shape == (480,844,3):
                frame1, pose = return_processed_frame(frame, pose1, True, False, False, model)
                frame2, pose = return_processed_frame(frame, pose1, False, True, True, model)
            else:
                frame1, pose = return_processed_frame(frame, pose2, True, False, False, model)
                frame2, pose = return_processed_frame(frame, pose2, False, True, True, model)
            frame1 = cv2.resize(frame1,(int(844/1.5),int(480/1.5)))
            frame2 = cv2.resize(frame2,(int(844/1.5),int(480/1.5)))
            cv2.imshow("pose landmarks", frame1)
            cv2.imshow("pose landmarks2", frame2)
        out = cv2.waitKey(0)
        if out == 27:
            break
        elif out == 49:
            i += 1 #"1"
        elif out == 50:
            i -= 1 #"2"
            if i<0:
                i = 0
        elif out == 100 and frame is not None: # "d"
            os.remove(f)
            print("deleted " + f)
        else:
            continue
            


def main():
    read = True
    out_folder = "data/hand_labelled/"

    # if read == False:
    #     print("here")
    #     delete_files(out_folder)
    # else:
    print("there")
    video_path = "data/navy_morse_video.mp4"
    video_path = "data/alphabet_room.mp4"

    camera = cv2.VideoCapture(video_path)
    wait = 0
    while camera.isOpened():
        success, frame = camera.read()


        if success:

            cv2.imshow("feed",frame)        
            out = cv2.waitKey(0)
            if out == 27:
                break
            elif out in semaphore_letters:
                save_frame(frame,semaphore_letters[out],out_folder)

        else:
            break


if __name__ == "__main__":
    main()

    