import cv2
from os import listdir

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
    

def main():

    out_folder = "data/hand_labelled/"
    video_path = "data/navy_morse_video.mp4"

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

    