import pandas as pd
import glob
import cv2
from pose import Pose, semaphore_numbers

"""
This file loads the given images, and extracts the keypoints using the Pose class
These and their labels are saved as csvs for saving or testing
"""



def get_training_size_first_dataset(path : str):
    """
    get training data size
    we have 38 (=2x17) of real data, and a 27 long one hot encoding (space key is to be added later)
    """
    count = 0
    for i in range(1,27):
        filepaths = glob.glob("{}/{}/*.png".format(path,str(i)))
        for f in filepaths:
            count += 1
    print("There are {} datapoints".format(count))
    return count

def return_real_data_first_dataset(path : str, count : int):
    """
    extract pose data from training images
    NB: labels 0 column is a space, and the real data has column indices 1 through 26
    """
    pose = Pose()
    df = pd.DataFrame(columns=range(38),index=range(count))
    labels = pd.DataFrame(columns=range(27), index=range(count))
    ct = 0
    for i in range(1,27):
        filepaths = glob.glob("{}/{}/*.png".format(path,str(i)))
        label = [0]*27
        label[i] = 1
        for j,f in enumerate(filepaths):
            frame = cv2.imread(f)
            pose.compute_frame(frame)
            df.loc[ct] = pose.return_keypoint_vector()[1]
            labels.loc[ct] = label
            ct += 1
    return df, labels
            # print(pose.return_keypoint_vector())
            # pose.show_pose_landmarks()

def return_real_data_hand_labelled(path : str):
    """
    extract pose data from training images
    NB: labels 0 column is a space, and the real data has column indices 1 through 26
    this function is for hand labelled images where the first character is the label

    """
    filepaths = glob.glob("{}/*.png".format(path))
    count = len(filepaths)
    pose = Pose()
    df = pd.DataFrame(columns=range(38),index=range(count))
    labels = pd.DataFrame(columns=range(27), index=range(count))
    ct = 0

    for j,f in enumerate(filepaths):
        label = [0]*27
        f_rev = f[::-1]
        c = f_rev[f_rev.find("\\")-1]
        idx = semaphore_numbers[c]
        label[idx] = 1
        frame = cv2.imread(f)
        pose.compute_frame(frame)
        df.loc[ct] = pose.return_keypoint_vector()[1]
        labels.loc[ct] = label
        ct += 1
    return df, labels

def load_data(data_file : str, label_file : str):
    """ 
    this loads the data and labels into tensors 
    note that the data is cast to float32 before being converted to a tensor
    """
    data = pd.read_csv(data_file)
    labels = pd.read_csv(label_file)
    data.drop(columns=data.columns[0], axis=1, inplace=True)
    labels.drop(columns=labels.columns[0], axis=1, inplace=True)
    return data, labels

def save_data(data_file : str, label_file : str, df, labels):
    df.to_csv(data_file)
    labels.to_csv(label_file)

    print("Arrays saved to file")

    



load_from_file = False
generate_synthetic = False
first_dataset = False
hand_labelled = True

if hand_labelled:
    path = "data/hand_labelled"
    df, labels = return_real_data_hand_labelled(path)
    data_path = "data_labelled.csv"
    label_path = "labels_labelled.csv"
    save_data(data_path, label_path, df, labels)

if first_dataset:

    data_file = "data.csv"
    label_file = "labels.csv"


    if load_from_file:
        df, labels = load_data(data_file, label_file)
    else:
        count = get_training_size_first_dataset("data")
        df, labels = return_real_data_first_dataset(data_file, count)



    ########## saving data ########
    if not generate_synthetic:
        data_path = 'data.csv'
        label_path = 'labels.csv'

        save_data(data_path, label_path, df, labels)