import pandas as pd
import glob
import cv2
from pose import Pose, semaphore_numbers
import time
import random
import numpy as np
from math import cos, sin, tan
from pose import semaphore_letters
from matplotlib import pyplot as plt
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
    pose1 = Pose(complexity=1)# for the sake of memory you want a single pose
                              # object per array shape
    pose2 = Pose(complexity=1)
    df = pd.DataFrame(columns=range(38),index=range(count))
    labels = pd.DataFrame(columns=range(27), index=range(count))
    ct = 0
    t1 = time.time()
    for j,f in enumerate(filepaths):
        label = [0]*27
        f_rev = f[::-1]
        c = f_rev[f_rev.find("\\")-1]
        idx = semaphore_numbers[c]
        label[idx] = 1
        frame = cv2.imread(f)
        if frame.shape == (480,844,3):
            pose1.compute_frame(frame)
            df.loc[ct] = pose1.return_keypoint_vector()[1]
        else:
            pose2.compute_frame(frame)
            df.loc[ct] = pose2.return_keypoint_vector()[1]
        labels.loc[ct] = label
        ct += 1
        if (j+1) % 20 == 0:
            print("[{}/{}] images processed in {:.1f}s".format(j+1,count,(time.time()-t1)))
            t1 = time.time()
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

def handlabelled():
    path = "data/hand_labelled"
    df, labels = return_real_data_hand_labelled(path)
    data_path = "data_labelled.csv"
    label_path = "labels_labelled.csv"
    save_data(data_path, label_path, df, labels)

def first_dataset():
    data_file = "data.csv"
    label_file = "labels.csv"
    data_path = 'data.csv'
    label_path = 'labels.csv'

    save_data(data_path, label_path, df, labels)

    count = get_training_size_first_dataset("data")
    df, labels = return_real_data_first_dataset(data_file, count)

def load_data(data_file : str, label_file : str):
    """ 
    this loads the data and labels into pd dataframes 
    """
    data = pd.read_csv(data_file)
    labels = pd.read_csv(label_file)
    return data, labels

def data_to_vectors(d):
    v = []
    for i in range(19):
        x = d[i]
        y = d[i+1]
        v.append([x,y])
    return v

def vectors_to_data(v):
    d = []
    for v2 in v:
        d.append(v2[0])
        d.append(v2[1])
    return d

def roatate_array(arr, theta : float):
    vs = data_to_vectors(arr)
    theta = np.deg2rad(theta)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    for i, v in enumerate(vs):
        v_rot = np.dot(rot,v)
        vs[i] = v_rot
    return vectors_to_data(vs)

def shear_array(arr, theta : float):
    vs = data_to_vectors(arr)
    theta = np.deg2rad(theta)
    rot = np.array([[1, -tan(theta/2)], [0, 1]])
    for i, v in enumerate(vs):
        v_rot = np.dot(rot,v)
        vs[i] = v_rot
    return vectors_to_data(vs)

def generate_synthetic_data(data,labels,number_to_produce):

    real_count = data.shape[0]
    d_syn = pd.DataFrame(columns=range(38),index=range(number_to_produce))
    l_syn = pd.DataFrame(columns=range(27), index=range(number_to_produce))
    for n in range(number_to_produce):
        idx = random.randint(0,real_count-1)
        l_syn.loc[n] = np.array(labels.loc[idx])
        d = np.array(data.loc[idx])
        theta = random.uniform(-30,30)
        if n % 2 == 0:
            d_gen = roatate_array(d,theta)
        else:
            d_gen = shear_array(d,theta)
        
        d_syn.loc[n] = d_gen
    return d_syn, l_syn

def plot_random_data(data,labels):
    count = data.shape[0]
    idx = random.randint(0,count-1)

    d = data.loc[idx]
    l = np.argmax(labels.loc[idx])
    print(semaphore_letters[l])

    plt.figure()
    vs = data_to_vectors(d)
    for v in vs:
        plt.plot(v[1],v[0],'+')
    plt.show()

def main():

    generate_synthetic = True
    compute_first_dataset = False
    compute_hand_labelled = False



    if compute_hand_labelled:
        handlabelled()
    if compute_first_dataset:
        first_dataset()

    if generate_synthetic:
        data_file = "data.csv"
        label_file = "labels.csv"
        data_file_2 = "data_labelled.csv"
        label_file_2 = "labels_labelled.csv"

        num_produce = 1000

        df1, labels1 = load_data(data_file, label_file)
        df2, labels2 = load_data(data_file_2, label_file_2)

        df = pd.concat([df1,df2],axis=0, ignore_index=True, sort=False)
        labels = pd.concat([labels1,labels2], ignore_index=True, sort=False)

        df.drop(columns=df.columns[0], axis=1, inplace=True)
        labels.drop(columns=labels.columns[0], axis=1, inplace=True)
    
        print("The combined real data array has size {}".format(df.shape))

        d_syn, l_syn = generate_synthetic_data(df, labels, num_produce)

#        plot_random_data(df, labels)

        save_data("d_syn.csv","l_syn.csv",d_syn,l_syn)


if __name__ == "__main__":
    main()