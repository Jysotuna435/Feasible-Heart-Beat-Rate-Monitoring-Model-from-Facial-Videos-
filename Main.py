import numpy as np
import os
import cv2
from numpy import matlib
from BMO import BMO
from EOO import EOO
from FER import process_video
from Heart_Rate import extract_heart_rate
from Model_CNN import Model_CNN
from Model_DBN import Model_DBN
from Model_RAN3d import Model_RAN3d
from Model_RESNET import Model_RESNET
from Model_RNN import Model_RNN
from Model_W_ALSTM_AM import Model_W_ALSTM_AM
from NGO import NGO
from Objective_Function import Objfun_Cls
from PROPOSED import PROPOSED
from Plot_results import plot_conv, plot_Error_results, plot_Segment_results, plot_Error_results1
from SGO import SGO
from Vit_Yolov5 import Vit_Yolov5

no_of_dataset = 2

# Read the dataset 1
an = 0
if an == 1:
    Dataset = './Dataset/DATASET_1'
    HR = []
    Images = []
    Target = []
    count = 0
    dataset_no_path = os.listdir(Dataset)
    for i in range(len(dataset_no_path)):
        data_folder = Dataset + '/' + dataset_no_path[i]
        if os.path.isdir(data_folder):
            data_fold_path = os.listdir(data_folder)
            for j in range(len(data_fold_path)):
                data_file = data_folder + '/' + data_fold_path[j]
                filename = data_file.split('.')
                if filename[2] == 'xmp':
                    gtdata = np.genfromtxt(data_file, delimiter=',')
                    gtTrace = gtdata[:, 3] # Get ECG signal from signal files
                    gtTime = gtdata[:, 0] / 1000
                    gtHR = gtdata[:, 1] # Get the Heart Rate from the ECG Signal
                    HR.append(gtHR)
                elif filename[2] == 'avi':
                    cap = cv2.VideoCapture(data_file)
                    # Check if camera opened successfully
                    if cap.isOpened() == False:
                        print("Error opening video stream or file")
                    # Read until video is completed
                    while cap.isOpened():
                        # Capture frame-by-frame
                        ret, frame = cap.read()
                        if ret == True:
                            print(i, j, count)
                            original = cv2.resize(frame, [512, 512])
                            face_analysis =  process_video(original)
                            tar = face_analysis
                            Target.append(tar)
                            Images.append(original)
                        else:
                            break
                    # When everything done, release the video capture object
                    cap.release()
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    np.save('Images_1.npy', Images)
    np.save('Heart_rate_1.npy', HR[0])
    np.save('Target_1.npy',tar)

# Read Dataset 2
an = 0
if an == 1:
    dir = './Dataset/DATASET_2/'
    dir1 = os.listdir(dir)
    Target = []
    Images = []
    for i in range(len(dir1)):
        file = dir + dir1[i]
        cap = cv2.VideoCapture(file)
        Heart_rate = extract_heart_rate(file)
        # Check if camera opened successfully
        if cap.isOpened() == False:
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                original = cv2.resize(frame, [512, 512])
                face_analysis = process_video(original)
                tar = face_analysis
                Target.append(tar)
                Images.append(original)
            else:
                break
                # When everything done, release the video capture object
        cap.release()
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    np.save('Images_2.npy', Images)
    np.save('Heart_rate_2.npy', Heart_rate)
    np.save('Target_2.npy', tar)

#  face Detection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Image = []
        for i in range(len(Images)):
            print(i + 1, len(Images))
            image = Images[i]
            detected = Vit_Yolov5(image)
            Image.append(image)
        np.save('Detected_images_' + str(n + 1) + '.npy', Image)

# Feature extraction Set 1
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Detected_images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_'+str(n+1)+'.npy',allow_pickle=True)
        Feat1 = Model_RAN3d(Images,Target)
        np.save('RAN3d_Feat_'+str(n+1)+'.npy',Feat1)

# Feature extraction Set 2
an = 0
if an == 1:
    feat = []
    for n in range(no_of_dataset):
        Images = np.load('Detected_images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        RGB = []
        for i in range(len(Images)):
            print(i, len(Images))
            # rgb = cv2.cvtColor(Images[i], cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(Images[i], cv2.COLOR_BGR2RGB)
            # Extract the RGB values
            R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            # Calculate the mean RGB values
            mean_R = np.mean(R)
            mean_G = np.mean(G)
            mean_B = np.mean(B)
            Feat = np.append(mean_R, np.append(mean_G, mean_B))
            RGB.append(Feat)
        feats = Model_DBN(RGB,Target)
        np.save('DBN_Feat_'+str(n+1)+'.npy', feats)

# Optimization for weighted fused feature and Classification
an = 0
if an == 1:
    FITNESS = []
    Bst = []
    for n in range(no_of_dataset):
        Feat1 = np.load('RAN3d_Feat_'+str(n+1)+'.npy', allow_pickle=True)
        Feat2 = np.load('DBN_Feat_'+str(n+1)+'.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Npop = 10  # no of population
        Chlen = 4
        xmin = matlib.repmat([0.01,0.01,5, 5], Npop, 1)  # minimum length
        xmax = matlib.repmat([0.99,0.99,255, 50], Npop, 1)  # maximum length
        initsol = np.zeros(xmax.shape)  # initial solution
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Objfun_Cls  # objective function
        Max_iter = 50  # maximum iterations

        print("EOO...")
        [bestfit1, fitness1, bestsol1, time1] = EOO(initsol, fname, xmin, xmax, Max_iter)

        print("BMO...")
        [bestfit2, fitness2, bestsol2, time2] = BMO(initsol, fname, xmin, xmax, Max_iter)

        print("NGO...")
        [bestfit3, fitness3, bestsol3, time3] = NGO(initsol, fname, xmin, xmax, Max_iter)

        print("SGO...")
        [bestfit4, fitness4, bestsol4, time4] = SGO(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1.squeeze(),bestsol2.squeeze(), bestsol3.squeeze(),bestsol4.squeeze(),bestsol5.squeeze()]
        Fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]
        FITNESS.append(Fitness)
        Bst.append(BestSol)
    np.save('Fitness.npy', FITNESS)
    np.save('Bestsol.npy',Bst)

# Optimized Feature fusion
an = 0
if an == 1:
    feat = []
    for n in range(no_of_dataset):
        Feat1 = np.load('RAN3d_Feat_'+str(n+1)+'.npy', allow_pickle=True)
        Feat2 = np.load('DBN_Feat_'+str(n+1)+'.npy', allow_pickle=True)
        sol =   np.load('Bestsol.npy', allow_pickle=True)[n][4,:]
        weight1 = sol[0]
        weight2 = sol[1]
        weighted_feat1 = Feat1 * weight1
        weighted_feat2 = Feat2 * weight2
        Fused_Feat = np.concatenate((weighted_feat1, weighted_feat2), axis=1)
        np.save('Weighted_Fused_Feature_'+str(n+1)+'.npy',Fused_Feat)

# Classification
an = 0
if an == 1:
    EVAL_ALL = []
    for k in range(no_of_dataset):
        Feature = np.load('Weighted_Fused_Feature_'+str(k+1)+'.npy', allow_pickle=True)
        HR = np.load('Heart_rate_' + str(k + 1) + '.npy', allow_pickle=True)
        HR = np.resize(HR,[Feature.shape[0],1])
        Feat = np.concatenate((Feature,HR),axis=1)
        Target = np.load('Target_' + str(k + 1) + '.npy', allow_pickle=True)
        Target = np.reshape(Target, (-1, 1))
        BestSol = np.load('Bestsol.npy', allow_pickle=True)[k]
        EVAL = []
        Epoch = [50, 100, 150, 200, 250, 300]
        for Ep in range(len(Epoch)):
            learnperc = round(Feat.shape[0] * 0.75)
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 9))
            for j in range(BestSol.shape[0]):
                print(Ep, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :] = Model_W_ALSTM_AM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :]= Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :] = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :] = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_W_ALSTM_AM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :]= Eval[4, :]
            EVAL.append(Eval)
        EVAL_ALL.append(EVAL)
    np.save('Eval_err.npy', EVAL_ALL)  # Save the Eval all

plot_conv()
plot_Error_results()
plot_Segment_results()
plot_Error_results1()
