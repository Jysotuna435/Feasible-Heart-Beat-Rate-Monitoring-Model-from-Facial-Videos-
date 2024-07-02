import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

no_of_dataset = 2
def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v
def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'EOO-WALSTM-AM', 'BMO-WALSTM-AM', 'NGO-WALSTM-AM', 'SGO-WALSTM-AM ', 'ESGO-WALSTM-AM']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report Dataset ', i + 1,
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]


        plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=8, label='EOO-WALSTM-AM')
        plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=8, label='BMO-WALSTM-AM')  # c
        plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, marker='*', markerfacecolor='cyan',
                 markersize=8, label='NGO-WALSTM-AM')
        plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=8, label='SGO-WALSTM-AM')  # y
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='o', markerfacecolor='black',
                 markersize=5, label='ESGO-WALSTM-AM')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_%s.png" % (i + 1))
        plt.show()

def plot_Error_results():  # Irrigation Prediciton
    eval1 = np.load('Eval_All_err.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'mse', 'NMSE', 'ONENORM', 'TWONORM']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6]

    learnper = [0, 1, 2, 3, 4, 5]
    for i in range(eval1.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2]))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    Graph[k, l] = eval1[i, k, l, Graph_Terms[j]]

            plt.plot(learnper, Graph[:, 0], color='b', linewidth=3, marker='o', markerfacecolor='blue',
                     markersize=12, label="EOO-WALSTM-AM")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red',
                     markersize=12, label="BMO-WALSTM-AM")
            plt.plot(learnper, Graph[:, 2], color='r', linewidth=3, marker='o', markerfacecolor='green',
                     markersize=12, label="NGO-WALSTM-AM")
            plt.plot(learnper, Graph[:, 3], color='g', linewidth=3, marker='o', markerfacecolor='yellow',
                     markersize=12, label="SGO-WALSTM-AM")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='magenta',
                     markersize=12, label="ESGO-WALSTM-AM")
            plt.xticks(learnper, ('50', '100', '150', '200', '250', '300'))
            plt.xlabel('Epoch')
            plt.ylabel(Terms[Graph_Terms[j]])

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_Prid_%s_line_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
            X = np.arange(6)

            ax.bar(X + 0.00, Graph[:, 5], color='#ff5b00', edgecolor='k', width=0.10, label="CNN")
            ax.bar(X + 0.10, Graph[:, 6], color='#75fd63', edgecolor='k', width=0.10, label="RNN")
            ax.bar(X + 0.20, Graph[:, 7], color='#3d7afd', edgecolor='k', width=0.10, label="RESNET")
            ax.bar(X + 0.30, Graph[:, 8], color='#cb00f5', edgecolor='k', width=0.10, label="WALSTM-AM")
            ax.bar(X + 0.40, Graph[:, 4], color='k', edgecolor='k', width=0.10, label="ESGO-WALSTM-AM")
            plt.xticks(X + 0.10, ['50', '100', '150', '200', '250', '300'])
            plt.xlabel('Epoch')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_Prid_%s_bar_lrean.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

def plot_Error_results1():  # Irrigation Prediciton
    eval1 = np.load('Eval_error_batch.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'mse', 'NMSE', 'ONENORM', 'TWONORM']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6]
    Algorithm = ['TERMS', 'EOO-WALSTM-AM', 'BMO-WALSTM-AM', 'NGO-WALSTM-AM', 'SGO-WALSTM-AM ', 'ESGO-WALSTM-AM']
    Classifier = ['TERMS','CNN', 'RNN', 'RESNET', 'WALSTM-AM', 'ESGO-WALSTM-AM']

    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, :]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- Dataset' + str( i+1) +
              '(Batch_Size)--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- Dataset' + str(
            i + 1) +
              '(Batch_Size)--------------------------------------------------')
        print(Table)
def plot_Segment_results():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)

    Terms = ['IoU', 'Precision', 'Sensitivity', 'Dice Coefficient', 'Jaccard', 'Accuracy', 'Specificity', 'FPR', 'FNR',
             'NPV', 'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i]) * 100
                    stats[i, j, 1] = np.min(value_all[j][:, i]) * 100
                    stats[i, j, 2] = np.mean(value_all[j][:, i]) * 100
                    stats[i, j, 3] = np.median(value_all[j][:, i]) * 100
                    stats[i, j, 4] = np.std(value_all[j][:, i]) * 100
            X = np.arange(stats.shape[2])


            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.10, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 5, :], color='#fc2647', edgecolor='k', width=0.10, label="Yolo")
            ax.bar(X + 0.10, stats[i, 6, :], color='#2ee8bb', edgecolor='k', width=0.10, label="Yolov3")
            ax.bar(X + 0.20, stats[i, 7, :], color='#aa23ff', edgecolor='k', width=0.10, label="Yolov5")
            ax.bar(X + 0.30, stats[i, 8, :], color='#fe46a5', edgecolor='k', width=0.10, label="M_Yolov5")
            ax.bar(X + 0.40, stats[i, 4, :] + 0.05, color='k', edgecolor='k', width=0.10, label="ViT_Yolov5")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_seg_%s_met.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()



if __name__ == '__main__':
    plot_conv()
    plot_Error_results()
    plot_Segment_results()
    plot_Error_results1()
