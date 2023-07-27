import torch
from matplotlib import pyplot as plt
from os.path import exists

from train_network import Model, run_test_set, load_data
from pose import semaphore_numbers


def run_test_set_by_label(test_data: torch.tensor, test_labels: torch.tensor, model):
    """
    This tests the model and returns the number correct by which label they are
    """
    model.eval()
    pred = model(test_data)
    n = pred.shape[0]
    count = [0] * 27
    count_corr = [0] * 27
    count_top_2 = [0] * 27
    for i in range(n):
        _, top_indices = torch.topk(pred[i], 2)
        label = torch.argmax(test_labels[i])
        count[label] += 1
        count_corr[label] += int(top_indices[0] == label)
        count_top_2[label] += int(label in top_indices)
    for i in range(27):
        count_corr[i] = count_corr[i] / count[i]
        count_top_2[i] = count_top_2[i] / count[i]

    return count_corr, count_top_2


"""
This function is simply to evaluate model performance on various
datasets and values
"""


def load_model(model_path, model):
    if exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded model at " + model_path)
        model.eval()
    else:
        ValueError("No classifier model was found at {}".format(model_path))

    return model


model = Model(38, 60, 27)
model_path = "train_semaphore.pt"
model = load_model(model_path, model)

# original dataset:
data_file = "data.csv"
label_file = "labels.csv"
# hand labelled:
data_file_2 = "data_labelled.csv"
label_file_2 = "labels_labelled.csv"

data, labels = load_data(data_file, label_file, "cpu")
data_2, labels_2 = load_data(data_file_2, label_file_2, "cpu")
data = torch.concat((data, data_2))
labels = torch.concat((labels, labels_2))


p_corr, p_top_two = run_test_set(data, labels, model)
print(
    "The model got {:.1f}% correct and {:.1f}% in the top two".format(
        p_corr * 100, p_top_two * 100
    )
)


p_cor, p_2 = run_test_set_by_label(data, labels, model)

plt.figure()
plt.bar(range(27), p_2, label="top two")
plt.bar(range(27), p_cor, label="correct")
plt.xticks(range(len(semaphore_numbers)), list(semaphore_numbers.keys()))
plt.legend()
plt.ylabel("Proportion correct")
plt.show()
