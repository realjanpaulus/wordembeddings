import matplotlib.pyplot as plt
import models


INPUT_DIM = 10000
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
OUTPUT_DIM = 2
DROPOUT = 0.5
PAD_IDX = 1

model = models.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
print(model)



"""

train_losses = [0.9, 0.877, 0.63, 0.7, 0.5]
val_losses = [1.5, 1.44, 1.2, 1.11, 1.24]
train_losses = [0.9]
val_losses = [1.5]

plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.legend()
plt.title("Losses")
plt.savefig(f"../../results/test.png")"""