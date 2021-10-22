import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from train import *
from predict import predict

predict("Stern")
predict("Starovoytov")
predict("Ken")

plt.figure()
plt.plot(all_losses)
plt.show()
