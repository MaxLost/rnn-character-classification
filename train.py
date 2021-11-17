import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rnn_model import *
from data_loader import *

# GLOBAL TRAINING PARAMETERS
hidden_size = 128
number_of_iterations = 10**5
print_gap = 5000
plot_gap = 1000
learning_rate = 0.0055

# INITIALIZATION OF RNN MODEL
rnn = RNN(len(all_letters), hidden_size, len(all_categories))
criterion = nn.NLLLoss()

current_loss = 0
all_losses = []


def category_from_output(output):
    value, num = output.topk(1)  # Returns greatest value in output tensor and category that has this value
    most_likely_category = num.item()
    return all_categories[most_likely_category], most_likely_category


def select_random_object(x):
    return x[random.randint(0, len(x)-1)]


def get_random_line():
    category = select_random_object(all_categories)
    line = select_random_object(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train_rnn(category_tensor, line_tensor):
    hidden_layer = rnn.init_hidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden_layer = rnn(line_tensor[i], hidden_layer)

    loss = criterion(output, category_tensor)
    loss.backward()

    for n in rnn.parameters():
        n.data.add_(n.grad.data, alpha=-learning_rate)

    return output, loss.item()


def elapsed_time(start):
    now = time.time()
    sec = now - start
    min = sec // 60
    sec -= min * 60

    return "%dmin %dsec" % (min, sec)


# TRAINING MODEL
start = time.time()

for i in range(1, number_of_iterations + 1):
    category, line, category_tensor, line_tensor = get_random_line()
    output, loss = train_rnn(category_tensor, line_tensor)

    current_loss += loss

    if i % print_gap == 0:
        guess, guess_val = category_from_output(output)
        if guess == category:
            result = "/ Correct"
        else:
            result = "/ Incorrect (%s)" % category
        print('%d %d%% (%s) / %.4f %s / %s %s' % (i, i / number_of_iterations * 100, elapsed_time(start), loss,
                                                      line, guess, result))

        if i % plot_gap == 0:
            all_losses.append(current_loss / plot_gap)
            current_loss = 0

torch.save(rnn.state_dict(), "rnn_parameters.pth")


# CREATING PLOTS
plt.figure()
plt.plot(all_losses)

confusion = torch.zeros(len(all_categories), len(all_categories))
n_confusion = 10000
number_of_confusions = 0

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = get_random_line()
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    guess, guess_i = category_from_output(output)
    category_i = all_categories.index(category)

    if guess_i != category_i:
        number_of_confusions += 1

    confusion[category_i][guess_i] += 1

print("RNN accuracy:", 100-(number_of_confusions/n_confusion)*100, "%")

for i in range(len(all_categories)):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()
