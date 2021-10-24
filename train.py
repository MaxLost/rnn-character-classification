import random
import time
import math
from rnn_model import *
from data_loader import *

# GLOBAL TRAINING PARAMETERS
hidden_size = 128
number_of_iterations = 10**5
print_gap = 5000
plot_gap = 1000
learning_rate = 0.007


# INITIALIZATION
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


parameters = find_parameters()
if parameters != -1:
    print("Founded parameters for model, do you want load it? (Y/N)")
    x = input()
    if x == 'Y' or x == 'y':
        rnn.load_state_dict(torch.load(*parameters), strict=False)
        model_parameters_loaded = True

if model_parameters_loaded == False:
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

    torch.save(rnn.state_dict(), "parameters.pth")

def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output
