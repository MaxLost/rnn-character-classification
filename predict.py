from rnn_model import *
from data_loader import *
import sys

parameters = find_parameters()
if parameters != -1:
    rnn = RNN(len(all_letters), 128, len(all_categories))
    rnn.load_state_dict(torch.load(*parameters), strict=False)
    model_parameters_loaded = True
else:
    exit("RNN model parameters not found!")


def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(input, n_predictions=3):
    print('\n> %s' % input)
    with torch.no_grad():
        output = evaluate(line_to_tensor(input))

        # Get top N categories
        top_value, top_number = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = top_value[0][i].item()
            category_index = top_number[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

if __name__ == "__main__":
    predict(sys.argv[1])