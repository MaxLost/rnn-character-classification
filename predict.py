from train import *


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
