import pandas 
from sklearn import tree 


data_train = pandas.read_pickle('data_train.pkl')
ans_train = pandas.read_pickle('ans_train.pkl')

def my_tree(animal):
    # if animal has no backbone it is not a predator
    if animal['backbone'] == 0:
        return 0 

    else:
        # if animal has no teeth it is not a predator
        if animal['toothed'] == 0:
            return 0
        # if animal has teeth and backbone it is a predator
        else:
            return 1 

            
def get_predictions(data):

    predictions = []
    for _, animal in data.iterrows():
        prediction = my_tree(animal)
        predictions.append(prediction)

    return predictions


def test_predictions(predictions, answers):
    correct = 0
    for pred, ans in zip(predictions, answers):
        if pred == ans:
            correct += 1
    print('%s/%s correct!'%(correct, len(answers))) 



predictions = get_predictions(data_train)
test_predictions(predictions, ans_train)

auto_tree = tree.DecisionTreeClassifier()
auto_tree = auto_tree.fit(data_train.drop(columns='animal name'), ans_train)

auto_preds = auto_tree.predict(data_train.drop(columns='animal name'))

test_predictions(auto_preds, ans_train) 