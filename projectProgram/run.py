import argparse
from ViterbiModel import Viterbi

# classifier = Viterbi()
# for i in range(24):
#     print("model", i)
#     classifier.train(i)

def inter(sent):
    # global variables
    classifier = Viterbi()
    modelName = "viterbiModel13.txt" # default model
    model = classifier.load_model(modelName)
    X, y = classifier.predict(sent, model)
    return X, y


def main():
    # global variables
    classifier = Viterbi()
    modelName = "viterbiModel13.txt" # default model

    if arg.train:
        featureNum = int(arg.train[0])
        print("is training ...")
        model = classifier.train(featureNum)
        print("training completed")

    if arg.dev:
        print("load model ", modelName)
        model = classifier.load_model(modelName)
        classifier.test(model)
    if arg.devN:
        modelName = "viterbiModel" + arg.devN[0] + ".txt"
        print("load model ", modelName)
        model = classifier.load_model(modelName)
        classifier.test(model)
    if arg.pred:
        inputSentence = arg.pred[0]
        model = classifier.load_model(modelName)
        classifier.predict(inputSentence, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=str, nargs=1, help='train vertibi model, input feature numbers')
    parser.add_argument('-d', '--dev', nargs='?', const=True, default=False, help="test current model")
    parser.add_argument('-dn', '--devN', type=str, nargs=1, help='test model, input model name')
    parser.add_argument('-p', '--pred', type=str, nargs=1, help='predict part-of-speech tag, input sentence')
    arg = parser.parse_args()

    main()
