import dspy

class GrammaticalitySignature(dspy.Signature):
    """Classify if the sentence is grammatically correct (1) or not (0)."""
    sentence = dspy.InputField()
    label = dspy.OutputField(desc="1 if correct, 0 if incorrect")

class GrammaticalityClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GrammaticalitySignature)

    def forward(self, sentence):
        return self.predict(sentence=sentence)

def custom_metric(example, pred, trace=None):
    return example.label == pred.label
