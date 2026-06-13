from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, prediction):

    smoothie = SmoothingFunction().method1

    score = sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=smoothie
    )

    return round(score, 4)