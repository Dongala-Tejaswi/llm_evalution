from bert_score import score

def calculate_bertscore(reference, prediction):

    P, R, F1 = score(
        [prediction],
        [reference],
        lang="en"
    )

    return round(
        F1.mean().item(),
        4
    )