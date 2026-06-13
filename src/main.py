from utils.data_loader import load_dataset

from models.llama_generator import generate_answer

from evaluation.bleu_metric import calculate_bleu
from evaluation.rouge_metric import calculate_rouge
from evaluation.bertscore_metric import calculate_bertscore
from evaluation.meteor_metric import calculate_meteor
from evaluation.semantic_similarity import calculate_similarity
from evaluation.latency_metric import measure_latency

from utils.report_generator import save_metrics

dataset = load_dataset(
    "datasets/sample_dataset.csv"
)

all_results=[]

for _,row in dataset.iterrows():

    question = row["question"]
    reference = row["reference_answer"]

    prediction,latency = measure_latency(
        generate_answer,
        question
    )

    result={

        "question":question,

        "prediction":prediction,

        "BLEU":
        calculate_bleu(
            reference,
            prediction
        ),

        "ROUGE":
        str(
            calculate_rouge(
                reference,
                prediction
            )
        ),

        "BERTScore":
        calculate_bertscore(
            reference,
            prediction
        ),

        "METEOR":
        calculate_meteor(
            reference,
            prediction
        ),

        "Semantic Similarity":
        calculate_similarity(
            reference,
            prediction
        ),

        "Latency":
        latency
    }

    all_results.append(result)

save_metrics(all_results)

print("Evaluation Completed")