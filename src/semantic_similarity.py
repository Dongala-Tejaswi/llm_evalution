from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def calculate_similarity(reference,prediction):

    emb1 = model.encode([reference])
    emb2 = model.encode([prediction])

    similarity = cosine_similarity(
        emb1,
        emb2
    )[0][0]

    return round(
        float(similarity),
        4
    )