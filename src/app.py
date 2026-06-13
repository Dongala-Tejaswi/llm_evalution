import streamlit as st
import time
import pandas as pd

from groq_client import generate_answer
from bleu_metric import calculate_bleu
from bertscore_metric import calculate_bertscore
from semantic_similarity import calculate_similarity

st.set_page_config(
    page_title="GroqEval",
    layout="wide"
)

st.title("🚀 GroqEval - LLM Evaluation Dashboard")

question = st.text_area(
    "Enter Question"
)

reference = st.text_area(
    "Reference Answer"
)

if st.button("Evaluate"):

    if question.strip() == "":
        st.warning("Enter a question")
        st.stop()

    if reference.strip() == "":
        st.warning("Enter reference answer")
        st.stop()

    with st.spinner("Generating answer..."):

        start = time.time()

        prediction = generate_answer(
            question
        )

        latency = round(
            time.time() - start,
            2
        )

    bleu = round(
        calculate_bleu(
            reference,
            prediction
        ),
        4
    )

    bert = "Disabled"

    similarity = calculate_similarity(
        reference,
        prediction
    )

    st.subheader("Generated Answer")

    st.success(prediction)

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "BLEU",
        bleu
    )

    col2.metric(
        "BERTScore",
        bert
    )

    col3.metric(
        "Similarity",
        similarity
    )

    col4.metric(
        "Latency (sec)",
        latency
    )

    df = pd.DataFrame({

        "Metric":[
            "BLEU",
            "BERTScore",
            "Similarity"
        ],

        "Score":[
            bleu,
            bert,
            similarity
        ]
    })

    st.subheader(
        "Metrics Chart"
    )

    st.bar_chart(
        df.set_index(
            "Metric"
        )
    )

    st.subheader(
        "Detailed Results"
    )

    st.dataframe(df)