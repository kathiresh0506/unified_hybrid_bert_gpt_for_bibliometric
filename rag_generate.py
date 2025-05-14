from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(docs, graph_facts):
    graph_context = ""
    if graph_facts["authors"]:
        graph_context += "Authors: " + ", ".join(graph_facts["authors"]) + "\n"
    if graph_facts["keywords"]:
        graph_context += "Keywords: " + ", ".join(graph_facts["keywords"]) + "\n"
    if graph_facts["entities"]:
        graph_context += "Entities: " + ", ".join(graph_facts["entities"]) + "\n"

    base_text = "\n".join(docs)
    full_input = graph_context + "\n" + base_text
    trimmed = full_input[:2000]  # token safety

    summary = summarizer(trimmed, max_length=150, min_length=40, do_sample=False)
    return summary[0]["summary_text"]
