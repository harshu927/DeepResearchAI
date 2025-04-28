import os
import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_community.tools.tavily_search import TavilySearchResults
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import speech_recognition as sr
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from gensim import corpora, models
from sklearn.preprocessing import normalize
import spacy
import torch
from gtts import gTTS
import socket
import threading
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from textblob import TextBlob
import webbrowser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
if not TAVILY_API_KEY:
    logging.error("TAVILY_API_KEY not found in environment variables.")
    sys.exit(1)

# Initialize tools and models with error handling
try:
    tavily_search = TavilySearchResults(max_results=10, api_key=TAVILY_API_KEY)
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("research_data")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logging.error(f"Initialization error: {e}")
    sys.exit(1)

# Local models
device = torch.device("cpu")
try:
    answer_model_name = "t5-base"
    answer_tokenizer = AutoTokenizer.from_pretrained(answer_model_name)
    answer_model = AutoModelForSeq2SeqLM.from_pretrained(answer_model_name).to(device)

    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)
    sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=-1)

    emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
except Exception as e:
    logging.error(f"Model loading error: {e}")
    sys.exit(1)

# Translators
translators = {
    "Spanish": GoogleTranslator(source='auto', target='es'),
    "French": GoogleTranslator(source='auto', target='fr'),
    "Hindi": GoogleTranslator(source='auto', target='hi'),
    "German": GoogleTranslator(source='auto', target='de')
}

# State schema with added fields for new features
class AgentState(TypedDict):
    query: str
    collected_data: List[dict]
    draft_answer: str
    feedback: str
    citations: List[str]
    sentiment: dict
    emotions: dict
    translations: dict
    wordcloud_path: str
    knowledge_graph: nx.Graph
    topics: List[str]
    summary: str
    quantum_insights: List[str]
    refined_query: str
    collab_messages: List[str]
    intent: str
    sentiment_trend: List[float]  # Track sentiment over iterations
    interactive_dashboard: str    # Path to HTML dashboard

# Real-time collaboration server
clients = []

def start_collab_server():
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(('localhost', 5555))
        server.listen(5)
        logging.info("Collaboration server started on port 5555")

        def handle_client(client_socket):
            while True:
                try:
                    message = client_socket.recv(1024).decode('utf-8')
                    if message:
                        for client in clients:
                            client.send(message.encode('utf-8'))
                except Exception as e:
                    logging.warning(f"Client error: {e}")
                    clients.remove(client_socket)
                    client_socket.close()
                    break

        while True:
            client_socket, _ = server.accept()
            clients.append(client_socket)
            threading.Thread(target=handle_client, args=(client_socket,), daemon=True).start()
    except Exception as e:
        logging.error(f"Server error: {e}")

threading.Thread(target=start_collab_server, daemon=True).start()

def send_collab_message(message):
    for client in clients:
        try:
            client.send(message.encode('utf-8'))
        except Exception as e:
            logging.warning(f"Send message error: {e}")

# Enhanced input handling with voice and text
def get_query():
    while True:
        try:
            print("Choose input method: 1 for text, 2 for voice")
            choice = input("Enter 1 or 2: ").strip()
            if choice == "1":
                query = input("Type your query: ").strip()
            elif choice == "2":
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    print("Say your query now!")
                    audio = r.listen(source, timeout=5)
                query = r.recognize_google(audio)
            else:
                print("Invalid choice, please enter 1 or 2.")
                continue

            intents = ["research", "summarize", "analyze", "translate"]
            result = intent_classifier(query, intents, multi_label=False)
            return query, result["labels"][0]
        except Exception as e:
            logging.warning(f"Query input error: {e}")
            query = input("Input failed, type your query: ").strip()
            return query, "research"  # Default intent

# Research agent
def research_node(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        results = tavily_search.invoke(query)
    except Exception as e:
        logging.error(f"Tavily search error: {e}")
        results = []
    state["collected_data"] = results
    try:
        for i, result in enumerate(results):
            collection.add(documents=[result["content"]], metadatas=[{"url": result["url"]}], ids=[f"doc_{i}"])
    except Exception as e:
        logging.error(f"Chroma collection error: {e}")
    send_collab_message(f"Research completed for: {query}")
    return state

# Answer drafter
def drafter_node(state: AgentState) -> AgentState:
    query = state["query"]
    try:
        collected_data = collection.query(query_texts=[query], n_results=5)
        context = "\n".join([doc for doc in collected_data["documents"][0]]) if collected_data["documents"] else "No data available."
        prompt = f"Based on the query '{query}', provide a detailed answer using this context: {context}"
        input_ids = answer_tokenizer.encode(prompt, return_tensors='pt').to(device)
        output = answer_model.generate(input_ids, max_length=300, num_beams=6, early_stopping=True)
        draft = answer_tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Draft generation error: {e}")
        draft = f"Error generating answer for '{query}'. Please refine your query."

    if state["intent"] == "analyze":
        draft = f"Analytical insight: {draft}"
    state["draft_answer"] = draft
    state["citations"] = [result["url"] for result in state["collected_data"] if "url" in result]
    return state

# Sentiment analysis with trend tracking
def sentiment_analysis_node(state: AgentState) -> AgentState:
    text = state["draft_answer"]
    try:
        sentiment = sentiment_analyzer(text)[0]
        state["sentiment"] = {"label": sentiment["label"], "score": sentiment["score"]}
        state["sentiment_trend"].append(sentiment["score"])
        if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.7:
            state["draft_answer"] += " (Adjusted for a more positive tone.)"
    except Exception as e:
        logging.error(f"Sentiment analysis error: {e}")
        state["sentiment"] = {"label": "UNKNOWN", "score": 0.0}
    return state

# Emotion detection
def emotion_node(state: AgentState) -> AgentState:
    text = state["draft_answer"]
    try:
        emotions = emotion_detector(text)[0]
        state["emotions"] = {"label": emotions["label"], "score": emotions["score"]}
    except Exception as e:
        logging.error(f"Emotion detection error: {e}")
        state["emotions"] = {"label": "UNKNOWN", "score": 0.0}
    return state

# Multi-language translation
def translation_node(state: AgentState) -> AgentState:
    text = state["draft_answer"]
    translations = {}
    for lang, translator in translators.items():
        try:
            translations[lang] = translator.translate(text)
        except Exception as e:
            logging.warning(f"Translation error for {lang}: {e}")
            translations[lang] = "Translation unavailable."
    state["translations"] = translations
    return state

# Enhanced word cloud
def wordcloud_node(state: AgentState) -> AgentState:
    try:
        text = " ".join([doc["content"] for doc in state["collected_data"] if "content" in doc])
        if text:
            wordcloud = WordCloud(width=1000, height=500, background_color="black", colormap="viridis").generate(text)
            wordcloud.to_file("wordcloud_enhanced.png")
            state["wordcloud_path"] = "wordcloud_enhanced.png"
        else:
            state["wordcloud_path"] = "No data for word cloud."
    except Exception as e:
        logging.error(f"Word cloud generation error: {e}")
        state["wordcloud_path"] = "Error generating word cloud."
    return state

# Enhanced knowledge graph with networkx
def knowledge_graph_builder_node(state: AgentState) -> AgentState:
    try:
        text = " ".join([doc["content"] for doc in state["collected_data"] if "content" in doc])
        doc = nlp(text)
        entities = list(set([ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]))
        G = nx.Graph()
        for i, ent in enumerate(entities[:10]):
            G.add_node(i, label=ent)
        for i in range(len(entities[:10]) - 1):
            G.add_edge(i, i + 1)
        state["knowledge_graph"] = G
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=3000, node_color='skyblue', font_size=10)
        plt.savefig("knowledge_graph.png")
    except Exception as e:
        logging.error(f"Knowledge graph error: {e}")
        state["knowledge_graph"] = nx.Graph()
    return state

# Topic modeling
def topic_modeling_node(state: AgentState) -> AgentState:
    try:
        texts = [doc["content"].split() for doc in state["collected_data"] if "content" in doc]
        if texts:
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
            state["topics"] = [topic[1] for topic in lda_model.print_topics(num_words=6)]
        else:
            state["topics"] = ["No topics available."]
    except Exception as e:
        logging.error(f"Topic modeling error: {e}")
        state["topics"] = ["Error in topic modeling."]
    return state

# Detailed summarization
def summarization_node(state: AgentState) -> AgentState:
    text = state["draft_answer"]
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        state["summary"] = summary
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        state["summary"] = "Error generating summary."
    return state

# Quantum-inspired insights
def quantum_insight_node(state: AgentState) -> AgentState:
    try:
        text = " ".join([doc["content"] for doc in state["collected_data"] if "content" in doc])
        words = text.split()
        if len(words) > 7:
            probs = normalize(np.random.rand(len(words)).reshape(1, -1), norm='l1')[0]
            top_words = [words[i] for i in np.argsort(probs)[-7:]]
            insight = f"Quantum vision: {', '.join(top_words)} indicate a paradigm where {top_words[0]} and {top_words[1]} converge to revolutionize {top_words[2]}."
            state["quantum_insights"] = [insight]
        else:
            state["quantum_insights"] = ["Insufficient data for quantum insights."]
    except Exception as e:
        logging.error(f"Quantum insight error: {e}")
        state["quantum_insights"] = ["Error generating quantum insights."]
    return state

# Enhanced query refinement with TextBlob
def query_refinement_node(state: AgentState) -> AgentState:
    original_query = state["query"]
    feedback = state["feedback"] or "No feedback provided."
    try:
        blob = TextBlob(feedback)
        key_phrases = blob.noun_phrases
        if key_phrases:
            refined_query = f"{original_query} focusing on {', '.join(key_phrases)}"
        else:
            refined_query = f"{original_query} with detailed insights"
    except Exception as e:
        logging.error(f"Query refinement error: {e}")
        refined_query = f"{original_query} with detailed insights"
    state["refined_query"] = refined_query
    return state

# New: Interactive dashboard generation
def dashboard_node(state: AgentState) -> AgentState:
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deep Research AI Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .section {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .sentiment-positive {{ color: green; }}
                .sentiment-negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Deep Research AI Dashboard</h1>
            <div class="section">
                <h2>Query</h2>
                <p>{state['query']}</p>
            </div>
            <div class="section">
                <h2>Answer</h2>
                <p>{state['draft_answer']}</p>
            </div>
            <div class="section">
                <h2>Sentiment</h2>
                <p class="sentiment-{state['sentiment']['label'].lower()}">{state['sentiment']['label']} (Score: {state['sentiment']['score']:.2f})</p>
            </div>
            <div class="section">
                <h2>Summary</h2>
                <p>{state['summary']}</p>
            </div>
            <div class="section">
                <h2>Topics</h2>
                <ul>{''.join(f'<li>{topic}</li>' for topic in state['topics'])}</ul>
            </div>
            <div class="section">
                <h2>Word Cloud</h2>
                <img src="{state['wordcloud_path']}" alt="Word Cloud" style="max-width: 100%;">
            </div>
        </body>
        </html>
        """
        with open("dashboard.html", "w", encoding='utf-8') as f:
            f.write(html_content)
        state["interactive_dashboard"] = "dashboard.html"
    except Exception as e:
        logging.error(f"Dashboard generation error: {e}")
        state["interactive_dashboard"] = "Error generating dashboard."
    return state

# Collaboration and feedback
def feedback_handler_node(state: AgentState) -> AgentState:
    try:
        print(f"\nDraft Answer: {state['draft_answer']}")
        print(f"Sentiment: {state['sentiment']}")
        print(f"Emotions: {state['emotions']}")
        print(f"Translations: {state['translations']}")
        print(f"Citations: {state['citations']}")
        print(f"Topics: {state['topics']}")
        print(f"Summary: {state['summary']}")
        print(f"Quantum Insights: {state['quantum_insights']}")
        print(f"Refined Query Suggestion: {state['refined_query']}")
        print(f"Collaboration Messages: {state['collab_messages']}")
        feedback = input("Feedback (or 'approve'): ").strip()
        state["feedback"] = feedback
        state["collab_messages"] = state.get("collab_messages", []) + [f"User feedback: {feedback}"]
        send_collab_message(f"Feedback received: {feedback}")

        if feedback.lower() != "approve":
            state["query"] = state["refined_query"]
    except Exception as e:
        logging.error(f"Feedback handler error: {e}")
        state["feedback"] = "Error processing feedback."
    return state

# Workflow
workflow = StateGraph(AgentState)
workflow.add_node("research_agent", research_node)
workflow.add_node("answer_drafter", drafter_node)
workflow.add_node("sentiment_analyzer", sentiment_analysis_node)
workflow.add_node("emotion_detector", emotion_node)
workflow.add_node("translator", translation_node)
workflow.add_node("wordcloud_generator", wordcloud_node)
workflow.add_node("knowledge_graph_builder", knowledge_graph_builder_node)
workflow.add_node("topic_modeler", topic_modeling_node)
workflow.add_node("summarizer", summarization_node)
workflow.add_node("quantum_insight_generator", quantum_insight_node)
workflow.add_node("query_refiner", query_refinement_node)
workflow.add_node("dashboard_generator", dashboard_node)  # New node
workflow.add_node("feedback_handler", feedback_handler_node)

workflow.set_entry_point("research_agent")
workflow.add_edge("research_agent", "answer_drafter")
workflow.add_edge("answer_drafter", "sentiment_analyzer")
workflow.add_edge("sentiment_analyzer", "emotion_detector")
workflow.add_edge("emotion_detector", "translator")
workflow.add_edge("translator", "wordcloud_generator")
workflow.add_edge("wordcloud_generator", "knowledge_graph_builder")
workflow.add_edge("knowledge_graph_builder", "topic_modeler")
workflow.add_edge("topic_modeler", "summarizer")
workflow.add_edge("summarizer", "quantum_insight_generator")
workflow.add_edge("quantum_insight_generator", "query_refiner")
workflow.add_edge("query_refiner", "dashboard_generator")
workflow.add_edge("dashboard_generator", "feedback_handler")
workflow.add_conditional_edges("feedback_handler",
                               lambda state: "research_agent" if state["feedback"].lower() != "approve" else END)

# Compile and run
graph = workflow.compile()
query, intent = get_query()
initial_state = {
    "query": query,
    "collected_data": [],
    "draft_answer": "",
    "feedback": "",
    "citations": [],
    "sentiment": {},
    "emotions": {},
    "translations": {},
    "wordcloud_path": "",
    "knowledge_graph": nx.Graph(),
    "topics": [],
    "summary": "",
    "quantum_insights": [],
    "refined_query": "",
    "collab_messages": [],
    "intent": intent,
    "sentiment_trend": [],
    "interactive_dashboard": ""
}
try:
    result = graph.invoke(initial_state, config={"recursion_limit": 50})
except Exception as e:
    logging.error(f"Graph invocation error: {e}")
    result = initial_state

# Generate PDF report using reportlab
try:
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Deep Research AI Report")
    c.drawString(100, 730, "Query: " + result["query"])
    c.drawString(100, 710, "Answer: " + result["draft_answer"][:100] + "...")
    c.drawString(100, 690, "Sentiment: " + str(result["sentiment"]))
    c.drawString(100, 670, "Emotions: " + str(result["emotions"]))
    c.drawString(100, 650, "Summary: " + result["summary"])
    c.save()
    packet.seek(0)
    with open("research_report.pdf", "wb") as f:
        f.write(packet.read())
    print("PDF report saved as 'research_report.pdf'")
except Exception as e:
    logging.error(f"PDF generation error: {e}")
    print("Failed to save PDF report.")

# Audio output
try:
    tts = gTTS(result["draft_answer"])
    tts.save("answer_audio.mp3")
    print("Audio summary saved as 'answer_audio.mp3'")
except Exception as e:
    logging.error(f"Audio generation error: {e}")
    print("Failed to save audio summary.")

# Open interactive dashboard
try:
    if os.path.exists(result["interactive_dashboard"]):
        webbrowser.open(f"file://{os.path.abspath(result['interactive_dashboard'])}")
        print("Interactive dashboard opened in browser.")
except Exception as e:
    logging.error(f"Dashboard opening error: {e}")
    print("Failed to open interactive dashboard.")