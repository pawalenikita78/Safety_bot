from django.shortcuts import render
from django.http import JsonResponse
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
import requests
import os



# 1. Extract PDF Text
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Load and prepare both PDF texts
pdf_text_general = extract_text_from_pdf(r"C:\Desktop\Safety_Chatbot\general_safety_rules.pdf")
pdf_text_emergency = extract_text_from_pdf(r"C:\Desktop\Safety_Chatbot\FA-manual.pdf")

# Split both texts into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks_general = splitter.split_text(pdf_text_general)
chunks_emergency = splitter.split_text(pdf_text_emergency)

# Mistral API query function (Using API key)
def query_mistral(prompt, context, history=None):
    # Build conversation history string
    history_str = ""
    if history:
        for h in history:
            history_str += f"User: {h['user']}\nAssistant: {h['bot']}\n"
    history_str += f"User: {prompt}\n"

    full_prompt = (
        "You are a polite, helpful assistant. "
        "For greetings, goodbyes, and general conversation, answer naturally using your general knowledge. "
        "For all other questions, use the information provided below if relevant. "
        "If you do not know the answer, reply exactly: 'Sorry, the answer is not available.'\n\n"
        "Do not mention the provided document or its contents directly.\n\n"
        f"{context}\n\n"
        f"Conversation history:\n{history_str}"
    )

    api_url = "https://api.mistral.ai/v1/chat/completions"
    api_key = "tYE6EWGxo8N9X2EUY7d7B1qqdY1pewgK" 

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        print("Error communicating with Mistral API:", e)
        return "Sorry, I could not get a response from the chatbot server."

def find_relevant_chunks(user_input, chatbot_type):
    """
    Returns a list of relevant chunks from the selected PDF based on user input and chatbot type.
    """
    if chatbot_type == 'emergency':
        chunks = chunks_emergency
    else:
        chunks = chunks_general

    relevant_chunks = []
    user_words = set(user_input.lower().split())
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        if user_words & chunk_words:
            relevant_chunks.append(chunk)
    return relevant_chunks

# 2. View to Render the Chatbot UI
def chatbot_home(request):
    return render(request, 'chatbot.html')

# 3. View to Handle Chatbot Input
def get_chatbot_response(request):
    if request.method == 'GET':
        user_input = request.GET.get('user_input', '').strip()
        source = request.GET.get('source', 'general').lower()
        print("DEBUG: Source received from frontend:", source)

        # List of general conversation keywords/phrases
        general_keywords = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "goodbye", "bye", "see you", "thank you", "thanks"
        ]

        assistant_type = source

        # Check if user input is a general conversation
        if any(greet in user_input.lower() for greet in general_keywords):
            context = f"(You are a {assistant_type} safety assistant.)"
            bot_response = query_mistral(user_input, context, [])
        else:
            # Use the new function to find relevant chunks
            relevant_chunks = find_relevant_chunks(user_input, assistant_type)

            # If relevant chunks found, use them as context
            if relevant_chunks:
                context = " ".join(relevant_chunks[:2])
                print(f"DEBUG: Responding from {assistant_type} PDF")
                history = request.session.get('chat_history', [])
                history = history[-5:]
                bot_response = query_mistral(user_input, context, history)
                bot_response += f"\n\n[DEBUG: Source={assistant_type}]"
            else:
                print(f"DEBUG: No relevant info found in {assistant_type} PDF.")
                bot_response = (
                    "Sorry, I couldn't find an answer to your question. Please try rephrasing your question or ask something else."
                    f"\n\n[DEBUG: Source={assistant_type}]"
                )

        # Update conversation history in session
        history = request.session.get('chat_history', [])
        history.append({'user': user_input, 'bot': bot_response})
        request.session['chat_history'] = history

        return JsonResponse({'response': bot_response})



