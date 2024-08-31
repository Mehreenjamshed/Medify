# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import streamlit as st
import speech_recognition as sr
from deep_translator import GoogleTranslator

# Setup model and tokenizer for AI healthcare assistant
model_name = "ruslanmv/Medical-Llama3-8B"
device_map = 'auto'
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        use_cache=False,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    st.error(f"Model loading failed: {e}")

# Define the function to ask questions
def askme(question):
    sys_message = '''
    You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
    provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
    '''
    # Create messages structured for the chat template
    messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": question}]

    # Applying chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)

    # Extract and return the generated text, removing the prompt
    response_text = tokenizer.batch_decode(outputs)[0].strip()
    answer = response_text.split('<|im_start|>assistant')[-1].strip()
    return answer

# Initialize recognizer and translator
recognizer = sr.Recognizer()
translator = GoogleTranslator(source='ur', target='en')

# Function to transcribe live audio to text
def transcribe_live_audio():
    with sr.Microphone() as source:
        st.write("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening...")
        audio_data = recognizer.listen(source)
        try:
            # Transcribe the audio
            text = recognizer.recognize_google(audio_data, language='ur')
            return text
        except sr.UnknownValueError:
            return "Speech Recognition could not understand the audio"
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service"

# Function to translate text to English
def translate_text(text):
    translation = translator.translate(text)
    return translation

# Streamlit UI
st.title("AI Healthcare Assistant and Transcription Service")
st.write("Click the button below to start transcription and translation.")

# Button to start transcription
if st.button("Start Transcription"):
    with st.spinner("Transcribing and translating..."):
        transcribed_text = transcribe_live_audio()
        translated_text = translate_text(transcribed_text)
        
        # Display results
        st.write("Transcribed Text (Urdu):", transcribed_text)
        st.write("Translated Text (English):", translated_text)

# Continuous interaction loop for AI Medical Assistant
if st.button("Ask a Medical Question"):
    question = st.text_input("Please enter your medical question:")
    if question:
        answer = askme(question)
        st.write("AI Medical Assistant: ", answer)
