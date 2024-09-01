!pip install transformers gradio torch

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import os
import torch  # Import torch to check for GPU availability

# Set up Hugging Face token
hf_token = "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"
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

# Load the model with token authentication
model_name = 'EleutherAI/gpt-neo-2.7B'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

# Ensure you're using CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if device == 'cuda' else -1)

# Function to predict disease based on symptoms
def predict_disease(symptoms):
    # Structuring the prompt for a more accurate medical diagnosis
    prompt = (
        f"The patient had multiple symptoms over several days: {symptoms}. "
        "Considering all these symptoms, what is the most likely disease or condition?"
    )
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

# Create the Gradio interface
def create_interface():
    with gr.Blocks(
        css="""
        .gradio-container {
            background: linear-gradient(to right, #f8d7d0, #8a2be2);  /* Light orange to purple gradient */
            min-height: 100vh;  /* Full viewport height */
            width: 100vw;       /* Full viewport width */
            position: relative;
            overflow: hidden;
        }
        .gradio-button {
            font-size: 14px;    /* Adjust font size for buttons */
            padding: 6px 12px;  /* Adjust padding for buttons */
            font-weight: bold;  /* Make button text bold */
        }
        .gradio-textbox {
            width: 60%;  /* Adjust text box width */
            max-width: 300px; /* Set a maximum width */
            margin: 0 auto; /* Center-align text box */
            font-size: 16px; /* Increase font size for text inside text boxes */
        }
        .title {
            font-size: 72px;  /* Double the font size of the title */
            font-weight: bold; /* Make title bold */
            text-align: center; /* Center-align title */
            margin-bottom: 10px; /* Add spacing below title */
            color: #4B0082; /* Set title color (Indigo) */
        }
        .subtitle {
            font-size: 48px;  /* Double the font size of the subtitle */
            text-align: center; /* Center-align subtitle */
            margin-bottom: 20px; /* Add spacing below subtitle */
            color: #4B0082; /* Set subtitle color (Indigo) */
        }
        .emoji {
            font-size: 1.5em; /* Adjust emoji size */
        }
        .circle {
            position: absolute;
            background: rgba(138, 43, 226, 0.5); /* Purple circles */
            border-radius: 50%;
            opacity: 0.3; /* Make circles semi-transparent */
        }
        .circle1 { width: 150px; height: 150px; top: 10%; left: 5%; }
        .circle2 { width: 100px; height: 100px; top: 50%; left: 20%; }
        .circle3 { width: 200px; height: 200px; top: 75%; left: 60%; }
        .circle4 { width: 100px; height: 100px; top: 30%; left: 80%; }
        .circle5 { width: 120px; height: 120px; top: 15%; left: 90%; }
        .circle6 { width: 180px; height: 180px; top: 40%; left: 40%; }
        .circle7 { width: 130px; height: 130px; top: 60%; left: 25%; }
        .circle8 { width: 170px; height: 170px; top: 20%; left: 70%; }
        """
    ) as demo:
        gr.Markdown('<div class="title">Medify 🤖</div>')
        gr.Markdown('<div class="subtitle">Your AI Healthcare Assistant 🏥</div>')
        gr.Markdown("Welcome to Medify, your personal AI healthcare assistant. Ask any medical question and receive detailed, informed responses. Please note that while Medify strives to provide accurate information, it\'s important to consult healthcare professionals for personal medical advice. 💡👩‍⚕️")

        with gr.Row():
            with gr.Column(scale=2):
                symptoms_input = gr.Textbox(label="Enter patient's symptoms over time", placeholder="Type symptoms here...", lines=3)
                predict_button = gr.Button("Predict Disease 📤")
                diagnosis_output = gr.Textbox(label="Predicted Disease or Condition 🧠", lines=5)
                predict_button.click(fn=predict_disease, inputs=symptoms_input, outputs=diagnosis_output)

        gr.Markdown("""
        **Disclaimer:** The information provided by Medify is intended for general informational purposes only and should not be considered as medical advice. Always consult a qualified healthcare provider for professional medical advice and treatment. ⚠️
        """)

    return demo

# Launch Gradio app
if __name__ == "__main__":
    app = create_interface()
    app.launch()
