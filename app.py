import streamlit as st
from groq import Groq
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_detailed_image_description(image):
    base64_image = encode_image_to_base64(image)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are an expert at describing images for children's picture books. 
Analyze this image in great detail and provide a rich, vivid description that captures:
- All characters, objects, and settings
- Colors, textures, and visual details
- Emotions and expressions
- Actions and movements
- Mood and atmosphere
Be extremely descriptive and paint a complete picture with words."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=500,
            temperature=0.7
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating description: {str(e)}"

def enhance_caption_with_emotion(description, original_text=""):
    try:
        prompt = f"""You are a creative children's book narrator. Transform this detailed image description into an emotionally engaging, child-friendly narrative.

Original story text (if any): {original_text}

Detailed image description: {description}

Create an enhanced, emotionally-rich description that:
- Uses vivid, child-friendly language
- Incorporates sensory details
- Adds emotional depth and wonder
- Makes the scene come alive for a visually impaired child
- Feels natural and story-like"""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled children's book narrator creating vivid descriptions for visually impaired children."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=400,
            temperature=0.8
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error enhancing caption: {str(e)}"

def main():
    st.set_page_config(page_title="Picture Book Caption Generator", layout="wide")
    st.title("AI Picture Book Caption Generator")
    st.markdown("""
Generate rich, emotionally-driven descriptions of picture book images 
for visually impaired children using a two-stage AI approach:
- **Stage 1:** Detailed image analysis
- **Stage 2:** Emotional enhancement
""")
    
    with st.sidebar:
        st.header("Configuration")
        if os.getenv("GROQ_API_KEY"):
            st.success("Groq API Key loaded")
        else:
            st.error("Groq API Key not found")
            st.info("Add GROQ_API_KEY to your .env file")
        st.markdown("---")
        st.markdown("**Models Used:**")
        st.markdown("- Llama Scout Vision: Image analysis")
        st.markdown("- Llama 3.3 70B: Text enhancement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose a picture book image...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            original_text = st.text_area("Original Story Text (Optional)", height=100)
    
    with col2:
        st.header("Generated Captions")
        if uploaded_file and st.button("Generate Enhanced Caption"):
            with st.spinner("Analyzing image..."):
                description = generate_detailed_image_description(image)
                st.subheader("Stage 1: Detailed Description")
                st.text_area("Raw Image Analysis:", value=description, height=150, key="stage1")
            
            with st.spinner("Enhancing..."):
                enhanced_caption = enhance_caption_with_emotion(description, original_text)
                st.subheader("Stage 2: Enhanced Caption")
                st.text_area("Final Enhanced Caption:", value=enhanced_caption, height=200, key="stage2")
            
            st.success("Caption generation complete!")
            caption_data = f"""ORIGINAL TEXT:
{original_text}

DETAILED DESCRIPTION:
{description}

ENHANCED CAPTION:
{enhanced_caption}
"""
            st.download_button("Download Results", data=caption_data, file_name=f"caption_{uploaded_file.name}.txt", mime="text/plain")

if __name__ == "__main__":
    main()
