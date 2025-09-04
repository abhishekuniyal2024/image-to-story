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
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_text_from_image(image):
    """Extract text from image using Groq's vision model"""
    base64_image = encode_image_to_base64(image)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Look carefully at this image and extract ALL visible text, including:
- Title text
- Book titles
- Any words or letters visible in the image
- Signs, labels, or written content

If you see any text, write it exactly as it appears. If there's absolutely no readable text, respond with just "NO_TEXT"."""
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
            max_tokens=200,
            temperature=0.1
        )
        result = chat_completion.choices[0].message.content.strip()
        return "" if result == "NO_TEXT" else result
    except Exception:
        return ""


def generate_detailed_image_description(image):
    """Generate a detailed description of the image"""
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


def enhance_caption_with_emotion(description, extracted_text=""):
    """Turn a description into an emotionally rich, child-friendly narrative"""
    try:
        if extracted_text:
            prompt = f"""You are a creative children's book narrator. Transform this detailed image description into an emotionally engaging, child-friendly narrative.

Text found in the image: "{extracted_text}"

Detailed image description: {description}

Create an enhanced, emotionally-rich description that:
- Uses vivid, child-friendly language
- Naturally incorporates the text found in the image into the narrative
- Incorporates sensory details
- Adds emotional depth and wonder
- Makes the scene come alive for a visually impaired child
- Feels natural and story-like"""
        else:
            prompt = f"""You are a creative children's book narrator. Transform this detailed image description into an emotionally engaging, child-friendly narrative.

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
for visually impaired children using an intelligent AI approach:
- **Text Detection:** Automatically checks for text in images
- **Image Analysis:** Detailed visual description
- **Enhancement:** Creates emotionally engaging narratives
""")

    with st.sidebar:
        st.header("Configuration")
        if os.getenv("GROQ_API_KEY"):
            st.success("Groq API Key loaded")
        else:
            st.error("Groq API Key not found")
            st.info("Add GROQ_API_KEY to your .env file")

        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("- Automatic text detection")
        st.markdown("- Detailed image analysis")
        st.markdown("- Emotional enhancement")
        st.markdown("- Child-friendly narratives")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a picture book image...",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width="stretch")

    with col2:
        st.header("Generated Caption")
        if uploaded_file and st.button("Generate Enhanced Caption"):
            with st.spinner("Processing image..."):
                extracted_text = extract_text_from_image(image)

                if extracted_text:
                    st.success("✓ Text detected in image")
                    with st.expander("View detected text"):
                        st.text(extracted_text)
                else:
                    st.info("No text detected - proceeding with visual description only")

            with st.spinner("Analyzing image details..."):
                description = generate_detailed_image_description(image)
                with st.expander("View detailed analysis"):
                    st.text_area(
                        "Image Analysis",
                        value=description,
                        height=120,
                        key="analysis",
                        label_visibility="collapsed"
                    )

            with st.spinner("Creating enhanced caption..."):
                enhanced_caption = enhance_caption_with_emotion(description, extracted_text)
                st.subheader("Enhanced Caption")
                st.text_area(
                    "Enhanced Caption",
                    value=enhanced_caption,
                    height=200,
                    key="final_caption",
                    label_visibility="collapsed"
                )

            st.success("Caption generation complete!")

            caption_data = f"""IMAGE FILE: {uploaded_file.name}
GENERATION DATE: {st.session_state.get('timestamp', 'N/A')}

TEXT DETECTED: {"Yes" if extracted_text else "No"}
{f'EXTRACTED TEXT: {extracted_text}' if extracted_text else ''}

DETAILED ANALYSIS:
{description}

ENHANCED CAPTION:
{enhanced_caption}
"""
            st.download_button(
                "Download Results",
                data=caption_data,
                file_name=f"caption_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()
