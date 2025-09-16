# app8_simplified.py: Children's Book Story Generator with Vision LLM + Edge-TTS

import streamlit as st
import os
import base64
import asyncio
import tempfile
from io import BytesIO
from dotenv import load_dotenv
from groq import Groq
import edge_tts

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error('GROQ_API_KEY is not available in the environment or .env file')
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Define constants
VISION_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
}

MODEL_OPTIONS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": "LLaMA 4 - Scout 17B (Vision)",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "LLaMA 4 - Maverick 17B (Vision)"
}

AGE_GROUPS = {
    "preschool": "Preschool (3-5 years)",
    "elementary": "Elementary (6-10 years)",
    "middle_grade": "Middle Grade (11-14 years)"
}

AGE_GUIDELINES = {
    "preschool": "Use simple, warm language with basic emotions and short sentences.",
    "elementary": "Use engaging vocabulary with educational elements and moderate complexity.",
    "middle_grade": "Use rich descriptions with complex emotional themes and advanced vocabulary."
}

VOICE_OPTIONS = {
    "Female Voice": {"voice": "en-US-AriaNeural"},
    "Male Voice": {"voice": "en-US-GuyNeural"},
    "Female (UK)": {"voice": "en-GB-SoniaNeural"},
    "Male (UK)": {"voice": "en-GB-RyanNeural"},
    "Child Voice": {"voice": "en-US-JennyNeural"}
}

EMOTION_OPTIONS = {
    "Neutral": {"speed": 1.0, "pitch": "normal"},
    "Happy/Excited": {"speed": 1.1, "pitch": "high"},
    "Calm/Soothing": {"speed": 0.9, "pitch": "low"},
    "Mysterious": {"speed": 0.8, "pitch": "low"},
    "Energetic": {"speed": 1.2, "pitch": "high"}
}

LENGTH_MAPPING = {
    "Short (100-200 words)": "100-200 words",
    "Medium (300-500 words)": "300-500 words",
    "Long (500+ words)": "500+ words"
}

# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def encode_image_file(file):
    """Encode uploaded image to base64"""
    try:
        file.seek(0)
        return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return None

def apply_emotion_to_text(text, emotion):
    """Modify text based on emotion for better TTS output"""
    emotion_transforms = {
        "Happy/Excited": lambda t: f"*excited voice* {t.replace('.', '!').replace('?', '?!')}",
        "Calm/Soothing": lambda t: f"*gentle voice* {t.replace('.', '... ').replace('!', '.')}",
        "Mysterious": lambda t: f"*mysterious whisper* {t.replace('.', '... ').replace(',', '... ')}",
        "Energetic": lambda t: f"*energetic voice* {t.replace('.', '! ').replace('very', 'VERY')}"
    }
    return emotion_transforms.get(emotion, lambda t: t)(text)

async def generate_speech_async(text, voice, output_path, emotion_config):
    """Async function to generate speech using edge-tts"""
    rate = f"{int((emotion_config['speed'] - 1) * 50):+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output_path)

def text_to_speech(text, voice_config, emotion_config):
    """Enhanced TTS with edge-tts for better voice and emotion support"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            temp_path = tmp_file.name
        
        asyncio.run(generate_speech_async(text, voice_config["voice"], temp_path, emotion_config))
        
        with open(temp_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        os.unlink(temp_path)
        
        audio_fp = BytesIO(audio_data)
        audio_fp.seek(0)
        return audio_fp
        
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def create_story_prompt(age_group, custom_prompt, word_count):
    """Create the story generation prompt"""
    guideline = AGE_GUIDELINES.get(age_group, AGE_GUIDELINES['elementary'])
    
    return f"""
You are creating descriptions for a children's book that will be read to visually impaired students aged {age_group}.
{guideline}

Create an emotionally rich, story-integrated description of the image that includes:
1. Scene's mood/atmosphere
2. Character interactions (if any)
3. Sensory details (sound, texture, smell)
4. Age-appropriate narrative
5. Context to help children imagine being part of the scene
6. Colors, light, style

Target length: {word_count}

{f"Additional instructions: {custom_prompt}" if custom_prompt else ""}

Make it immersive and engaging.
"""

def generate_story_description(uploaded_file, model_name, age_group, custom_prompt, creativity_level, story_length):
    """Generate story description using selected LLM"""
    base64_image = encode_image_file(uploaded_file)
    if not base64_image and model_name in VISION_MODELS:
        return 'ERROR: Could not encode image'

    word_count = LENGTH_MAPPING.get(story_length, "100-200 words")
    story_prompt = create_story_prompt(age_group, custom_prompt, word_count)

    try:
        if model_name in VISION_MODELS:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": story_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        else:
            fallback_prompt = story_prompt + "\nNote: You cannot see the image. Use your imagination and generate a generic but rich description suitable for a children's book."
            messages = [{"role": "user", "content": fallback_prompt}]

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=creativity_level,
            max_tokens=1024
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

def create_download_buttons(audio_bytes, description, mode, index=None, selected_voice=None):
    """Create download buttons for audio and text"""
    col1, col2 = st.columns(2)
    
    if mode == "single":
        with col1:
            st.download_button(
                label="Download Audio File",
                data=audio_bytes.getvalue(),
                file_name=f"story_audio_{selected_voice.replace(' ', '_').lower()}.mp3",
                mime="audio/mp3",
                type="secondary",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="Download Story Text",
                data=description,
                file_name="generated_story.txt",
                mime="text/plain",
                type="secondary",
                use_container_width=True
            )
    else:
        with col1:
            st.download_button(
                label=f"Download Audio {index}",
                data=audio_bytes.getvalue(),
                file_name=f"story_{index}_audio.mp3",
                mime="audio/mp3",
                key=f"audio_{index}",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label=f"Download Text {index}",
                data=description,
                file_name=f"story_{index}_text.txt",
                mime="text/plain",
                key=f"text_{index}",
                use_container_width=True
            )

def process_story_and_audio(uploaded_file, settings, mode, index=None):
    """Process a single image to generate story and audio"""
    with st.spinner("Creating your magical story..."):
        description = generate_story_description(
            uploaded_file, 
            settings["model_choice"], 
            settings["age_group"],
            settings["custom_prompt"],
            settings["creativity_level"],
            settings["story_length"]
        )
    
    # Display story
    header = "### Generated Story:" if mode == "single" else f"**Story {index}:**"
    st.markdown(header)
    st.markdown(f"*{description}*")
    
    # Generate and display audio
    with st.spinner("Generating audio narration..."):
        voice_config = VOICE_OPTIONS[settings["selected_voice"]]
        emotion_config = EMOTION_OPTIONS[settings["selected_emotion"]]
        
        emotional_text = apply_emotion_to_text(description, settings["selected_emotion"])
        audio_bytes = text_to_speech(emotional_text, voice_config, emotion_config)
        
        if audio_bytes:
            if mode == "single":
                st.markdown("### Audio Narration:")
            st.audio(audio_bytes, format="audio/mp3")
            create_download_buttons(audio_bytes, description, mode, index, settings["selected_voice"])

# ═══════════════════════════════════════════════════════════════════════════════
# UI PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_image(uploaded_file, settings):
    """Handle single image processing"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.image(uploaded_file, caption="Selected Image", use_container_width=True)
    
    with col2:
        model_name = settings["model_choice"].split('/')[-1]
        st.info(f"""**Current Settings:**
- Model: {model_name}
- Age: {settings["age_group"]}
- Length: {settings["story_length"]}
- Voice: {settings["selected_voice"]}
- Emotion: {settings["selected_emotion"]}""")
    
    if st.button("Generate Story and Audio", type="primary", use_container_width=True):
        process_story_and_audio(uploaded_file, settings, "single")

def process_batch_images(uploaded_files, settings):
    """Handle batch processing"""
    # Preview uploaded images
    if st.checkbox("Preview All Images"):
        cols = st.columns(min(4, len(uploaded_files)))
        for i, up_file in enumerate(uploaded_files[:8]):
            with cols[i % 4]:
                st.image(up_file, caption=f"Image {i+1}", use_container_width=True)
        if len(uploaded_files) > 8:
            st.caption(f"... and {len(uploaded_files) - 8} more images")
    
    if st.button("Generate All Stories and Audio", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, up_file in enumerate(uploaded_files, 1):
            progress = i / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing image {i} of {len(uploaded_files)}...")
            
            with st.expander(f"Story {i}: {up_file.name}", expanded=(i==1)):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.image(up_file, caption=f"Image {i}", use_container_width=True)
                
                with col2:
                    model_name = settings["model_choice"].split('/')[-1]
                    st.info(f"""**Current Settings:**
- Model: {model_name}
- Age: {settings["age_group"]}
- Length: {settings["story_length"]}
- Voice: {settings["selected_voice"]}
- Emotion: {settings["selected_emotion"]}""")
                
                process_story_and_audio(up_file, settings, "batch", i)
        
        status_text.text("All stories generated successfully!")
        st.success(f"**Batch processing complete!** Generated {len(uploaded_files)} stories with audio narration.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application function"""
    st.title("Children's Book Story Generator")
    st.markdown("*Transform images into stories with advanced customization*")

    # Sidebar - All user settings
    with st.sidebar:
        st.header("Settings Panel")
        
        # AI Model Configuration
        st.subheader("AI Model Configuration")
        model_choice = st.selectbox(
            "Choose a Language Model:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x]
        )

        # Age group selection
        age_group = st.selectbox(
            "Select Target Age Group:",
            options=list(AGE_GROUPS.keys()),
            format_func=lambda x: AGE_GROUPS[x]
        )

        # Story Customization
        st.subheader("Story Customization")
        custom_prompt = st.text_area(
            "Custom Story Instructions (optional):",
            placeholder="e.g., Include a magical forest, make it rhyme, focus on problem-solving...",
            help="Add specific instructions to personalize your story"
        )

        creativity_level = st.slider(
            "Creativity Level", 0.1, 1.0, 0.7, 0.1,
            help="Higher values make stories more creative and unpredictable"
        )

        story_length = st.selectbox(
            "Story Length:",
            list(LENGTH_MAPPING.keys())
        )

        # Voice and Audio Settings
        st.subheader("Voice and Audio Settings")
        selected_voice = st.selectbox("Select Voice:", list(VOICE_OPTIONS.keys()))
        selected_emotion = st.selectbox("Voice Emotion:", list(EMOTION_OPTIONS.keys()))

    # Main content area
    st.subheader("Upload Images")
    uploaded_files = st.file_uploader(
        "Choose image file(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one image for single processing, or multiple images for batch processing"
    )

    if uploaded_files:
        # Collect all settings
        settings = {
            "model_choice": model_choice,
            "age_group": age_group,
            "custom_prompt": custom_prompt,
            "creativity_level": creativity_level,
            "story_length": story_length,
            "selected_voice": selected_voice,
            "selected_emotion": selected_emotion
        }

        # Auto-detect single vs batch mode
        num_files = len(uploaded_files)
        
        if num_files == 1:
            st.info("**Single Image Mode** - Processing 1 image")
            process_single_image(uploaded_files[0], settings)
        else:
            st.info(f"**Batch Mode** - Processing {num_files} images")
            process_batch_images(uploaded_files, settings)

if __name__ == "__main__":
    main()
