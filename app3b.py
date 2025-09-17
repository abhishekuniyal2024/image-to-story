import streamlit as st
from groq import Groq
import base64
from PIL import Image
import io
import os
import time
from dotenv import load_dotenv
from datetime import datetime
import zipfile
from gtts import gTTS  # Added gTTS import
from pydub import AudioSegment  # Added for audio combining

# Load environment variables
load_dotenv()

# Configure Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "Single Image"
if 'final_text' not in st.session_state:
    st.session_state.final_text = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = None
if 'last_audio_file' not in st.session_state:
    st.session_state.last_audio_file = None
if 'audio_downloaded' not in st.session_state:
    st.session_state.audio_downloaded = False

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
        
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
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
        st.error(f"Error generating description: {str(e)}")
        return f"Error generating description: {str(e)}"

def ensure_complete_story(text, max_tokens=400):
    """Ensure story ends with complete sentence"""
    if not text:
        return text
        
    text = text.strip()
    
    # Check if text ends abruptly (no proper punctuation)
    if not text.endswith(('.', '!', '?')):
        # Find last complete sentence
        for punct in ['. ', '! ', '? ']:
            last_punct = text.rfind(punct)
            if last_punct != -1:
                # Cut at last complete sentence
                text = text[:last_punct + 1]
                break
        
        # If still no proper ending, add simple conclusion
        if not text.endswith(('.', '!', '?')):
            text = text.rstrip() + "."
    
    return text

def integrate_story_with_description(original_story, image_description, extracted_text=""):
    """Integrate image description with original story text"""
    try:
        if original_story.strip():
            prompt = f"""You are a creative children's book narrator specializing in creating engaging stories for visually impaired children.

ORIGINAL STORY TEXT: "{original_story}"

DETAILED IMAGE DESCRIPTION: {image_description}

{f'TEXT IN IMAGE: "{extracted_text}"' if extracted_text else ''}

Your task: Create a complete, concise story in 2-3 short paragraphs that:
- Seamlessly blends visual details into the original story
- Maintains the original story's tone and style
- Creates a single, flowing narrative
- Makes the scene come alive for a visually impaired child
- MUST END with a proper conclusion - no incomplete sentences
- Keep it concise but emotionally rich
- Ensure it reads naturally when spoken aloud

IMPORTANT: Write a complete story that finishes properly within 2-3 paragraphs."""
        else:
            # Fallback to enhanced caption if no story text provided
            return enhance_caption_with_emotion(image_description, extracted_text)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert children's story narrator. Create complete, concise stories that finish properly within 2-3 paragraphs. Always end sentences and stories completely."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=400,  # KEEP AT 400
            temperature=0.8
        )
        
        result = chat_completion.choices[0].message.content
        return ensure_complete_story(result)
        
    except Exception as e:
        st.error(f"Error integrating story: {str(e)}")
        return f"Error integrating story: {str(e)}"

def enhance_caption_with_emotion(description, extracted_text=""):
    """Turn a description into an emotionally rich, child-friendly narrative (fallback)"""
    try:
        if extracted_text:
            prompt = f"""You are a creative children's book narrator. Create a complete, engaging story from this image description.

Text found in the image: "{extracted_text}"

Detailed image description: {description}

Create a complete story in 2-3 short paragraphs that:
- Uses vivid, child-friendly language
- Naturally incorporates the text found in the image
- Includes sensory details and emotional depth
- Makes the scene come alive for a visually impaired child
- MUST have a proper ending - no cut-off sentences
- Reads naturally when spoken aloud
- Is concise but emotionally rich

IMPORTANT: Write a complete story that ends properly within 2-3 paragraphs."""
        else:
            prompt = f"""You are a creative children's book narrator. Create a complete, engaging story from this image description.

Detailed image description: {description}

Create a complete story in 2-3 short paragraphs that:
- Uses vivid, child-friendly language
- Includes sensory details and emotional depth
- Makes the scene come alive for a visually impaired child
- MUST have a proper ending - no cut-off sentences
- Reads naturally when spoken aloud
- Is concise but emotionally rich

IMPORTANT: Write a complete story that ends properly within 2-3 paragraphs."""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a skilled children's narrator. Create complete, concise stories in 2-3 paragraphs that always end properly. Never leave sentences unfinished."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=400,  # KEEP AT 400
            temperature=0.8
        )
        
        result = chat_completion.choices[0].message.content
        return ensure_complete_story(result)
        
    except Exception as e:
        st.error(f"Error enhancing caption: {str(e)}")
        return f"Error enhancing caption: {str(e)}"

def process_single_image(image, story_text, filename):
    """Process a single image and return results"""
    results = {}
    
    # Text extraction
    extracted_text = extract_text_from_image(image)
    results['extracted_text'] = extracted_text
    
    # Image description
    description = generate_detailed_image_description(image)
    results['description'] = description
    
    # Story integration
    if story_text.strip():
        final_text = integrate_story_with_description(story_text, description, extracted_text)
        results['type'] = 'integrated'
    else:
        final_text = enhance_caption_with_emotion(description, extracted_text)
        results['type'] = 'enhanced'
    
    results['final_text'] = final_text
    results['filename'] = filename
    results['story_text'] = story_text
    
    return results

def split_text_into_chunks(text, max_size=400):
    """Split text into chunks while preserving sentence boundaries - FIXED VERSION"""
    chunks = []
    
    # Simple sentence-based splitting without paragraph break placeholders
    sentences = text.split('. ')
    current_chunk = ""
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add period back except for the last sentence (it might already have one)
        if i < len(sentences) - 1 and not sentence.endswith('.'):
            sentence += '. '
        elif not sentence.endswith('.') and not sentence.endswith('!') and not sentence.endswith('?'):
            sentence += '. '
        else:
            sentence += ' '
        
        # Check if adding this sentence would exceed the limit
        if len(current_chunk + sentence) > max_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Single sentence is too long, split by words
                words = sentence.split(' ')
                for word in words:
                    if len(current_chunk + word + " ") > max_size:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                            current_chunk = word + " "
                        else:
                            # Single word too long, just add it
                            chunks.append(word)
                            current_chunk = ""
                    else:
                        current_chunk += word + " "
        else:
            current_chunk += sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_audio_with_gtts(text, filename, accent='us'):
    """Generate audio using Google Text-to-Speech - WAV format for reliability"""
    try:
        # Use the EXACT same text as displayed
        audio_text = text
        
        # Set TLD based on accent preference
        tld = 'com' if accent == 'us' else 'co.in'
        
        # Force WAV extension for reliability
        if filename.endswith('.mp3'):
            filename = filename.replace('.mp3', '.wav')
        
        # Full path to output directory
        full_filename = os.path.join(OUTPUT_DIR, filename)
        
        # Check text length - if too long, split into chunks
        max_chunk_size = 400
        
        if len(audio_text) <= max_chunk_size:
            # Short text - process normally, save as MP3 first for gTTS
            temp_mp3 = os.path.join(OUTPUT_DIR, "temp_single.mp3")
            tts = gTTS(text=audio_text, lang='en', tld=tld)
            tts.save(temp_mp3)
            
            # Convert to WAV
            audio_segment = AudioSegment.from_mp3(temp_mp3)
            audio_segment.export(full_filename, format="wav")
            
            # Clean up temp MP3
            try:
                os.remove(temp_mp3)
            except:
                pass
            
            st.info(f"✅ Audio saved as WAV in output folder")
            return True, audio_text, full_filename
        else:
            # Long text - chunk processing with fixed chunking function
            chunks = split_text_into_chunks(audio_text, max_chunk_size)
            audio_segments = []
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    temp_filename = os.path.join(OUTPUT_DIR, f"temp_chunk_{i}.mp3")
                    tts = gTTS(text=chunk.strip(), lang='en', tld=tld)
                    tts.save(temp_filename)
                    
                    segment = AudioSegment.from_mp3(temp_filename)
                    audio_segments.append(segment)
                    audio_segments.append(AudioSegment.silent(duration=500))
                    
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
            
            if audio_segments:
                combined_audio = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined_audio += segment
                
                # Export as WAV
                combined_audio.export(full_filename, format="wav")
                st.info(f"✅ Audio saved as WAV in output folder")
                
                return True, audio_text, full_filename
        
    except Exception as e:
        return False, str(e), os.path.join(OUTPUT_DIR, filename)

def save_text_file(data, filename):
    """Save text file to output directory"""
    full_path = os.path.join(OUTPUT_DIR, filename)
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(data)
        return full_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Accessible Picture Book Generator",
        layout="wide",
        page_icon="📚"
    )
    
    st.title("📚 Accessible Picture Book Generator")
    st.markdown(f"""
    Create accessible picture books for visually impaired children by seamlessly integrating
    detailed image descriptions with existing story text:
    
    - **Text Detection:** Automatically extracts text from images
    - **Image Analysis:** Generates detailed visual descriptions
    - **Story Integration:** Blends descriptions naturally into original story
    - **Batch Processing:** Handle single images or complete picture books
    - **Audio Output:** Natural WAV audio synthesis matching displayed text exactly
    
    📁 **All files saved to:** `{OUTPUT_DIR}/` folder
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key status
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            st.success("✅ Groq API Key loaded")
        else:
            st.error("❌ Groq API Key not found")
            st.info("Add GROQ_API_KEY to your .env file")
        
        st.markdown("---")
        st.markdown("**File Storage:**")
        st.success(f"📁 Output folder: `{OUTPUT_DIR}/`")
        
        st.markdown("---")
        st.markdown("**Processing Pipeline:**")
        st.markdown("1. 🔍 Text extraction from image")
        st.markdown("2. 🖼️ Detailed image analysis")
        st.markdown("3. 📝 Story integration")
        st.markdown("4. 🔊 Complete story within 400 tokens")
        st.markdown("5. 💾 Save to output folder")
        
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("- Single & batch processing")
        st.markdown("- Seamless story integration")
        st.markdown("- Child-friendly narratives")
        st.markdown("- Audio matches text exactly")
        st.markdown("- Complete stories, no cut-offs")
        
        # Add clear results button in sidebar
        if st.session_state.results is not None:
            st.markdown("---")
            if st.button("🗑️ Clear Results", help="Clear current results to process new images"):
                st.session_state.results = None
                st.session_state.final_text = None
                st.session_state.uploaded_files_data = None
                st.session_state.last_audio_file = None
                st.session_state.audio_downloaded = False  # RESET DOWNLOAD STATE
                st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Content")
        
        # Processing mode selection
        processing_mode = st.radio(
            "Processing Mode:",
            ["Single Image", "Multiple Images (Complete Book)"],
            horizontal=True,
            help="Choose single image for testing or multiple images for complete picture book processing"
        )

        if processing_mode == "Single Image":
            uploaded_file = st.file_uploader(
                "Choose a picture book image...",
                type=["png", "jpg", "jpeg"]
            )
            uploaded_files = [uploaded_file] if uploaded_file else []
        else:
            uploaded_files = st.file_uploader(
                "Choose multiple picture book images...",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                help="Upload all pages of your picture book (typically 15-30 images)"
            )

        # Display uploaded images
        if uploaded_files and uploaded_files[0] is not None:
            if processing_mode == "Single Image":
                image = Image.open(uploaded_files[0])
                st.image(image, caption="Uploaded Image")
            else:
                st.write(f"📚 **{len(uploaded_files)} images uploaded**")
                # Show thumbnails in a grid
                cols = st.columns(min(4, len(uploaded_files)))
                for i, file in enumerate(uploaded_files[:8]):
                    with cols[i % 4]:
                        img = Image.open(file)
                        st.image(img, caption=f"Page {i+1}", use_container_width=True)
                if len(uploaded_files) > 8:
                    st.info(f"... and {len(uploaded_files) - 8} more images")

        # Story text input section
        st.subheader("📖 Story Text")
        if processing_mode == "Single Image" and uploaded_files and uploaded_files[0] is not None:
            story_text = st.text_area(
                "Enter the original story text for this page (optional):",
                height=120,
                placeholder="Once upon a time, in a magical forest...",
                help="Provide the existing story text to create an integrated narrative. Leave empty for description-only mode."
            )
            story_texts = [story_text]
        elif processing_mode == "Multiple Images (Complete Book)" and uploaded_files:
            st.write("Enter story text for each page:")
            story_texts = []
            
            # Create tabs for easier navigation
            if len(uploaded_files) <= 5:
                for i, file in enumerate(uploaded_files):
                    story_text = st.text_area(
                        f"Page {i+1} - {file.name}:",
                        height=80,
                        placeholder=f"Story text for page {i+1}...",
                        key=f"story_{i}"
                    )
                    story_texts.append(story_text)
            else:
                # For books with many pages, use expandable sections
                story_texts = []
                for i, file in enumerate(uploaded_files):
                    with st.expander(f"📄 Page {i+1} - {file.name}"):
                        story_text = st.text_area(
                            "Story text:",
                            height=80,
                            placeholder=f"Story text for page {i+1}...",
                            key=f"story_{i}"
                        )
                        story_texts.append(story_text)
        else:
            story_texts = []

    with col2:
        st.header("✨ Generated Content")
        
        # Process images when button is clicked
        if uploaded_files and uploaded_files[0] is not None and st.button("🚀 Generate Accessible Story", type="primary"):
            results = []
            
            # Store uploaded files data for later use
            st.session_state.uploaded_files_data = []
            for file in uploaded_files:
                img_data = Image.open(file)
                st.session_state.uploaded_files_data.append({
                    'name': file.name,
                    'image': img_data
                })
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each image
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                
                # Load image
                image = Image.open(uploaded_file)
                current_story_text = story_texts[i] if i < len(story_texts) else ""
                
                # Process image
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = process_single_image(image, current_story_text, uploaded_file.name)
                    results.append(result)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.processing_mode = processing_mode
            
            st.success(f"✅ Successfully processed {len(results)} {'image' if len(results) == 1 else 'images'}!")

        # Display results from session state
        if st.session_state.results is not None:
            results = st.session_state.results
            processing_mode_used = st.session_state.processing_mode

            # Display results
            if processing_mode_used == "Single Image":
                # Single image display
                result = results[0]
                
                if result['extracted_text']:
                    st.success("✅ Text detected in image")
                    with st.expander("👀 View detected text"):
                        st.text_area("Extracted Text:", value=result['extracted_text'], height=80, disabled=True, key="extracted_text_display")
                else:
                    st.info("ℹ️ No text detected - proceeding with visual analysis")
                
                with st.expander("🔍 View detailed analysis"):
                    st.text_area("Image Analysis:", value=result['description'], height=120, disabled=True, key="analysis_display")
                
                st.subheader("📚 Final Result")
                result_label = "Integrated Story" if result['type'] == 'integrated' else "Enhanced Caption"
                st.text_area(result_label, value=result['final_text'], height=200, disabled=True, key="final_result_display")
                
                final_text = result['final_text']
            else:
                # Multiple images display
                st.subheader("📖 Complete Accessible Book")
                
                # Show summary
                text_detected_count = sum(1 for r in results if r['extracted_text'])
                integrated_count = sum(1 for r in results if r['type'] == 'integrated')
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Pages Processed", len(results))
                with col_stat2:
                    st.metric("Text Detected", text_detected_count)
                with col_stat3:
                    st.metric("Story Integrated", integrated_count)
                
                # Display each page in expandable sections
                for i, result in enumerate(results):
                    with st.expander(f"📄 Page {i+1}: {result['filename']}"):
                        col_left, col_right = st.columns([1, 2])
                        
                        with col_left:
                            # Show image if available in session state
                            if st.session_state.uploaded_files_data and i < len(st.session_state.uploaded_files_data):
                                st.image(st.session_state.uploaded_files_data[i]['image'], use_container_width=True)
                            else:
                                st.write(f"**Image:** {result['filename']}")
                            
                            if result['extracted_text']:
                                st.success("✅ Text detected")
                                st.text(result['extracted_text'])
                            else:
                                st.info("No text detected")
                        
                        with col_right:
                            st.text_area(
                                "Final Story:",
                                value=result['final_text'],
                                height=150,
                                disabled=True,
                                key=f"final_{i}_display"
                            )
                
                # Compile complete book text
                complete_book = ""
                for i, result in enumerate(results):
                    complete_book += f"PAGE {i+1}: {result['filename']}\n\n"
                    complete_book += result['final_text'] + "\n\n"
                    complete_book += "---\n\n"
                
                final_text = complete_book

            # Store final text in session state
            st.session_state.final_text = final_text

            # SIMPLIFIED AUDIO OUTPUT SECTION
            st.subheader("🔊 Audio Output")
            audio_enabled = st.checkbox("Enable audio generation", value=False, key="audio_checkbox")

            if audio_enabled and st.session_state.final_text:
                # Show text length info
                text_length = len(st.session_state.final_text)
                word_count = len(st.session_state.final_text.split())
                estimated_duration = word_count / 2.5
                st.info(f"📊 Text: {text_length} characters, {word_count} words (≈{estimated_duration:.1f} minutes)")
                
                # Show chunking info for long text
                if text_length > 400:
                    chunks = split_text_into_chunks(st.session_state.final_text, 400)
                    st.info(f"🔄 Long text detected: Will be processed in {len(chunks)} audio chunks")

                # Audio settings
                with st.expander("🎛️ Audio Settings"):
                    accent_choice = st.selectbox(
                        "Voice Accent:",
                        ["US English", "Indian English"],
                        index=0,
                        help="Choose between US or Indian English accent"
                    )
                    
                    st.success(f"✅ Audio saved to `{OUTPUT_DIR}/` folder")

                col_audio1, col_audio2 = st.columns(2)

                with col_audio1:
                    if st.button("🎧 Generate & Play Audio Preview", key="preview_audio_btn"):
                        try:
                            with st.spinner("Generating audio preview..."):
                                # Generate preview (first 200 words)
                                preview_text = ' '.join(st.session_state.final_text.split()[:200])
                                if len(st.session_state.final_text.split()) > 200:
                                    preview_text += "..."
                                
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                preview_filename = f"preview_{timestamp}.wav"
                                
                                accent = 'us' if accent_choice == "US English" else 'in'
                                success, audio_text, final_filename = generate_audio_with_gtts(preview_text, preview_filename, accent)
                                
                                if success:
                                    st.success(f"🎵 Audio preview saved!")
                                    
                                    # Audio player
                                    with open(final_filename, "rb") as audio_file:
                                        audio_bytes = audio_file.read()
                                        st.audio(audio_bytes, format="audio/wav")
                                    
                                    # Show saved file path
                                    st.info(f"📁 Preview saved: `{final_filename}`")
                                    
                                else:
                                    st.error(f"Failed to generate audio: {audio_text}")
                                    
                        except Exception as e:
                            st.error(f"Audio preview error: {str(e)}")

                with col_audio2:
                    if st.button("💾 Generate Complete Audio File", key="save_audio_btn"):
                        try:
                            with st.spinner("Generating complete audio file..."):
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                audio_filename = f"accessible_book_audio_{timestamp}.wav"
                                
                                accent = 'us' if accent_choice == "US English" else 'in'
                                success, audio_text, final_filename = generate_audio_with_gtts(st.session_state.final_text, audio_filename, accent)
                                
                                if success:
                                    st.success(f"✅ Complete audio file saved!")
                                    
                                    # Store audio file path and reset download state
                                    st.session_state.last_audio_file = final_filename
                                    st.session_state.audio_downloaded = False  # RESET DOWNLOAD STATE FOR NEW AUDIO
                                    
                                    # Show file info
                                    if os.path.exists(final_filename):
                                        file_size = os.path.getsize(final_filename)
                                        st.info(f"📁 File: `{final_filename}`")
                                        st.info(f"📏 Format: WAV, Size: {file_size/1024:.1f} KB")
                                    
                                else:
                                    st.error(f"Failed to generate audio: {audio_text}")
                                    
                        except Exception as e:
                            st.error(f"Audio generation error: {str(e)}")

            elif audio_enabled:
                st.info("💡 Generate content first to enable audio features")
            else:
                st.info(f"💡 Enable audio generation above - files saved to `{OUTPUT_DIR}/`")

            # DOWNLOAD SECTION WITH PROPER STATE MANAGEMENT
            if st.session_state.last_audio_file and os.path.exists(st.session_state.last_audio_file):
                st.markdown("---")
                st.subheader("📥 Download Audio")
                
                if not st.session_state.audio_downloaded:
                    st.info("💡 Audio file is already saved in output folder. Use button below if you need to download it elsewhere.")
                    
                    # Direct download button
                    with open(st.session_state.last_audio_file, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.download_button(
                            label=f"⬇️ Download Audio File (WAV)",
                            data=audio_bytes,
                            file_name=os.path.basename(st.session_state.last_audio_file),
                            mime="audio/wav",
                            key="direct_download_audio_btn",
                            help="Download audio file to your device"
                        )
                    
                    # Mark as downloaded button
                    if st.button("✅ Mark as Downloaded", key="mark_downloaded_btn", help="Click after downloading to hide the download button"):
                        st.session_state.audio_downloaded = True
                        st.success("🎉 Marked as downloaded!")
                        st.rerun()
                        
                else:
                    st.success("✅ Audio file marked as downloaded!")
                    st.info(f"📁 Local file: `{st.session_state.last_audio_file}`")
                    
                    # Reset button to show download again
                    if st.button("🔄 Show Download Button Again", key="show_download_again_btn"):
                        st.session_state.audio_downloaded = False
                        st.rerun()

            # SAVE TEXT RESULTS SECTION
            st.subheader("💾 Save Text Results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if processing_mode_used == "Single Image":
                # Single file save
                result = results[0]
                caption_data = f"""ACCESSIBLE PICTURE BOOK GENERATOR

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMAGE FILE: {result['filename']}

ORIGINAL STORY TEXT:
{result['story_text'] if result['story_text'].strip() else "No original story text provided"}

TEXT DETECTION STATUS: {"Text Found" if result['extracted_text'] else "No Text Detected"}

{f'EXTRACTED TEXT:\n{result["extracted_text"]}\n' if result['extracted_text'] else ''}

RAW IMAGE ANALYSIS:
{result['description']}

FINAL INTEGRATED STORY:
{result['final_text']}

---
Generated by Accessible Picture Book Generator
For visually impaired children - bridging stories and sight through AI
Audio content matches text exactly for consistent experience
"""

                # SINGLE SAVE BUTTON ONLY
                if st.button("💾 Save Text Results", key="save_single_btn", help=f"Save text file to {OUTPUT_DIR}/ folder"):
                    filename = f"accessible_story_{timestamp}.txt"
                    saved_path = save_text_file(caption_data, filename)
                    if saved_path:
                        st.success(f"✅ Text results saved to: `{saved_path}`")
                    else:
                        st.error("❌ Failed to save text file")

            else:
                # Complete book save
                complete_book_data = f"""ACCESSIBLE PICTURE BOOK - COMPLETE EDITION

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Total Pages: {len(results)}
Processing Mode: Batch Processing

{'='*60}
COMPLETE ACCESSIBLE STORY
{'='*60}
"""

                for i, result in enumerate(results):
                    complete_book_data += f"""
PAGE {i+1}: {result['filename']}

{result['final_text']}

{'─'*40}
"""

                complete_book_data += f"""
{'='*60}
PROCESSING DETAILS
{'='*60}
"""

                for i, result in enumerate(results):
                    complete_book_data += f"""PAGE {i+1} - {result['filename']}:
Original Story Text: {result['story_text'] if result['story_text'].strip() else "None provided"}
Text Detected: {"Yes" if result['extracted_text'] else "No"}
{f'Extracted Text: {result["extracted_text"]}' if result['extracted_text'] else ''}
Raw Analysis: {result['description']}

"""

                complete_book_data += """
---
Generated by Accessible Picture Book Generator
For visually impaired children - bridging stories and sight through AI
Audio content matches text exactly for consistent experience
"""

                # SINGLE SAVE BUTTON ONLY
                if st.button("💾 Save Complete Book", key="save_complete_btn", help=f"Save complete book to {OUTPUT_DIR}/ folder"):
                    filename = f"complete_accessible_book_{timestamp}.txt"
                    saved_path = save_text_file(complete_book_data, filename)
                    if saved_path:
                        st.success(f"✅ Complete book saved to: `{saved_path}`")
                    else:
                        st.error("❌ Failed to save complete book file")

if __name__ == "__main__":
    main()
