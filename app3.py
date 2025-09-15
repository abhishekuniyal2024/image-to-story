import streamlit as st
from groq import Groq
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv
from datetime import datetime
import zipfile

# Load environment variables
load_dotenv()

# Configure Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "Single Image"
if 'final_text' not in st.session_state:
    st.session_state.final_text = None
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = None

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

def integrate_story_with_description(original_story, image_description, extracted_text=""):
    """Integrate image description with original story text"""
    try:
        if original_story.strip():
            prompt = f"""You are a creative children's book narrator specializing in creating engaging stories for visually impaired children.

ORIGINAL STORY TEXT: "{original_story}"

DETAILED IMAGE DESCRIPTION: {image_description}

{f'TEXT IN IMAGE: "{extracted_text}"' if extracted_text else ''}

Your task: Seamlessly blend the visual details into the original story to create one cohesive, enriched narrative that:
- Maintains the original story's tone and style
- Naturally weaves in visual details without feeling forced
- Creates a single, flowing narrative (not separate description + story)
- Makes the scene come alive for a visually impaired child
- Feels like the story was originally written with these visual details included"""
        else:
            # Fallback to enhanced caption if no story text provided
            return enhance_caption_with_emotion(image_description, extracted_text)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating integrated, accessible children's stories that blend narrative and visual description seamlessly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=500,
            temperature=0.8
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error integrating story: {str(e)}")
        return f"Error integrating story: {str(e)}"

def enhance_caption_with_emotion(description, extracted_text=""):
    """Turn a description into an emotionally rich, child-friendly narrative (fallback)"""
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

def main():
    st.set_page_config(
        page_title="Accessible Picture Book Generator", 
        layout="wide",
        page_icon="📚"
    )
    
    st.title("📚 Accessible Picture Book Generator")

    st.markdown("""
    Create accessible picture books for visually impaired children by seamlessly integrating 
    detailed image descriptions with existing story text:
    - **Text Detection:** Automatically extracts text from images
    - **Image Analysis:** Generates detailed visual descriptions  
    - **Story Integration:** Blends descriptions naturally into original story
    - **Batch Processing:** Handle single images or complete picture books
    - **Audio Output:** Optional text-to-speech for complete accessibility
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key status
        if os.getenv("GROQ_API_KEY"):
            st.success("✅ Groq API Key loaded")
        else:
            st.error("❌ Groq API Key not found")
            st.info("Add GROQ_API_KEY to your .env file")

        st.markdown("---")
        st.markdown("**Processing Pipeline:**")
        st.markdown("1. 🔍 Text extraction from image")
        st.markdown("2. 🖼️ Detailed image analysis")
        st.markdown("3. 📝 Story integration")
        st.markdown("4. 🔊 Optional audio generation")
        
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("- Single & batch processing")
        st.markdown("- Seamless story integration")
        st.markdown("- Child-friendly narratives")
        st.markdown("- Complete book export")
        st.markdown("- Audio output support")

        # Add clear results button in sidebar
        if st.session_state.results is not None:
            st.markdown("---")
            if st.button("🗑️ Clear Results", help="Clear current results to process new images"):
                st.session_state.results = None
                st.session_state.final_text = None
                st.session_state.uploaded_files_data = None
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
                for i, file in enumerate(uploaded_files[:8]):  # Show max 8 thumbnails
                    with cols[i % 4]:
                        img = Image.open(file)
                        st.image(img, caption=f"Page {i+1}", use_column_width=True)
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
                                st.image(st.session_state.uploaded_files_data[i]['image'], use_column_width=True)
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

            # Audio Output Section
            st.subheader("🔊 Audio Output")
            audio_enabled = st.checkbox("Enable audio generation", value=False, key="audio_checkbox")
            
            if audio_enabled and st.session_state.final_text:
                col_audio1, col_audio2 = st.columns(2)
                
                with col_audio1:
                    if st.button("▶️ Play Audio", key="play_audio_btn"):
                        try:
                            import pyttsx3
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 150)
                            engine.say(st.session_state.final_text)
                            engine.runAndWait()
                            st.success("Audio playback completed!")
                        except ImportError:
                            st.error("Please install pyttsx3: pip install pyttsx3")
                        except Exception as e:
                            st.error(f"Audio error: {str(e)}")
                
                with col_audio2:
                    if st.button("💾 Save Audio File", key="save_audio_btn"):
                        try:
                            import pyttsx3
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            audio_filename = f"accessible_book_audio_{timestamp}.wav"
                            
                            engine = pyttsx3.init()
                            engine.setProperty('rate', 150)
                            engine.save_to_file(st.session_state.final_text, audio_filename)
                            engine.runAndWait()
                            st.success(f"Audio saved as {audio_filename}")
                        except ImportError:
                            st.error("Please install pyttsx3: pip install pyttsx3")
                        except Exception as e:
                            st.error(f"Audio save error: {str(e)}")
            elif audio_enabled:
                st.info("💡 Generate content first to enable audio features")
            else:
                st.info("💡 Enable audio generation above for text-to-speech features")

            # Download section
            st.subheader("📁 Download Results")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if processing_mode_used == "Single Image":
                # Single file download
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
"""
                
                st.download_button(
                    "📥 Download Results",
                    data=caption_data,
                    file_name=f"accessible_story_{timestamp}.txt",
                    mime="text/plain",
                    key="download_single_btn"
                )
                
            else:
                # Complete book download
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
"""
                
                st.download_button(
                    "📥 Download Complete Accessible Book",
                    data=complete_book_data,
                    file_name=f"complete_accessible_book_{timestamp}.txt",
                    mime="text/plain",
                    key="download_complete_btn"
                )

if __name__ == "__main__":
    main()
