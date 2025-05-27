import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="SDXL-Turbo Text to Image",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state for the model
@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32,
        variant="fp16"
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

# Load the model
with st.spinner("Loading SDXL-Turbo model..."):
    pipe = load_model()

# App title and description
st.title("🎨 SDXL-Turbo Text to Image")
st.markdown("Generate images from text prompts using Stability AI's SDXL-Turbo model.")

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    # Input section
    st.subheader("Input")
    prompt = st.text_area(
        "Enter your prompt",
        placeholder="A beautiful sunset over mountains, photorealistic, 8k",
        height=100
    )
    
    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        placeholder="blurry, low quality, distorted",
        height=100
    )
    
    # Generate button
    if st.button("Generate Image", type="primary"):
        if not prompt:
            st.error("Please enter a prompt")
        else:
            with st.spinner("Generating image..."):
                try:
                    # Generate image
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=1,
                        guidance_scale=0.0,
                    ).images[0]
                    
                    # Display the image
                    st.session_state['generated_image'] = image
                    
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")

with col2:
    # Output section
    st.subheader("Generated Image")
    
    if 'generated_image' in st.session_state:
        # Display the generated image
        st.image(st.session_state['generated_image'], use_column_width=True)
        
        # Download button
        buf = io.BytesIO()
        st.session_state['generated_image'].save(buf, format="PNG")
        img_str = base64.b64encode(buf.getvalue()).decode()
        href = f'data:file/png;base64,{img_str}'
        st.download_button(
            label="Download Image",
            data=buf.getvalue(),
            file_name="generated_image.png",
            mime="image/png"
        )
    else:
        st.info("Your generated image will appear here")

# Add some helpful information
with st.expander("Tips for better results"):
    st.markdown("""
    - Be specific in your prompts
    - Use descriptive adjectives
    - Mention the style you want (e.g., photorealistic, anime, oil painting)
    - Use negative prompts to avoid unwanted elements
    - The model is optimized for speed, so results may vary
    """) 