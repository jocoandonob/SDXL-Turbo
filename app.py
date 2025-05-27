import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.utils import load_image
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="SDXL-Turbo Text to Image",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state for the models
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        st.warning("SDXL-Turbo is optimized for CUDA (GPU). Running on CPU may be very slow.")
    pipe_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None
    )
    pipe_text2image.to(device)
    pipe_image2image = AutoPipelineForImage2Image.from_pipe(pipe_text2image)
    pipe_image2image.to(device)
    return pipe_text2image, pipe_image2image, device

# Load the models
with st.spinner("Loading SDXL-Turbo models..."):
    pipe_text2image, pipe_image2image, device = load_models()

# App title and description
st.title("🎨 SDXL-Turbo Image Generation")
st.markdown("Generate and transform images using Stability AI's SDXL-Turbo model.")

# Create tabs
tab1, tab2 = st.tabs(["Text to Image", "Image to Image"])

with tab1:
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
        if st.button("Generate Image", type="primary", key="text2img_btn"):
            if not prompt:
                st.error("Please enter a prompt")
            else:
                with st.spinner("Generating image..."):
                    try:
                        # Generate image
                        image = pipe_text2image(
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
                mime="image/png",
                key="text2img_download"
            )
        else:
            st.info("Your generated image will appear here")

with tab2:
    # Create two columns for input and output
    col1, col2 = st.columns(2)

    with col1:
        # Input section
        st.subheader("Input")
        
        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Display uploaded image
            init_image = Image.open(uploaded_file).convert("RGB").resize((512, 512))
            st.image(init_image, caption="Uploaded Image", use_column_width=True)
            
            # Prompt input
            prompt = st.text_area(
                "Enter your prompt",
                placeholder="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
                height=100,
                key="img2img_prompt"
            )
            
            # Strength slider
            strength = st.slider(
                "Transformation strength",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher values will transform the image more, lower values will preserve more of the original image"
            )
            
            # Generate button
            if st.button("Transform Image", type="primary", key="img2img_btn"):
                if not prompt:
                    st.error("Please enter a prompt")
                else:
                    with st.spinner("Transforming image..."):
                        try:
                            # Generate image
                            image = pipe_image2image(
                                prompt=prompt,
                                image=init_image,
                                strength=strength,
                                guidance_scale=0.0,
                                num_inference_steps=2
                            ).images[0]
                            
                            # Display the image
                            st.session_state['transformed_image'] = image
                            
                        except Exception as e:
                            st.error(f"Error transforming image: {str(e)}")

    with col2:
        # Output section
        st.subheader("Transformed Image")
        
        if 'transformed_image' in st.session_state:
            # Display the transformed image
            st.image(st.session_state['transformed_image'], use_column_width=True)
            
            # Download button
            buf = io.BytesIO()
            st.session_state['transformed_image'].save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            href = f'data:file/png;base64,{img_str}'
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="transformed_image.png",
                mime="image/png",
                key="img2img_download"
            )
        else:
            st.info("Your transformed image will appear here")

# Add some helpful information
with st.expander("Tips for better results"):
    st.markdown("""
    ### Text to Image Tips:
    - Be specific in your prompts
    - Use descriptive adjectives
    - Mention the style you want (e.g., photorealistic, anime, oil painting)
    - Use negative prompts to avoid unwanted elements
    
    ### Image to Image Tips:
    - Start with a clear, high-quality image
    - Use a lower strength value (0.3-0.5) to preserve more of the original image
    - Use a higher strength value (0.6-0.8) for more dramatic transformations
    - The model is optimized for speed, so results may vary
    """) 