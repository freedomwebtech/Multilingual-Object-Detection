import os
import cv2
import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock

# Define language options
language_options = {
     "English": "Provide the response in English.",
    "Hindi": "Provide the response in Hindi.",
    "Marathi": "Provide the response in Marathi.",
    "Spanish": "Provide the response in Spanish.",
    "French": "Provide the response in French.",
    "German": "Provide the response in German.",
    "Chinese (Simplified)": "Provide the response in Chinese (Simplified).",
    "Chinese (Traditional)": "Provide the response in Chinese (Traditional).",
    "Japanese": "Provide the response in Japanese.",
    "Korean": "Provide the response in Korean.",
    "Russian": "Provide the response in Russian.",
    "Portuguese": "Provide the response in Portuguese.",
    "Arabic": "Provide the response in Arabic.",
    "Italian": "Provide the response in Italian.",
    "Dutch": "Provide the response in Dutch.",
    "Turkish": "Provide the response in Turkish.",
    "Swedish": "Provide the response in Swedish.",
    "Danish": "Provide the response in Danish.",
    "Finnish": "Provide the response in Finnish.",
    "Norwegian": "Provide the response in Norwegian.",
    "Greek": "Provide the response in Greek.",
    "Hebrew": "Provide the response in Hebrew.",
    "Thai": "Provide the response in Thai.",
    "Polish": "Provide the response in Polish.",
    "Ukrainian": "Provide the response in Ukrainian.",
    "Czech": "Provide the response in Czech.",
    "Hungarian": "Provide the response in Hungarian.",
    "Romanian": "Provide the response in Romanian.",
    "Indonesian": "Provide the response in Indonesian.",
    "Malay": "Provide the response in Malay.",
    "Filipino (Tagalog)": "Provide the response in Filipino (Tagalog).",
    "Vietnamese": "Provide the response in Vietnamese.",
    "Bengali": "Provide the response in Bengali.",
    "Tamil": "Provide the response in Tamil.",
    "Telugu": "Provide the response in Telugu.",
    "Urdu": "Provide the response in Urdu.",
    "Gujarati": "Provide the response in Gujarati.",
    "Punjabi": "Provide the response in Punjabi.",
    "Swahili": "Provide the response in Swahili.",
    "Afrikaans": "Provide the response in Afrikaans.",
    "Serbian": "Provide the response in Serbian.",
    "Croatian": "Provide the response in Croatian.",
    "Slovak": "Provide the response in Slovak.",
    "Lithuanian": "Provide the response in Lithuanian.",
    "Latvian": "Provide the response in Latvian.",
    "Estonian": "Provide the response in Estonian.",
    "Macedonian": "Provide the response in Macedonian.",
    "Bosnian": "Provide the response in Bosnian.",
    "Albanian": "Provide the response in Albanian.",
    "Georgian": "Provide the response in Georgian.",
    "Armenian": "Provide the response in Armenian.",
    "Maltese": "Provide the response in Maltese.",
    "Icelandic": "Provide the response in Icelandic.",
    "Welsh": "Provide the response in Welsh.",
    "Irish": "Provide the response in Irish.",
    "Basque": "Provide the response in Basque.",
    "Galician": "Provide the response in Galician.",
    "Malayalam": "Provide the response in Malayalam.",
    "Sanskrit": "Provide the response in Sanskrit.",
    "Kannada": "Provide the response in Kannada."
}

# Streamlit UI
st.title("🎥 Live Object Detection with Multilingual Support")
st.write("Live webcam feed. Capture an image and get object detection results in your selected language.")

# Language selection
selected_language = st.selectbox("🌍 Select Language", list(language_options.keys()))

# OpenCV live webcam feed
stframe = st.empty()  # Placeholder for live webcam feed
cap = cv2.VideoCapture(0)  # Open webcam

if not cap.isOpened():
    st.error("Error: Could not open webcam.")

# UI for capturing and closing image preview
capture_button = st.button("📸 Capture Image")
close_preview_button = st.button("❌ Close Image Preview", key="close")

captured_image = None  # Variable to store captured image
preview_mode = False   # Toggle to show or hide captured image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to grab frame from webcam.")
        break

    # Convert frame from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # If image is not captured, show live feed
    if not preview_mode:
        stframe.image(frame, channels="RGB", use_container_width=True)

    # If capture button is clicked
    if capture_button and not preview_mode:
        captured_image = frame  # Store the captured frame
        preview_mode = True  # Show captured image instead of live feed

    # Exit loop when the image is captured
    if preview_mode:
        break

# Release webcam but only when closing app
if not preview_mode:
    cap.release()
    cv2.destroyAllWindows()

# If image is captured, process it
if preview_mode and captured_image is not None:
    # Save the captured image
    temp_image_path = "captured_image.jpg"
    cv2.imwrite(temp_image_path, cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR))

    # Display the captured image with a close button
    st.image(temp_image_path, caption="Captured Image", use_container_width=True)

    # If close preview button is clicked, return to live feed
    if close_preview_button:
        preview_mode = False
        st.rerun()  # Restart script to show live feed again

    # Resize image
    image_resized = cv2.resize(captured_image, (1020, 600))
    resized_image_path = "resized_image.jpg"
    cv2.imwrite(resized_image_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))

    # Set API Keys
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCkBApCCKznY1OjcSqcvCTMj3ZFXiTR4UI"
    os.environ["DEEPSEEK_API_KEY"] = "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"

    # Initialize Models
    gemini_pro = Gemini(model_name="models/gemini-1.5-flash")
    llm = Groq(model="deepseek-r1-distill-llama-70b", api_key=os.environ["DEEPSEEK_API_KEY"])

    # Step 1: Send Image + Prompt to Gemini
    msg_gemini = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            ImageBlock(path=resized_image_path, image_mimetype="image/jpeg"),
            TextBlock(text="Detect objects in this image")
        ]
    )

    response_gemini = gemini_pro.chat(messages=[msg_gemini])

    # Extract detected objects text
    if response_gemini and response_gemini.message:
        detected_objects = " ".join(
            block.text for block in response_gemini.message.blocks if hasattr(block, "text")
        ).strip()

        # Step 2: Send Gemini's Output to DeepSeek-R1 for Enhancement
        system_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=language_options[selected_language]
        )

        msg_deepseek = ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text=f"Enhance this object detection data. Provide full details of all objects in {selected_language}: {detected_objects}")]
        )

        response_llm = llm.chat(messages=[system_message, msg_deepseek])

        # Display Final Enhanced Response
        if response_llm and response_llm.message:
            enhanced_response = " ".join(
                block.text for block in response_llm.message.blocks if hasattr(block, "text")
            ).strip()
            st.write(f"**Enhanced Response in {selected_language}:**")
            st.write(enhanced_response)
        else:
            st.error("No response from DeepSeek-R1.")
    else:
        st.error("Gemini did not return a valid response.")
