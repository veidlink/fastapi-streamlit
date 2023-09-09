import requests
import numpy as np
import json
import cv2
import streamlit as st
from PIL import Image
from io import BytesIO


def main():

    st.title("Traffic detection")
        
    source_image = st.file_uploader("Upload an image")
    image_url = st.text_input("##### Or pass URL", "")

    if source_image is not None:  # Проверка, загружено ли изображение
        source_image = Image.open(source_image)
        source_image = source_image.convert('RGB') 
    elif image_url != "":  # Проверка, был ли введен URL
        # Download the image from the URL
        response = requests.get(image_url)
        source_image = Image.open(BytesIO(response.content))
        source_image = source_image.convert('RGB')
    else:
        st.write("### You haven't provided us with a picture yet")

    col1, col2 = st.columns(2)
    if source_image:
        with col1:
            try:
                st.image(source_image,
                        caption="Source",
                        )
            except Exception as ex:
                st.error(
                    f"Upload a picture")
        with col2:
            try:
                # Convert PIL image to BytesIO
                image_buffer = BytesIO()
                source_image.save(image_buffer, format='JPEG')
                # Reset the buffer's file pointer
                image_buffer.seek(0)
                # Create the files dictionary with the image buffer
                file = {"file": image_buffer}
                response = requests.post("http://0.0.0.0:8000/detect", files=file)
                bboxNlabel = json.loads(response.text)['data']
                print(bboxNlabel)

                source_image = np.array(source_image)
                for x in bboxNlabel:
                    image = cv2.rectangle(source_image, x[0:2], x[2:4], (0, 255, 0), 2)
                    for lst in bboxNlabel:
                        image = cv2.putText(image, lst[4], (lst[0], lst[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                st.image(image,
                        caption="Detection",
                        )
            except Exception as ex:
                st.error(
                    "Unable to detect")
                st.error(ex)

if __name__ == '__main__':
    main()
