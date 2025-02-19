from streamlit_utilities.utils_streamlit import *

st.set_page_config(page_title="XRAY Helper", layout="centered")

# ---- Cargar modelos ----

yolo_model = load_yolo_model()
classification_model = load_classification_model()

# PANTALLAS
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "selected_tool" not in st.session_state:
    st.session_state.selected_tool = None

# LOGIN
if not st.session_state.logged_in:
    st.image('streamlit_utilities/TRiAPPGe Logo.png', width=300)
    st.title("Login")
    
    email = st.text_input("Email", placeholder="Enter your email")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    
    if st.button("Login", use_container_width=True):
        if email and password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.warning("⚠ Please enter both email and password.")
# SELECCIÓN DE HERRAMIENTA
elif not st.session_state.selected_tool:
    st.image('streamlit_utilities/TRiAPPGe Logo.png', width=300)
    st.title("SELECT TOOL")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("XRAY Helper", use_container_width=True, help="Analyze X-ray images", key="xray_btn", 
                     type="secondary"):
            st.session_state.selected_tool = "XRAY"
            st.rerun()
    with col2:
        if st.button("SCAN Helper", use_container_width=True, help="Scan and detect fractures", key="scan_btn",
                     type="secondary"):
            st.session_state.selected_tool = "SCAN"
            st.rerun()

# ---- Pantalla de XRAY Helper ----
elif st.session_state.selected_tool == "XRAY":
    st.image('streamlit_utilities/TRiAPPGe Logo.png', width=300)
    st.title("XRAY Helper")

    uploaded_files = st.file_uploader("Upload X-ray images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if st.button("Predict"):
        if uploaded_files:
            target_array = preprocess_images_streamlit(uploaded_files)  # Preprocesamiento de imágenes (mismo preprocesamiento que para el modelo)
            predictions = classification_model.predict(target_array)  # Predicción con modelo de fracturas
            predicted_classes = (predictions > 0.5).astype(int)

            labels = {0: 'Non Fractured', 1: 'Fractured'}

            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)

                # Detección de fracturas con YOLO
                results = yolo_model.predict(image_np)
                processed_img = draw_detections(image_np, results)  # Superponer detecciones

                # Mostramos Resultados
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(processed_img, caption="Fracture Detection", use_container_width=True)

                st.subheader(f"Prediction: {labels[predicted_classes[i][0]]}")
                st.progress(float(predictions[i][0]))
                st.write(f"Fracture Probability: {predictions[i][0]:.2%}")

        else:
            st.warning("⚠ Please upload an X-ray image.")

