import streamlit as st
import requests
import json
import time
from datetime import datetime
import logging
from streamlit_float import *
from deep_translator import GoogleTranslator
from streamlit_extras.stylable_container import stylable_container
from streamlit_mic_recorder import speech_to_text
from streamlit_geolocation import streamlit_geolocation
import streamlit_js_eval
import folium
from streamlit_folium import st_folium

logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

language_mapping = {
    "English": "en",
    "Spanish": "es",
    "Vietnamese": "vi",
    "Russian": "ru",
    "Arabic": "ar",
    "Chinese": "zh-CN",
    "German": "de",
    "Japanese": "ja",
    "Hindi": "hi",
    "Korean": "ko"
}
float_init(theme=True, include_unstable_primary=False)

location_string = streamlit_js_eval.get_geolocation() 

def prompt_location():
    latitude = ""
    longitude = ""

    if location_string is None:
        st.toast('Please allow location access and reload the page for this feature 🗺️')
        return None
    else:
        latitude = location_string["coords"]["latitude"]
        longitude = location_string["coords"]["longitude"]
        return [latitude, longitude]


if "language" not in st.session_state:
    st.session_state.language = "English"

def _(content, target_lang=language_mapping[st.session_state.language]):
    return GoogleTranslator(source='auto', target=target_lang).translate(content)

# Sidebar for language selection
st.sidebar.title(_("Language Selector"))
selected_language = st.sidebar.selectbox(
    _("Choose a language to translate"), 
    ["English", "Spanish", "Vietnamese", "Russian", "Arabic", "Chinese", "German", "Japanese", "Hindi", "Korean"],  # Language names
    index=["English", "Spanish", "Vietnamese", "Russian", "Arabic", "Chinese", "German", "Japanese", "Hindi", "Korean"].index(st.session_state.language),  # Match the name
    key="language"
)

# Update language in session state
if st.session_state.language != selected_language:
    st.session_state.language = selected_language
    st.experimental_rerun()
    
# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

mic_buttons_container = st.container()
mic_buttons_container.float("bottom: 3rem;")

col7, col8, col9 = mic_buttons_container.columns([0.8, 0.1, 0.1], gap="small")
with col7:
    prompt = st.chat_input(placeholder='What can I help you today?', key=1) 
with col8:
    text = speech_to_text(
        language=language_mapping[st.session_state.language], 
        use_container_width=True, 
        just_once=True, 
        key='STT',
        start_prompt="🎤", 
        stop_prompt="🗣️"
    )

    if text:
        prompt = text
with col9:
    icon = "📍"

    # The button will trigger the logging function
    if st.button(icon):
        location_return = prompt_location()

        if location_return is not None:
            location_value = 'latitude: ' + str(location_return[0]) + ' , longitude: ' + str(location_return[1])
            prompt = "My location is " + location_value + ", find attrations near me"
        

caption_container = st.container()
caption_container.float("bottom: 1rem")
caption_container.caption("<div style='text-align: center; margin-bottom: 1rem'>" + _('Ragooon can make mistakes. Check important info.') +  "</div>", unsafe_allow_html=True)

# st.caption("<div style='text-align: center; margin-bottom: 1rem'>" + _('Ragooon can make mistakes. Check important info.') +  "</div>", unsafe_allow_html=True)

chat_container = st.container()
chat_container.float("top: 5rem; height: 71vh; overflow-y: scroll; overflow-x: hidden; padding-bottom: 5vh; padding-right: 1vh")
with chat_container:
    st.title("❄️ Ragoon ❄️")
    st.subheader(_("Your personal pocket guide dog 🦮"))

    st.write(_(
        "A guide dog just for you, talk to Ragoon now! 💬 "
    ))

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})

        # st.markdown(
        #     """
        # <style>
        #     .st-emotion-cache-janbn0 {
        #         flex-direction: row-reverse;
        #         text-align: right;
        #     }
        # </style>
        # """,
        #     unsafe_allow_html=True,
        # )

        with st.chat_message("user"):
            st.markdown(prompt)

        url = 'https://ragoooon.onrender.com/stream_rag'
        myobj = {"prompts": prompt, "history": st.session_state.messages}
        with st.spinner('Ragoon ' + _('is finding the way...')):
            stream = requests.post(url, json = myobj)
        
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        
        def stream_data():
            for word in stream.json()['stream']:
                yield word
                time.sleep(0.01)

        with st.chat_message("assistant"):
            with stylable_container(
                "codeblock",
                """
                code {
                    white-space: pre-wrap !important;
                }
                """,
            ):
                response = st.write_stream(stream_data)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# This function logs the last question and answer in the chat messages
def log_feedback(icon):
    # We display a nice toast
    st.toast("Thanks for your feedback!", icon=icon)

    # We retrieve the last question and answer
    last_messages = json.dumps(st.session_state["messages"][-2:])

    # We record the timestamp
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "

    # And include the messages
    activity += "positive" if icon == "👍" else "negative"
    activity += ": " + last_messages

    # And log everything
    logger.info(activity)

st.write("")
# If there is at least one message in the chat, we display the options
if len(st.session_state["messages"]) > 0:
    action_buttons_container = st.container()
    action_buttons_container.float("bottom: 6.9rem;background-color: var(--default-backgroundColor); padding-top: 1rem;")

    # We set the space between the icons thanks to a share of 100
    cols_dimensions = [7, 19.4, 19.3, 9, 8.6, 8.6, 28.1] # add column 28.1
    col0, col1, col2, col3, col4, col5, col6 = action_buttons_container.columns(cols_dimensions)

    with col1:
        # Converts the list of messages into a JSON Bytes format
        json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")

        # And the corresponding Download button
        st.download_button(
            label="📥 Save chat!",
            data=json_messages,
            file_name="chat_conversation.json",
            mime="application/json",
        )

    with col2:
        # We set the message back to 0 and rerun the app
        # (this part could probably be improved with the cache option)
        if st.button("Clear Chat 🧹"):
            st.session_state["messages"] = []
            st.rerun()

    with col3:
        icon = "🔁"
        if st.button(icon):
            st.session_state["rerun"] = True
            st.rerun()

    with col4:
        icon = "👍"

        # The button will trigger the logging function
        if st.button(icon):
            log_feedback(icon)

    with col5:
        icon = "👎"

        # The button will trigger the logging function
        if st.button(icon):
            log_feedback(icon)

    with col6:
        location_return_1 = prompt_location()

        with st.popover("Open map 🗺️", use_container_width=False):
            if location_return_1 is not None:
                latitude = location_return_1[0]
                longitude = location_return_1[1]

                m = folium.Map(location=[latitude, longitude], zoom_start=32)
                folium.Marker(
                    [latitude, longitude], popup="Your location", tooltip="Your location"
                ).add_to(m)

                # call to render Folium map in Streamlit
                st_data = st_folium(m, width=1000)