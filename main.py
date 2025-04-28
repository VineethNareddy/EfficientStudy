import streamlit as st

from dotenv import load_dotenv
from db import insert_user, insert_chatbot, insert_conversation, get_conversations

st.set_page_config(
    page_title="Efficient Study",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --sidebar-bg: #0B0E13;   /* darkest */
  --main-bg:    #161B22;   /* lighter than sidebar */
  --card-bg:    #1E242D;   /* file‑uploader & chat bubbles */
  --border:     #2C313C;
  --text:       #F5F7FA;
  --accent:     #B3A369;   /* GT gold */
  --muted:      #9CA3AF;
}


html, body, [class*='stApp'], .block-container {
    background-color:var(--main-bg);
    color:var(--text);
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}


/* Sidebar */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
    background-color:var(--sidebar-bg) !important;
    color:var(--text) !important;
}


/* File‑uploader card */
div[data-testid="stFileUploader"] > div {
    background-color:var(--card-bg) !important;
    border:1px solid var(--border) !important;
}


/* Chat bubbles */
.chat-bubble {
    max-width: 80%;
    margin: .25rem 0;
    padding: .75rem 1rem;
    border-radius: 1rem;
    line-height: 1.4;
    font-size: 0.95rem;
    border:1px solid var(--border);
    background:var(--card-bg);
    color:var(--text);
}


.user-bubble {border-color:#444B57;}
.assistant-bubble {border-color:var(--border);}


/* Buttons */
.css-9ycgxx, .stButton>button {
    background-color:var(--accent);
    color:var(--sidebar-bg);
    border:none;
    border-radius:6px;
}


.stButton>button:hover {background-color:#C8B87A;}
button:focus {outline:none;}


/* Placeholder & secondary text */
span, p, label {color:var(--text);}  /* ensure forms stay bright */
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.image("gtlogo.png", use_container_width=True)
    # pdf_file = st.file_uploader("", type="pdf")
    st.markdown("---")

def user_form():
	with st.form('userform', clear_on_submit=True):
		load_dotenv()
		username = st.text_input('Username:', placeholder = 'Please enter your username here')
		submitted = st.form_submit_button('Submit')
	if submitted:
		user_id = insert_user(username)
		chatbot_id = insert_chatbot()
		conversation_id = insert_conversation(user_id, chatbot_id)
		if user_id not in st.session_state:
			st.session_state["user_id"] = user_id
		if chatbot_id not in st.session_state:
			st.session_state["chatbot_id"] = chatbot_id
		if conversation_id not in st.session_state:
			st.session_state["conversation_id"] = conversation_id

if __name__ == "__main__":

	convos = get_conversations()

	def landing():
		st.title("User Sign In")
		user_form()

	pages_map = {}
	for convo in convos:
		key = convo[0]
		if key not in pages_map:
			pages_map[key] = []
		pages_map[key].append((convo[1], convo[2]))
	
	def convert_to_page():
		for key in pages_map.keys():
			st.markdown("Conversation " + str(key))
			for line in pages_map[key]:
				st.markdown(line[0] + ": " + line[1])
			st.markdown("---")
	
	convo_pages = []
	
	page_to_add = st.Page(convert_to_page, title = "conversation history")
	convo_pages.insert(0, page_to_add)
	convo_pages.insert(0, "chatbot_page.py")
	convo_pages.insert(0, landing)
	pages = st.navigation(convo_pages)

	pages.run()

# Probably some help from ChatGPT but don't remember where exactly