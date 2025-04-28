import streamlit as st

from dotenv import load_dotenv
from db import insert_user, insert_chatbot, insert_conversation, get_conversations

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
			st.markdown("______________________________________________")
	
	convo_pages = []
	
	page_to_add = st.Page(convert_to_page, title = "Conversation History")
	convo_pages.insert(0, page_to_add)
	convo_pages.insert(0, "chatbot_page.py")
	convo_pages.insert(0, landing)
	pages = st.navigation(convo_pages)

	pages.run()

# Probably some help from ChatGPT but don't remember where exactly