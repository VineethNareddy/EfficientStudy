import psycopg2
import os

from dotenv import load_dotenv

def insert_user(username):
    load_dotenv()
    conn = psycopg2.connect(
        host = os.getenv("HOST"),
        database = os.getenv("DATABASE"),
        user = os.getenv("USER"),
        password = os.getenv("PASSWORD")
    )
    cur = conn.cursor()

    cur.execute("SET search_path TO efficient_study_database;")

    cur.execute(""" 
        INSERT INTO users (username)
        VALUES (%s)
    """, (username,))

    cur.execute("""SELECT id FROM users WHERE username = %s""", (username, ))
    user_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    return user_id


def insert_chatbot():
    load_dotenv()
    conn = psycopg2.connect(
        host = os.getenv("HOST"),
        database = os.getenv("DATABASE"),
        user = os.getenv("USER"),
        password = os.getenv("PASSWORD")
    )
    cur = conn.cursor()

    cur.execute("SET search_path TO efficient_study_database;")

    cur.execute(""" 
        INSERT INTO chatbots DEFAULT VALUES
    """)

    cur.execute("""
        SELECT id FROM CHATBOTS ORDER BY id DESC LIMIT 1
    """)
    chatbot_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    return chatbot_id

def insert_conservations(user_id, chat_id):
    load_dotenv()
    conn = psycopg2.connect(
        host = os.getenv("HOST"),
        database = os.getenv("DATABASE"),
        user = os.getenv("USER"),
        password = os.getenv("PASSWORD")
    )
    cur = conn.cursor()

    cur.execute("SET search_path TO efficient_study_database;")

    cur.execute("""
        INSERT INTO conversations (user_id, chatbot_id)
        VALUES (%s, %s)
    """, (user_id, chat_id))

    cur.execute("""
        SELECT id FROM conversations ORDER BY id DESC LIMIT 1
    """)
    conversation_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()

    return conversation_id


def insert_chatbot_messages(chatbot_id, conversation_id, message):
    load_dotenv()
    conn = psycopg2.connect(
        host = os.getenv("HOST"),
        database = os.getenv("DATABASE"),
        user = os.getenv("USER"),
        password = os.getenv("PASSWORD")
    )
    cur = conn.cursor()

    cur.execute("SET search_path TO efficient_study_database;")

    cur.execute(""" 
        INSERT INTO chatbot_messages (chatbot_id, conversation_id, message)
        VALUES (%s, %s, %s)
    """, (chatbot_id, conversation_id, message))

    conn.commit()
    cur.close()
    conn.close()

def insert_user_messages(user_id, conversation_id, message):
    load_dotenv()
    conn = psycopg2.connect(
        host = os.getenv("HOST"),
        database = os.getenv("DATABASE"),
        user = os.getenv("USER"),
        password = os.getenv("PASSWORD")
    )
    cur = conn.cursor()

    cur.execute("SET search_path TO efficient_study_database;")

    cur.execute(""" 
        INSERT INTO user_messages (user_id, conversation_id, message)
        VALUES (%s, %s, %s)
    """, (user_id, conversation_id, message))

    conn.commit()
    cur.close()
    conn.close()


