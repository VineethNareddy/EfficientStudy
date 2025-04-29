

import random
from datetime import datetime
from db import (
    insert_user,
    insert_chatbot,
    insert_conversation,
    insert_user_messages,
    insert_chatbot_messages,
)

# ------------------------------------------------------------------
# Tunables ─ change these if you want more / fewer records
NUM_USERS        = 3          # how many fake human users
NUM_CHATBOTS     = 2          # how many chatbots
CONV_PER_USER    = 2          # conversations each user has with a random bot
MSG_PER_CONV     = 6          # total messages per conversation (user+bot)
# ------------------------------------------------------------------

USERNAMES = [
    "Seb",
    "Jonathan",
    "Will",
    "vineeth",
    "Sebastian",
]

BOT_PERSONAS = [
    "GPT1",
    "QGPT2",
    "AI",
    "Flashcard",
]

USER_LINES = [
    "Hey, can you help me review Chapter 4?",
    "What does this SQL error actually mean?",
    "Got any tips for staying focused?",
    "Let’s run some practice questions.",
    "Explain heteroskedasticity like I'm five.",
]

BOT_LINES = [
    "Sure! First, let's outline the key concepts.",
    "Here’s a quick query that should fix it:",
    "Absolutely. Pomodoro technique works great!",
    "Question 1: What is the null hypothesis?",
    "In simple terms, it just means the variance isn’t constant.",
]

def random_choice(seq):
    return random.choice(seq)

def main():
    random.seed()  

    #Create users
    user_ids = []
    for i in range(NUM_USERS):
        uname = random_choice(USERNAMES) + f"_{random.randint(1000,9999)}"
        uid = insert_user(uname)
        user_ids.append(uid)
        print(f"  ✔ user '{uname}' inserted as id={uid}")

    #Create chatbots
    bot_ids = []
    for i in range(NUM_CHATBOTS):
        bid = insert_chatbot()
        bot_ids.append(bid)
        print(f"  ✔ chatbot id={bid}")

    #Create conversations & messages
    print("\nCreating conversations and messages …")
    for uid in user_ids:
        for _ in range(CONV_PER_USER):
            bid = random_choice(bot_ids)
            conv_id = insert_conversation(uid, bid)
            print(f"  ✔ conversation id={conv_id} (user {uid} ⇄ bot {bid})")

            # alternate sender: user ➜ bot ➜ user ➜ …
            for m in range(MSG_PER_CONV):
                if m % 2 == 0:   # user turn
                    msg = random_choice(USER_LINES)
                    insert_user_messages(uid, conv_id, msg)
                else:            # bot turn
                    msg = random_choice(BOT_LINES)
                    insert_chatbot_messages(bid, conv_id, msg)

    print("\n Random seed data inserted successfully!\n")

if __name__ == "__main__":
    main()
