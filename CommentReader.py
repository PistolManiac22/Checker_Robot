import sys

import google.generativeai as genai # gemini
from gtts import gTTS # text-to-speech by google
import logging
import io
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # hide pygame startup welcome message
import pygame

# setting up the model
# the way it works is a constant chat session where we keep sending gemini input
genai.configure(api_key="")

# Create the model
generation_config = {
    "temperature": 0.55, # determinism of the model (0 to 2.0)
    "top_p": 0.95, # degree of creativity (0 to 1.0)
    "top_k": 40, # size of output vocabulary
    "max_output_tokens": 2000,
    "response_mime_type": "text/plain",
}

# generating the model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# start the session
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "\"You are assisting a player in a game of checkers. The player you are helping is playing against a you. Your role is only to analyze and provide friendly, supportive feedback on moves or board states given to you.  \n\n**Important Rules:**  \n1. The term **\"Opponent\"** refers to the HUMAN PLAYER (your user), not you. You will need to refer to them as 'YOU', to make it sound like you have a .  \n2. NEVER create or infer moves on your own. Only comment on moves explicitly provided to you in the input.  \n3. If you are given two board states, you may describe what has changed between them and provide constructive insights.  \n4. Your tone should be positive, fun, and encouraging. Congratulate the player on strong moves and progress.  \n5. You are still playing the game against your HUMAN opponent (your user)   \n6. Be the supportive checkers mentor of your opponent.\n\n**Example Input 1:**  \nOpponent did this move: \"Move piece from (2,3) to (3,4).\"  \n\n**Example Response 1:**  \n\"Great move! You advanced your piece in the middle of the board. This puts pressure on the center of the board—nicely done!\"  \n\nRemember: ONLY analyze the moves and board states given to you. NEVER create your own moves or suggestions. Keep your answer brief, about 20 words. Limit the use of 'Okay' in your response and NEVER use diagonially or diagonal. Do NEVER use quotation marks in your response.\"\n",
            ],
        },

        {
            "role": "model",
            "parts": [
                "Okay, I understand! I'm ready to be your supportive checkers mentor. Let's play and have some fun! I'll be here to cheer you on and analyze your moves. Bring on the board states and moves! Let's see what you've got!\n",
            ],
        },
    ]
)


class CommentReader:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.clock = pygame.time.Clock()
        self.talking_thread = None
        self.clock = pygame.time.Clock()
        pygame.event.timeout_sec = 5

    def generate_and_read_comment(self, prompt):
        comment = self.llm_generate_response(prompt)
        logging.info(f"Generated response: {comment}")
        self.read_text_simple(comment)

    def llm_generate_response(self, prompt):
        print(prompt)
        response = chat_session.send_message(prompt)
        print(response.text)
        return response.text

    def read_text(self, text):
        tts = gTTS(text=text, lang='en', slow=False, tld='co.uk')
        stream = tts.stream()
        idx = 0
        #chunk_count = sum(1 for _ in stream)  # Count the total chunks first
        for chunk in stream:
            if idx == 0:  # play first chunk right away
                mp3_fp = self.get_mp3_fp(chunk)
                pygame.mixer.music.set_endevent(pygame.USEREVENT + 1)
                pygame.mixer.music.load(mp3_fp, 'mp3')
                pygame.mixer.music.play()
            elif idx == 1: #and chunk_count > 1:  # put second chunk in queue
                mp3_fp = self.get_mp3_fp(chunk)
                pygame.mixer.music.queue(mp3_fp, 'mp3')
            else:  # wait for finishing of previous chunk and put new chunk in queue
                mp3_fp = self.get_mp3_fp(chunk)
                stuck_cnt = 0
                while True and stuck_cnt < 10000:
                    if pygame.event.peek(pygame.USEREVENT + idx - 1):
                        pygame.event.get(pygame.USEREVENT + idx - 1)  # remove event from queue
                        pygame.mixer.music.set_endevent(pygame.USEREVENT + idx)
                        break
                    self.clock.tick(1)
                    stuck_cnt += 1
                pygame.mixer.music.queue(mp3_fp, 'mp3')
            idx += 1
        while pygame.mixer.music.get_busy():  # wait for reading end
            self.clock.tick(1)
        pygame.event.get(pygame.USEREVENT + idx - 1)  # remove last event from queue

    def read_text_simple(self, text):
        tts = gTTS(text=text, lang='en', slow=False, tld='co.uk')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        logging.info("Sound loaded into fp")
        pygame.mixer.music.load(mp3_fp, 'mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # wait for reading end
            self.clock.tick(1)

    def get_mp3_fp(self, chunk):
        mp3_fp = io.BytesIO()
        mp3_fp.write(chunk)
        mp3_fp.seek(0)
        return mp3_fp

def main(args=None):
    logging.basicConfig(level=logging.INFO)
    if args is None or len(args) <= 1:
        logging.error("Please provide a prompt for the comment")
        return
    prompt = args[1]
    comment_reader = CommentReader()
    comment_reader.generate_and_read_comment(prompt)

if __name__ == "__main__":
    main(sys.argv)
