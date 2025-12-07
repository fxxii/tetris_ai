import base64
import random
import sys
import time
import os
from collections import deque
from io import BytesIO
from httpx import Timeout

import mss
import ollama
import pyautogui
from loguru import logger
from PIL import Image

# ---------------------------
# Logging
# ---------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
)

# --- Constants ---

MODEL_NAME = "maternion/fara:7b"
#MODEL_NAME = "llama3.2-vision"
IMAGE_DIR = "screencap"
RESIZE_FACTOR = 0.5  # Resize to 50% of the original size


# --- Reinforcement Learning Components ---

class ReplayBuffer:
    """A simple replay buffer to store experiences."""
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size=5):
        """Sample a batch of experiences, prioritizing high and low rewards."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        sorted_buffer = sorted(self.buffer, key=lambda x: x[1], reverse=True)
        
        half_batch = batch_size // 2
        top = sorted_buffer[:half_batch]
        bottom = sorted_buffer[-half_batch:]
        
        return top + bottom

# --- Game Interaction and AI Logic ---

def capture_screenshot(monitor, save_path=None):
    """
    Captures a screenshot, resizes it, and returns the raw bytes of the resized image.
    Optionally saves the image to a file.
    """
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)

        # Convert to a Pillow Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

        # Resize the image
        new_width = int(img.width * RESIZE_FACTOR)
        new_height = int(img.height * RESIZE_FACTOR)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        logger.debug(f"Resized screenshot from {img.width}x{img.height} to {new_width}x{new_height}")

        # Save the image if a path is provided
        if save_path:
            try:
                resized_img.save(save_path)
                logger.debug(f"Saved screenshot to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save screenshot to {save_path}: {e}")

        # Convert back to bytes in PNG format
        buffer = BytesIO()
        resized_img.save(buffer, format="PNG")
        return buffer.getvalue()

def get_game_score(monitor_score):
    """
    Captures a screenshot of the score region and uses the Fara model to extract the score.
    """
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(IMAGE_DIR, "score", f"score_{timestamp}.png")
        score_image_data = capture_screenshot(monitor_score, save_path=save_path)
        encoded_image = base64.b64encode(score_image_data).decode('utf-8')

        prompt = "Analyze the attached image and return ONLY the numerical score shown. For example, if the score is '12345', your response should be just '12345'."
        client = ollama.Client(
            timeout=Timeout(60)   # <-- timeout in seconds
        )
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [encoded_image],
                },
            ],
        )

        # Extract numbers from the response and convert to an integer
        score_text = "".join(filter(str.isdigit, response['message']['content']))
        if score_text:
            logger.debug(f"Successfully parsed score: {score_text}")
            return int(score_text)
        else:
            logger.warning(f"Fara could not read the score. Response: '{response['message']['content']}'")
            return 0
    except Exception as e:
        logger.error(f"Error reading score with Fara: {e}")
        return 0

def send_to_fara(image_data, experiences):
    """Sends the screenshot and past experiences to the Fara model."""
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    experience_prompt = "Here are some recent experiences (action, reward):\n"
    if experiences:
        for action, reward, _ in experiences:
            outcome = "good" if reward > 0 else "bad" if reward < 0 else "neutral"
            experience_prompt += f"- Move '{action}' was {outcome} (reward: {reward})\n"
    else:
        experience_prompt += "- No recent experiences.\n"

    prompt = (
        "You are a world-class Tetris AI. Your goal is to achieve the highest score possible.\n"
        "Analyze the attached image of the game board. Based on the position of the falling piece "
        "and the existing blocks, determine the single best action to take.\n\n"
        "Your response MUST be one of the following keywords: 'left', 'right', 'up' (for rotate), or 'down'.\n\n"
        f"{experience_prompt}"
    )

    client = ollama.Client(
            timeout=Timeout(60)   # <-- timeout in seconds
    )
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [encoded_image],
            },
        ],
    )
    return response['message']['content']

def parse_action(response):
    """Parses the model's response to determine the next action."""
    response_lower = response.lower()
    if "left" in response_lower:
        return "left"
    elif "right" in response_lower:
        return "right"
    elif "rotate" in response_lower or "up" in response_lower:
        return "up"
    elif "down" in response_lower:
        return "down"
    return None

def execute_action(action):
    """Executes the specified action by pressing the corresponding key."""
    if action:
        logger.info(f"Executing action: {action}")
        pyautogui.press(action)
    else:
        logger.warning("No action determined from model response.")

def main():
    """The main loop for the Tetris AI with reinforcement learning."""
    
    # --- Directory Setup ---
    os.makedirs(os.path.join(IMAGE_DIR, "game"), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DIR, "score"), exist_ok=True)
    logger.info(f"Saving screenshots to '{IMAGE_DIR}/' directory.")

    # --- CRITICAL CONFIGURATION ---
    # You MUST update these pixel values to match the location of the
    # game and score on YOUR screen. Use a screenshot tool to find the
    # correct coordinates.
    monitor_game = {"top": 145, "left": 125, "width": 790, "height": 850}
    monitor_score = {"top": 465, "left": 240, "width": 100, "height": 30}

    # RL parameters
    epsilon = 0.1
    replay_buffer = ReplayBuffer(max_size=1000)
    possible_actions = ["left", "right", "up", "down"]

    logger.info("Starting Tetris AI in 5 seconds... Click on the game window!")
    time.sleep(5)
    logger.info("AI active.")

    last_score = get_game_score(monitor_score)
    logger.info(f"Initial score detected: {last_score}")

    turn_number = 0
    while True:
        turn_number += 1
        logger.debug(f"--- Turn {turn_number} ---")

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(IMAGE_DIR, "game", f"turn_{turn_number}_{timestamp}.png")
        logger.debug("Capturing and resizing screenshot...")
        screenshot_data = capture_screenshot(monitor_game, save_path=save_path)

        if random.random() < epsilon:
            action = random.choice(possible_actions)
            logger.info(f"Exploring: Chose random action '{action}'")
        else:
            logger.debug("Sampling experiences from replay buffer...")
            experiences = replay_buffer.sample(batch_size=5)

            logger.debug("Sending screenshot and experiences to Fara...")
            response = send_to_fara(screenshot_data, experiences)

            logger.debug(f"Fara response: '{response.strip()}'")
            action = parse_action(response)
            logger.info(f"Exploiting: Fara suggests '{action}'")

        execute_action(action)
        
        logger.debug("Pausing for action to take effect...")
        time.sleep(0.5) 

        logger.debug("Getting current score...")
        current_score = get_game_score(monitor_score)
        reward = current_score - last_score

        if reward != 0:
            logger.success(f"Score changed! Current: {current_score}, Last: {last_score}, Reward: {reward}")
        else:
            logger.debug(f"No score change. Current: {current_score}")

        last_score = current_score

        # The hash is calculated from the *resized* screenshot data. This ensures the AI's
        # memory of a state is consistent with the visual data it uses for decisions.
        experience = (action, reward, hash(screenshot_data))
        logger.debug(f"Adding experience to replay buffer: ({action}, {reward})")
        replay_buffer.add(experience)

        logger.debug("Pausing before next turn...")
        time.sleep(1)

if __name__ == "__main__":
    main()
