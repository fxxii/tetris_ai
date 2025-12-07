import ollama
import mss
import pyautogui
import time
import base64
import random
from collections import deque

# --- Constants ---
#MODEL_NAME = "maternion/fara:7b"
MODEL_NAME = "llama3.2-vision"

# --- Reinforcement Learning Components ---

class ReplayBuffer:
    """A simple replay buffer to store experiences."""
    def __init__(self, max_size=1000):
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

def capture_screenshot(monitor):
    """Captures a screenshot of the specified monitor region and returns raw bytes."""
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        return mss.tools.to_png(sct_img.rgb, sct_img.size)

def get_game_score(monitor_score):
    """
    Captures a screenshot of the score region and uses the Fara model to extract the score.
    """
    try:
        score_image_data = capture_screenshot(monitor_score)
        encoded_image = base64.b64encode(score_image_data).decode('utf-8')

        prompt = "Analyze the attached image and return ONLY the numerical score shown. For example, if the score is '12345', your response should be just '12345'."

        response = ollama.chat(
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
            print(f"Score text: '{score_text}'")
            return int(score_text)
        else:
            print(f"Warning: Fara could not read the score. Response: '{response['message']['content']}'")
            return 0
    except Exception as e:
        print(f"Error reading score with Fara: {e}")
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
        "You are a helpful assistant. Describe the attached image in detail. "
        "What game is being played? What color are the blocks?"
    )

    response = ollama.chat(
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
        print(f"Executing action: {action}")
        pyautogui.press(action)
    else:
        print("No action determined.")

def main():
    """The main loop for the Tetris AI with reinforcement learning."""
    
    # --- Configuration ---
    monitor_game = {"top": 456, "left": 158, "width": 804, "height": 856}
    monitor_score = {"top": 415, "left": 264, "width": 450, "height": 364}
    #monitor_score = {"top": 0, "left": 0, "width": 100, "height": 100}
    # RL parameters
    epsilon = 0.1
    replay_buffer = ReplayBuffer(max_size=1000)
    possible_actions = ["left", "right", "up", "down"]

    print("Starting Tetris AI in 5 seconds...")
    time.sleep(5)
    print("AI active.")

    last_score = get_game_score(monitor_score)

    while True:
        screenshot_data = capture_screenshot(monitor_game)

        if random.random() < epsilon:
            action = random.choice(possible_actions)
            print(f"Exploring: chose random action '{action}'")
        else:
            experiences = replay_buffer.sample(batch_size=5)
            response = send_to_fara(screenshot_data, experiences)
            action = parse_action(response)
            print(f"Exploiting: Fara suggests '{action}' based on response: '{response[:50]}...'")

        execute_action(action)
        
        time.sleep(0.5) 

        current_score = get_game_score(monitor_score)
        reward = current_score - last_score
        last_score = current_score
        print(f"Reward for action '{action}': {reward}")

        experience = (action, reward, hash(screenshot_data))
        replay_buffer.add(experience)

        time.sleep(1)

if __name__ == "__main__":
    main()
