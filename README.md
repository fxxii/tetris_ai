# Fara Tetris AI

This project uses the Fara-7B multimodal model to play a simple, browser-based game of Tetris. It learns and improves its gameplay over time using a reinforcement learning loop based on in-context learning.

## How It Works

1.  **Perception:** The AI captures a screenshot of the Tetris game board.
2.  **Action:** It sends this screenshot, along with a memory of recent "good" and "bad" moves, to the Fara model via a local Ollama instance.
3.  **Execution:** It parses Fara's suggested move and sends the corresponding keypress (`left`, `right`, `up`, `down`) to the game window.
4.  **Reward:** After each move, it captures an image of the score, uses Fara to read the number, and calculates a reward based on the change in score.
5.  **Learning:** This `(action, reward)` pair is stored in a replay buffer. The most influential experiences are then used to guide the model in future decisions.

## 1. Setup Instructions

### a. Install Ollama
Follow the official instructions for your operating system from [https://ollama.com/download](https://ollama.com/download).

### b. Pull the Fara Model
Once Ollama is installed, open your terminal and run the following command to download the model. This is a large file and may take several minutes.

```bash
ollama pull maternion/fara:7b
```

### c. Set Up the Tetris Game
This project includes a simple, browser-based Tetris game as a Git submodule.

1.  **Initialize the Submodule:**
    In your terminal, run the following command from the root of this project:
    ```bash
    git submodule update --init
    ```
2.  **Start the Game Server:**
    Navigate into the `tetris` directory and start a simple web server. If you have Python 3, you can use its built-in server:
    ```bash
    cd tetris
    python3 -m http.server 8000
    ```
3.  **Open the Game:**
    Open your web browser and navigate to `http://localhost:8000`. You should now see the Tetris game.

### d. Install Python Dependencies
This project requires a few Python libraries to function.

1.  **Navigate to the project root.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Configure and Run the AI

### a. Adjust Screen Coordinates (CRITICAL STEP)
This is the most important step. The AI needs to know exactly where the game board and the score are on your screen.

1.  Open the `tetris_ai.py` script in a text editor.
2.  Find the `main()` function at the bottom of the script.
3.  You will see two dictionaries that you need to edit: `monitor_game` and `monitor_score`.

    ```python
    # --- CRITICAL CONFIGURATION ---
    # You MUST update these pixel values to match the location of the
    # game and score on YOUR screen. Use a screenshot tool to find the
    # correct coordinates.
    monitor_game = {"top": 100, "left": 50, "width": 400, "height": 800}
    monitor_score = {"top": 50, "left": 500, "width": 200, "height": 50}
    ```
4.  **How to find these values:**
    *   Arrange your screen so the Tetris game is visible in your browser.
    *   Use a screenshot tool that allows you to select a region and see its pixel coordinates (e.g., the built-in screenshot tool on macOS, or Greenshot on Windows).
    *   For `monitor_game`, select a box that tightly encloses the main Tetris playing area.
    *   For `monitor_score`, select a box that tightly encloses the score number.
    *   Update the `top`, `left`, `width`, and `height` values in the script for both dictionaries.

### b. Run the AI
Once you have configured the screen coordinates, you are ready to run the AI.

1.  Make sure the Tetris game is running in your browser and is visible on your screen.
2.  Open a new terminal window and navigate to the root of this project.
3.  Run the following command:
    ```bash
    python tetris_ai.py
    ```
4.  The script will pause for 5 seconds. **Click on the Tetris game window to make it active.**
5.  The AI will then start playing the game. You will see its decisions and the calculated rewards printed in your terminal.
