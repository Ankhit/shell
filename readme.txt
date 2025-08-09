# Chatbot Project

## Overview
This project implements a chatbot that utilizes machine learning algorithms to classify user inputs based on predefined intents. The chatbot is designed to respond to user queries by matching them against patterns defined in an intents JSON file. The project includes implementations of Logistic Regression, Naive Bayes, and Random Forest classifiers.

## Features
- **Intent Recognition**: The chatbot can recognize user intents based on predefined patterns.
- **Multiple Classifiers**: The project includes Logistic Regression, Naive Bayes, and Random Forest algorithms for intent classification.
- **Hyperparameter Optimization**: Logistic Regression is fine-tuned using Grid Search for optimal performance.
- **Performance Evaluation**: The models are evaluated using accuracy, precision, and recall metrics.

## Requirements
To run this project, you will need the following Python packages:
- nltk
- scikit-learn
- joblib
- matplotlib (optional for visualization)
- seaborn (optional for visualization)
- streamlit (for the web interface)

You can install the required packages using pip:


## Setup Instructions
1. **Download the Project Files**: Clone or download this repository to your local machine.

2. **Prepare the Intents JSON File**: Ensure you have an `intents.json` file in the same directory as the scripts. This file should contain the intents and their corresponding patterns and responses.

3. **Run the Training Script**: Open a terminal and navigate to the project directory. Run the following command to train the models:


4. **Run the Streamlit Application**: After training is complete, you can start the chatbot interface using Streamlit:


5. **Interact with the Chatbot**: Open your web browser and go to `http://localhost:8501` to interact with your chatbot.

## Usage Instructions
- Type your message in the input box and click "Send" to receive a response from the chatbot.
- You can clear the conversation history by clicking "Clear Conversation".

## Troubleshooting
- If you encounter any issues with model loading, ensure that you have run `chatbot.py` successfully before starting `app.py`.
- Make sure all required packages are installed.

## Future Improvements
- Implement additional classifiers such as Support Vector Machines (SVM) or Decision Trees.
- Add data visualization features to analyze model performance metrics.
- Enhance error handling and logging for better debugging.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
