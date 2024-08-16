import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the chatbot model
class ChatbotModel(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_length):
        super(ChatbotModel, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=max_length)
        self.encoder = LSTM(hidden_dim, return_state=True)
        self.repeat_vector = RepeatVector(max_length)
        self.decoder = LSTM(hidden_dim, return_sequences=True)
        self.output_layer = Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        embedded_inputs = self.embedding(inputs)
        encoder_outputs, state_h, state_c = self.encoder(embedded_inputs)
        repeated_context = self.repeat_vector(state_h)  # Repeat the context vector
        decoder_outputs = self.decoder(repeated_context, initial_state=[state_h, state_c])
        outputs = self.output_layer(decoder_outputs)
        return outputs

# Define the model parameters
vocab_size = 10000
embedding_dim = 128
hidden_dim = 256
max_length = 50

# Create the chatbot model
model = ChatbotModel(vocab_size, embedding_dim, hidden_dim, max_length)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Define the training dataset
conversations = [
     ("Hello", 
     "Hello there! How can I assist you with your courses today?"),
     ("Hi, I'm looking for a course on Python programming.", "Great choice! We have several Python courses available. What's your current level of experience with Python? Beginner, Intermediate, or Advanced?"),
    ("I'm a beginner.", "Excellent! Our 'Python for Beginners' course is a popular choice. It covers the basics of Python programming and includes interactive exercises and quizzes. Would you like to enroll or learn more about the course?"),
    ("Tell me more.", "Sure! The course includes 10 modules, each covering a specific topic like data types, functions, and file handling. You'll also get access to our online community to ask questions and get help from instructors and peers."),
    ("That sounds good. How long does the course take to complete?", "The course is self-paced, so you can complete it at your own speed. However, most students finish it within 4-6 weeks."),
    ("Okay, I'll enroll.", "Great choice! You'll receive an email with course details and login information. If you have any questions or need help, feel free to ask!"),
    ("Thanks for your help!", "You're welcome! Good luck with your course, and have fun learning Python!"),
    ("How do I access my course materials?", 
     "You can access your course materials by logging into your student portal and clicking on the course link."),
    ("What is the deadline for submitting my assignment?", 
     "The deadline for submitting your assignment is 25th August. Please check your course syllabus for more details."),
    ("How do I improve my programming skills?", 
     "You can improve your programming skills by practicing regularly, working on projects, and learning from online tutorials."),
    ("How do I submit my assignments?", 
     "You can submit your assignments through the 'Assignments' section of your course dashboard."),
    ("What should I do if I miss a deadline?", 
     "If you miss a deadline, contact your instructor immediately to discuss possible extensions or penalties."),
    ("How can I track my course progress?", 
     "You can track your course progress in the 'Progress' tab of your student dashboard."),
    ("Where can I find my grades?", 
     "Your grades can be found in the 'Grades' section of your course dashboard."),
    ("How do I contact my instructor?", 
     "You can contact your instructor via the messaging system in your student portal or by email if provided."),
    ("Can I retake a quiz if I fail?", 
     "Yes, most courses allow you to retake quizzes. Check the quiz instructions for specific rules."),
    ("What is a passing grade for this course?", 
     "A passing grade usually requires at least 60% or higher, but it may vary by course. Check your syllabus for exact requirements."),
    ("Are there any prerequisites for this course?", 
     "Prerequisites vary by course. Please check the course description for any required prior knowledge or skills."),
    ("How do I join a live session?", 
     "You can join a live session by clicking on the session link in your course dashboard at the scheduled time."),
]
# Preprocess the data
input_texts = [conversation[0] for conversation in conversations]
output_texts = [conversation[1] for conversation in conversations]

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(input_texts + output_texts)

input_sequences = tokenizer.texts_to_sequences(input_texts)
output_sequences = tokenizer.texts_to_sequences(output_texts)

input_sequences_padded = pad_sequences(input_sequences, maxlen=max_length, padding='post')
output_sequences_padded = pad_sequences(output_sequences, maxlen=max_length, padding='post')

# Train the model
output_labels = np.expand_dims(output_sequences_padded, axis=-1)
model.fit(input_sequences_padded, output_labels, epochs=1300, verbose=1)

# Define a function to generate a response
def generate_response(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')
    
    predictions = model.predict(input_sequence_padded)
    predicted_indices = np.argmax(predictions[0], axis=-1)
    
    predicted_words = [tokenizer.index_word.get(index, '') for index in predicted_indices if index != 0]
    response = ' '.join(predicted_words)
    return response

# Chat with the chatbot
print("Chatbot is ready to chat. Type 'exit' or 'quit' to end the session.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Thankyou!")
        break
    response = generate_response(user_input)
    print("Chatbot:", response)
