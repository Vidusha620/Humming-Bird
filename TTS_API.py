import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set the text to be converted to speech
text = "Hello, we are team HummingBird"

# Convert the text to speech
engine.say(text)

# Run the text-to-speech engine
engine.runAndWait()
