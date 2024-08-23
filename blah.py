import nltk
from nltk.tokenize import word_tokenize
import os

# Print the data path being used
print("NLTK Data Paths:", nltk.data.path)

# Check if the file exists in the expected directory
punkt_path = os.path.join('C:\\Users\\Eason\\AppData\\Roaming\\nltk_data', 'tokenizers', 'punkt', 'english.pickle')
print("Punkt File Exists:", os.path.exists(punkt_path))

nltk.data.path.append('C:\\Users\\Eason\\AppData\\Roaming\\nltk_data')
nltk.download('punkt')

text = "Hello! How are you?"
tokens = word_tokenize(text)
print(tokens)