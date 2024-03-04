from flask import Flask, render_template, request
import torchtext, torch
from function import *
from torchtext.data.utils import get_tokenizer

app = Flask(__name__)
# set the device
device=  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the tokenizer and vocab transform
tokenizer= get_tokenizer("basic_english")
vocab = torch.load('./models/vocab.pt', map_location=torch.device('cpu'))

# Load the custom sentence transformer model
params, state = torch.load('./models/s_bert.pt', map_location=torch.device('cpu'))
loaded_model = BERT(**params, device=device).to(device)
loaded_model.load_state_dict(state)
loaded_model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate_similarity():
    if request.method == 'POST':
        sentence1 = request.form['sentence1']
        sentence2 = request.form['sentence2']

        # Calculate cosine similarity
        similarity =  calculate_similarity_bert(loaded_model, tokenizer, vocab, sentence1, sentence2, device)

        return render_template('index.html', similarity=similarity, sentence1=sentence1, sentence2=sentence2)

if __name__ == '__main__':
    app.run(debug=True)
