from flask import Flask, render_template, request
import torchtext, torch
from function import *
from torchtext.data.utils import get_tokenizer
app = Flask(__name__)
# set the device
device=  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the tokenizer
tokenizer_en = get_tokenizer("basic_english")
# load the token and vocab transform
with open('./models/token_transform.pkl', 'rb') as f:
    token_transform = pickle.load(f)
with open('./models/vocab_transform.pkl', 'rb') as f:
    vocab_transform = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def index():
    most_similar_words = None
    if request.method == 'POST':
        search_word = request.form.get('search')
        if search_word:
            most_similar_words = get_generate(model,search_word)
            # combine the words and cut the <eos> token
            most_similar_words = "".join(most_similar_words)[:-5]
    print(most_similar_words)
    return render_template('index.html', most_similar_words=most_similar_words)
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")