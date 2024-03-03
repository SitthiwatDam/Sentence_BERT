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

# set the source and target language
SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'th'
#set the parameters for the model
input_dim   = len(vocab_transform[SRC_LANGUAGE])
output_dim  = len(vocab_transform[TRG_LANGUAGE])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1
attention = "additive"

SRC_PAD_IDX = 1
TRG_PAD_IDX = 1
EOS_IDX = 3
# create the encoder and decoder
encoder = Encoder(attention,
              input_dim,
              hid_dim,
              enc_layers,
              enc_heads,
              enc_pf_dim,
              enc_dropout,
              device)

decoder = Decoder(attention,
              output_dim,
              hid_dim,
              dec_layers,
              dec_heads,
              dec_pf_dim,
              enc_dropout,
              device)
# create the model and load the weights
model = Seq2SeqTransformer(encoder,decoder,SRC_PAD_IDX,TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load('./models/Seq2SeqTransformer_additive.pt', map_location=device))

def generate(model,src_text):
    # Set model to evaluation mode
    model.eval()

    # Perform inference on new data
    with torch.no_grad():
        # create a mask
        src_mask = model.make_src_mask(src_text)
        # pass the src text and mask to the encoder
        enc_src = model.encoder(src_text,src_mask)
        # one matrix
        target = torch.ones((src_text.shape[0], 1), device=device, dtype=torch.long) * TRG_PAD_IDX

        for i in range(1, 100):
            target_mask = model.make_trg_mask(target)
            output, _ = model.decoder(target, enc_src, target_mask, src_mask)
            print('output: ',output, "_: ", _)
            prediction_token = output.argmax(2)[:, -1].unsqueeze(1)
            target = torch.cat((target, prediction_token), dim=1)

            if prediction_token.item() == EOS_IDX:
                break

        return target[:, 1:]
    

#
def get_generate(model,search_word):
    src_text = ["<sos>"] + tokenizer_en(search_word.lower()) + ["<eos>"]
    print(src_text)
    src_num = [vocab_transform[SRC_LANGUAGE][i] for i in src_text]
    print('src_num', src_num)
    src_input = torch.tensor(src_num, dtype=torch.int64).reshape(1, -1).to(device)
    print('src_input: ',src_input) 
    model_output = generate(model,src_input)[0]
    print('model_output: ',model_output)
    mapping = vocab_transform[TRG_LANGUAGE].get_itos()
    thai_output = [mapping[token.item()] for token in model_output]
    return thai_output
        

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