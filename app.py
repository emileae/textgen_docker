# flask_web/app.py

from textgenrnn import textgenrnn
from flask_cors import CORS
from flask import Flask, render_template, request
app = Flask(__name__)
CORS(app)# currently this allows all origins access to the API

@app.route('/')
def hello_world():
    # return 'Hey, we have Flask in a Docker container!'
    seed = request.args.get("seed")

    if seed:
        prefix = seed
    else:
        prefix = "She walked down the street and suddenly "
    textgen = textgenrnn('litero_weights.hdf5', 'litero_vocab.json', 'litero_config.json', 'litero')
    # textgen.generate_to_file('./litero.txt', temperature=0.5, top_n=5, prefix="I mounted ", max_gen_length=2000)

    txt = textgen.generate(temperature=0.5, top_n=5, prefix=prefix, max_gen_length=2000, return_as_list=False)
    textgen.clear_session()
    # textgen.__init__()

    return txt


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')