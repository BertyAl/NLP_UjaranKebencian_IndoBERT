from flask import Flask, render_template, request, redirect, url_for
from model_utils import predict_toxicity

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        raw_text = request.form.get('user_text')

        if raw_text:
            result = predict_toxicity(raw_text)
            return render_template('result.html', original_text=raw_text, data=result)

    return render_template('index.html')


@app.route('/ulang')
def ulang():
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, port=5000)