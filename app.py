from flask import Flask, render_template, request, redirect, url_for
from model_utils import predict_text, process_file, AVAILABLE_MODELS

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_choice = request.form.get('model_choice', 'toxic_bert')
        model_info = AVAILABLE_MODELS.get(model_choice, {}).get('desc', 'Unknown Model')

        # 1. Cek Upload File
        if 'file_input' in request.files and request.files['file_input'].filename != '':
            file = request.files['file_input']
            results = process_file(file, model_choice)
            return render_template('result_file.html', results=results, model_name=model_info)

        # 2. Cek Input Manual
        raw_text = request.form.get('user_text')
        if raw_text:
            result = predict_text(raw_text, model_choice)
            # Debugging print
            # print("Hasil:", result)
            return render_template('result.html', data=result, model_name=model_info)

    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/ulang')
def ulang():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)