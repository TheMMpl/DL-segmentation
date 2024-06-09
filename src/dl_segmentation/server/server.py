'''
Warning, you must run pip install -e . in the root directory for these imports to work

'''
import os
from pathlib import Path
import matplotlib.pyplot as plt
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from dl_segmentation.pipelines.reporting.nodes import load_model, prepare_data, run_inference
from consts import NUM_CLASSES, MODEL_CHECKPOINT, UPLOAD_FOLDER, RESULTS_FOLDER

# UPLOAD_FOLDER = '../../../data'
# RESULTS_FOLDER = '../../../results'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__,static_folder=os.path.abspath("results/"))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_results(filepath,filename):
    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    model=load_model(MODEL_CHECKPOINT)
    data=prepare_data([filepath])
    results=run_inference(model,data)
    for res in results:
        plt.imsave(os.path.join(app.config['RESULTS_FOLDER'], filename),res)
    #result.save(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("file saved!")
            save_results(filepath,filename)
            print("results ok!")
            return redirect(url_for('retrieve_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/results/<name>')
def retrieve_file(name):
    #root_dir = os.path.dirname(os.getcwd())
    #os.path.join(root_dir,app.config["RESULTS_FOLDER"])
    return send_from_directory(app.static_folder, name)


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug=True)