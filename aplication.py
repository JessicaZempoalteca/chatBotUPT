from flask import Flask, render_template, jsonify, request
import procesadorModelo as processor


app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())



@app.route('/chatBotModel', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = processor.chatbot_response(the_question)

    return jsonify({"response": response })



if __name__ == '__main__':
    # se ejecuta el servidor de flask en el puerto 8888 y en modo debug para que se reinicie automaticamente cuando se haga un cambio
    app.run(host='0.0.0.0', port='8888', debug=True)