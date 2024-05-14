from chain import chain
from flask import Flask, request, jsonify, abort
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if 'userMessage' not in data:
        abort(400, description="Missing 'userMessage' in request data.")
    
    user_message = data['userMessage']
    try:
        response = chain.invoke(user_message)
    except Exception as e:
        abort(500, description=str(e))

    return jsonify({"assistant": response}), 200


if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)