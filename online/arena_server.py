from online.lib import *
from flask import Flask, escape, request, jsonify

store = Store()
app = Flask(__name__)

# preventing race condition
create_room_lock = Lock()


@app.route('/')
def hello():
    return jsonify(list(store.rooms.keys()))


@app.route('/room/<int:room_id>/reset', methods=['POST'])
def reset(room_id):
    with create_room_lock:
        room = store.rooms[room_id]
        return jsonify(room.reset())


@app.route('/room/<int:room_id>/step/<int:action>', methods=['POST'])
def step(room_id, action):
    token = request.args.get('token')
    room = store.rooms[room_id]
    return jsonify(room.step(action, token))


@app.errorhandler(Exception)
def value_exception(e):
    # now you're handling non-HTTP exceptions only
    return jsonify({
        'type': 'exception',
        'msg': e.args[0],
    }), 400


if __name__ == "__main__":
    app.run('0.0.0.0', 5000, debug=True, threaded=True)