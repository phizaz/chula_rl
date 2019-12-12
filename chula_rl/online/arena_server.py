from flask import Flask, escape, jsonify, request

from chula_rl.online.lib import *

store = Store()
app = Flask(__name__)

# preventing race condition
create_room_lock = Lock()


@app.route('/')
def hello():
    return jsonify(sorted(list(store.rooms.keys())))


@app.route('/room/<int:room_id>/reset', methods=['POST'])
def reset(room_id):
    token = request.args.get('token', None)
    play_match = request.args.get('match', False)
    if play_match is not False: play_match = True
    superpower = request.args.get('superpower', False)
    if superpower is not False: superpower = True

    print('token:', token)
    # print('match:', play_match)
    # print('superpower:', superpower)

    with create_room_lock:
        if room_id not in store.rooms:
            # create the room, if the said room does not exist
            if play_match:
                # create a match of rooms
                create_match(2, 3, room_id, store.rooms)
            else:
                # create only a single room
                store.rooms[room_id] = Room(superpower=superpower)
        # only join the room
        room = store.rooms[room_id]

    # you might need a token to reset the room (in case of a match room)
    return jsonify(room.reset(token=token))


@app.route('/room/<int:room_id>/step/<int:action>', methods=['POST'])
def step(room_id, action):
    token = request.args.get('token')
    room = store.rooms[room_id]
    return jsonify(room.step(action, token=token))


@app.errorhandler(Exception)
def value_exception(e):
    # now you're handling non-HTTP exceptions only
    return jsonify({
        'type': 'exception',
        'msg': e.args[0],
    }), 400


if __name__ == "__main__":
    app.run('0.0.0.0', 5000, debug=True, threaded=True)
