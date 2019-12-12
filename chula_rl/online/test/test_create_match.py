from chula_rl.online.lib import create_match

rooms = {}

first_room = create_match(2, 3, 1, rooms)

for k, v in rooms.items():
    print(f'{k}: super: {v.superpower}, auth: {v._need_authen}, first: {v.first_player} '
          f'next_room: {v.next_room}')
