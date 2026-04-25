import os
import time
import vizdoom as vzd
game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg")) # or any other scenario file

game.set_doom_skill(3)          # 0 (easiest) to 5 (nightmare)
game.add_game_args("+sv_cheats 1")
game.add_game_args("+skill 3")
game.set_mode(vzd.Mode.SPECTATOR)
game.init()
game.send_game_command("addbot") # add scripted opponent
game.send_game_command("addbot")
# Now run your neural agent policy against these bots

game.new_episode()
tic_duration = 1/35  # Doom runs at 35 fps; cap loop to avoid running too fast
while not game.is_episode_finished():
    t0 = time.time()
    game.advance_action(1)   # pumps SDL events; SPECTATOR ignores the action
    elapsed = time.time() - t0
    sleep_for = tic_duration - elapsed
    if sleep_for > 0:
        time.sleep(sleep_for)