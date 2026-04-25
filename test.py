import os
import vizdoom as vzd
game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg")) # or any other scenario file

game.set_doom_skill(3)          # 0 (easiest) to 5 (nightmare)
game.add_game_args("+sv_cheats 1")
game.add_game_args("+skill 3")
game.init()
game.send_game_command("addbot") # add scripted opponent
game.send_game_command("addbot")
# Now run your neural agent policy against these bots

game.new_episode()
while not game.is_episode_finished():
    game.advance_action(1)