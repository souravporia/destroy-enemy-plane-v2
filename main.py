from src.environment import EnemyBase
from src.agent import MissAgent
from src.visualization import Heatmap
import json

if __name__ == '__main__' :
    with open("config/config.json", "r") as f :
        config = json.load(f)
       
    env = EnemyBase(config["environment"])

    state = env.reset()

    agent = MissAgent(env, config['agent'])

    heatmap = Heatmap(env, agent)

    
    if(config['agent']['algorithm'] == "pi") :
        heatmap.visualize_policy_iteration(config['agent']['policy_iteration']['theta'])
        heatmap.show()
    elif(config['agent']['algorithm'] == "mc") :
        heatmap.visualize_mc(config['agent']['mc']['number_of_episodes'], config['visualization']['mc_episodes_inverval'])
        heatmap.show()
    else :
        print("! Supported algorithm : \'mc\', \'pi\' !")
        exit(1)