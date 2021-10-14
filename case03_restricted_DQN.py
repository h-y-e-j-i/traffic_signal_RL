import ray, numpy as np
import tensorflow
from gym import spaces
from case03TrainMultiEnv_restricted import SumoCase03TrainMultiEnvironment as SumoCase03TrainMultiEnvironment_restricted
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env


def policy_mapping(id):
    return id

OBS_LEN = 25
ACT_LEN = 4

if __name__ == "__main__":
    ray.init()
    config = dqn.DEFAULT_CONFIG
    config["num_workers"] = 1
    config["num_gpus"] = 0

    register_env("sumoCase03TrainMultiEnv_random", lambda _:SumoCase03TrainMultiEnvironment_restricted(
                                        net_file='/home/sonic/Desktop/nets/case03/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case03/intersection_random.rou.xml',
                                        use_gui=False,
                                        algorithm="DQN",
                                        sim_max_time=3600,
                                        actions_are_logits=False))
    
    stop_timesteps = 500000

    trainer = dqn.DQNTrainer(env="sumoCase03TrainMultiEnv_random", config={
        "multiagent": {
            "policies": {  #"policy_graphs"
                'gneJ00': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(OBS_LEN), high=np.array(['inf']*OBS_LEN)), spaces.Discrete(2), {}),
                'gneJ9': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(OBS_LEN), high=np.array(['inf']*OBS_LEN)), spaces.Discrete(2), {}),
                # 'gneJ26': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(OBS_LEN), high=np.array(['inf']*OBS_LEN)), spaces.Discrete(ACT_LEN), {}),
                
            },
            "policy_mapping_fn": policy_mapping  # Traffic lights are always controlled by this policy
        },
        #"timesteps_per_iteration" : stop_timesteps,
        "lr": 0.0001,
    })
    
    iter = 100000000000000
    for i in range(iter):        
        result = trainer.train()   
        if i%100==0 : 
            check_point_path = trainer.save() 
            print(pretty_print(result))
            print("checkpoint saved at", check_point_path)

        # if i%10 == 0:           
        #     policy_grpah = trainer.get_policy('gneJ00').get_session().graph
        #     with tensorflow.compat.v1.Graph().as_default():
        #         writer = tensorflow.compat.v1.summary.FileWriter("/home/sonic/Desktop/hyeji/graph/DQN/gneJ00", policy_grpah)

        #     policy_grpah = trainer.get_policy('gneJ9').get_session().graph
        #     with tensorflow.compat.v1.Graph().as_default():
        #         writer = tensorflow.compat.v1.summary.FileWriter("/home/sonic/Desktop/hyeji/graph/DQN/gneJ9", policy_grpah)

        #check_point_path = trainer.save()
        # print("checkpoint saved at", check_point_path)
    # /home/sonic/ray_results/DQN_sumoTestMultiEnv_2021-08-26_15-23-2848927j0a/checkpoint_000001/checkpoint-1
    


###### case03 ######
# local observation
# /home/sonic/ray_results/DQN_sumoTestMultiEnv_2021-08-30_18-29-42k9244t4v/checkpoint_001401/checkpoint-1401
# /home/sonic/ray_results/DQN_sumoMultiEnv_2021-08-31_13-34-541lkov5la/checkpoint_002604/checkpoint-2604
# /home/sonic/ray_results/DQN_sumoMultiEnv_2021-09-07_15-54-03gx2vbiik/checkpoint_002514/checkpoint-2514
#
# restricted
# /home/sonic/ray_results/DQN_sumoCase03TrainMultiEnv_random_2021-10-07_18-05-06j6czlmi9/checkpoint_000001/checkpoint-1
# /home/sonic/ray_results/DQN_sumoCase03TrainMultiEnv_random_2021-10-09_13-51-07cvm3o52u/checkpoint_000001/checkpoint-1

###### case04 #####
# period 0.1 
# 훈련 안됨ㅠㅠ
# /home/sonic/ray_results/DQN_sumoTrainMultiEnv_2021-09-23_16-18-55n85mfhgk

# period 2 + 1.5
# /home/sonic/ray_results/DQN_sumoTrainMultiEnv_2021-09-28_16-15-12wh45jno9/checkpoint_001201


###### case06 #####