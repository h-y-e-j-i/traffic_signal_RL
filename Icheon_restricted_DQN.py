import ray, numpy as np
from gym import spaces
from IcheonTrainMultiEnv_restricted import SumoIcheonTrainMultiEnvironment as SumoIcheonTrainMultiEnvironment_restricted
from ray.rllib.agents import dqn
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

def policy_mapping(id):
    return id


if __name__ == "__main__":
    ray.init()
    config = dqn.DEFAULT_CONFIG
    config["num_workers"] = 1
    config["num_gpus"] = 0

    register_env("sumoIcheonTrainMultiEnv_restricted", lambda _:SumoIcheonTrainMultiEnvironment_restricted(
                                        net_file='/home/sonic/Desktop/nets/case06/intersection.net.xml',
                                        route_file='/home/sonic/Desktop/nets/case06/intersection_random.rou.xml',
                                        use_gui=False,
                                        algorithm="DQN",
                                        sim_max_time=18000,
                                        actions_are_logits=False))


    # register_env("sumoMultiEnv", lambda _:SumoIcheonMultiEnvironment(
    #                                     config_file='/home/sonic/Desktop/nets/case06/intersection.sumocfg',
    #                                     use_gui=False,
    #                                     algorithm="DQN",
    #                                     sim_max_time=18000,
    #                                     actions_are_logits=False))

    # register_env("sumoTestMultiEnv", lambda _:SumoTestMultiEnvironment(
    #                                     config_file='/home/sonic/Desktop/hyeji/sumo-rl-foundation/multi-agent/sumo/case03.sumocfg',
    #                                     use_gui=True,
    #                                     algorithm="DQN",                                        
    #                                     sim_max_time=3600,
    #                                     actions_are_logits=False))
    
    stop_timesteps = 500000

    # trainer = dqn.DQNTrainer(env="sumoIcheonTrainMultiEnv_restricted", config={
    #     "multiagent": {
    #         "policies": {  #"policy_graphs"
    #             'J3': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(19*2+1), high=np.array(['inf']*(19*2+1))), spaces.Discrete(5), {}),
    #             'J5': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(14*2+1), high=np.array(['inf']*(14*2+1))), spaces.Discrete(3), {}),
    #             'J9': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(28*2+1), high=np.array(['inf']*(28*2+1))), spaces.Discrete(4), {}),
    #             'J14': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(13*2+1), high=np.array(['inf']*(13*2+1))), spaces.Discrete(3), {}),
    #             'J18': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(32*2+1), high=np.array(['inf']*(32*2+1))), spaces.Discrete(4), {}),
    #             'J22': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(18*2+1), high=np.array(['inf']*(18*2+1))), spaces.Discrete(3), {}),               
    #         },
    #         "policy_mapping_fn": policy_mapping  # Traffic lights are always controlled by this policy
    #     },
    #     #"timesteps_per_iteration" : stop_timesteps,
    #     "lr": 0.0001,
    # })

    trainer = dqn.DQNTrainer(env="sumoIcheonTrainMultiEnv_restricted", config={
        "multiagent": {
            "policies": {  #"policy_graphs"
                'J3': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(19*2+1), high=np.array(['inf']*(19*2+1))), spaces.Discrete(2), {}),
                'J5': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(14*2+1), high=np.array(['inf']*(14*2+1))), spaces.Discrete(2), {}),
                'J9': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(28*2+1), high=np.array(['inf']*(28*2+1))), spaces.Discrete(2), {}),
                'J14': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(13*2+1), high=np.array(['inf']*(13*2+1))), spaces.Discrete(2), {}),
                'J18': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(32*2+1), high=np.array(['inf']*(32*2+1))), spaces.Discrete(2), {}),
                'J22': (dqn.DQNTFPolicy, spaces.Box(low=np.zeros(18*2+1), high=np.array(['inf']*(18*2+1))), spaces.Discrete(2), {}),               
            },
            "policy_mapping_fn": policy_mapping  # Traffic lights are always controlled by this policy
        },
        #"timesteps_per_iteration" : stop_timesteps,
        "lr": 0.0001,
    })
    
    iter = 100000000000000
    for i in range(iter):
        result = trainer.train()        
        if i%100 == 0:
            print(pretty_print(result))
            check_point_path = trainer.save()            
            print("checkpoint saved at", check_point_path)

# local observation
# home/sonic/ray_results/DQN_sumoMultiEnv_2021-08-31_17-36-11g58ktd2c/checkpoint_000628/checkpoint-628


# random action
# /home/sonic/ray_results/DQN_sumoIcheonTrainMultiEnv_2021-09-29_19-49-360bim66xk/checkpoint_000701

# restricted
# /home/sonic/ray_results/DQN_sumoIcheonTrainMultiEnv_restricted_2021-10-07_18-09-44lo48ljt8/checkpoint_000001/checkpoint-1
# #home/sonic/ray_results/DQN_sumoIcheonTrainMultiEnv_restricted_2021-10-08_17-09-11y3v51lrp/checkpoint_000001/checkpoint-1
# /home/sonic/ray_results/DQN_sumoIcheonTrainMultiEnv_restricted_2021-10-09_14-38-06m63ldfj9/checkpoint_000001/checkpoint-1