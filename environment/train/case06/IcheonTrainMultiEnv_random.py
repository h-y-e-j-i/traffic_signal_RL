from argparse import Action
from math import inf
from os import stat
from ray.rllib.env import MultiAgentEnv
import gym, gym.spaces, traci, sys, numpy as np

# train할 sumo 환경

MIN_TIME = 3
NUM_ROUTE_FILE = 50

class traffic_signal:
    def __init__(self):
        self.current_phase_id = None
        self.next_phase_id = None
        self.yellow_run_time = 0 # 황색불 점등 시간


class SumoIcheonTrainMultiEnvironment(MultiAgentEnv):      
    def __init__(self, actions_are_logits, use_gui, sim_max_time, net_file, route_file, algorithm):
        self._algorithm = algorithm # 사용한 알고리즘 종류
        self._net_file = net_file # sumo net file path
        self._route_file = route_file # sumo route file path
        #self._config_file= config_file # sumo config file 위치      
        self._sim_max_time=sim_max_time # 한 에피소드당 시뮬레이션 진행 최대 시간
        self._sim_episode = 0 # 에피소드 횟수
        # MADDPG emits action logits instead of actual discrete actions
        self._actions_are_logits = actions_are_logits # DDPG를 사용할 때만 True
        # self.sim_max_episoide = sim_max_episoide        
        self._use_gui=use_gui # gui 사용 여부
        if self._use_gui:self._sumoBinary = "sumo-gui"
        else : self._sumoBinary = "sumo"    

        self._program_ID = 0

        self._traffic_signal = dict() # 신호등

        # self.config_file = config_file
        # self.use_gui = use_gui
        # self.sim_time = sim_time

        # route file 이름 속 번호(1~50)
        # episode가 진행될때마다 1씩 늘어난다.
        self._route_file_num = 1

        # 초기에 sumo를 실행하여 lane의 아이디를 가져온다
        #_sumoInitCmd = ['sumo', '-c', self._config_file]
        _sumoInitCmd = ['sumo', '-n', self._net_file]
        traci.start(_sumoInitCmd)

        self._ts_IDs = traci.trafficlight.getIDList() # 모든 신호등 ID
        self._lane_IDs = traci.lane.getIDList() # 모든 lane ID
        self._lane_IDs_by_ts = dict() # 신호등별 lane
        self._inlane_IDs_by_ts = dict() # 신호등별 진입 lane
        self._outlane_IDs_by_ts = dict() # 신호들별 진출 lane

        self._phases_info = dict() # 신호등별 phase info

        for ts_ID in self._ts_IDs:
            self._phases_info[ts_ID] = dict()
            self._inlane_IDs_by_ts[ts_ID]=list(traci.trafficlight.getControlledLanes(ts_ID)) # tuple => list 
            self._outlane_IDs_by_ts[ts_ID] = list()

            # 신호등별 logic을 가져온다
            Logic = traci.trafficlight.getAllProgramLogics(ts_ID)
            pid = 0
            # 신호등별 programID에 따른 phase를 가져온다                 
            for program in Logic:                
                if pid not in self._phases_info[ts_ID].keys() :
                    self._phases_info[ts_ID][pid] = list()                   
                for phase in program.getPhases():
                    self._phases_info[ts_ID][pid].append(phase.duration)
                pid += 1   
                

            for lanes in traci.trafficlight.getControlledLinks(ts_ID): # 차선1, 차선2, 차선1과 차선2를 잇는 링크
                # 차선1와 차선2가 진입 lane에 없다면 진출 lane임
                if lanes[0][0] not in self._inlane_IDs_by_ts[ts_ID] and lanes[0][0] not in self._outlane_IDs_by_ts[ts_ID]:
                    self._outlane_IDs_by_ts[ts_ID].append(lanes[0][0])
                if lanes[0][1] not in self._inlane_IDs_by_ts[ts_ID] and lanes[0][1] not in self._outlane_IDs_by_ts[ts_ID]:
                    self._outlane_IDs_by_ts[ts_ID].append(lanes[0][1])
                    
            # 신호등별 lane = 신호등별 진입 lane + 진출 lane                       
            self._lane_IDs_by_ts[ts_ID] = self._inlane_IDs_by_ts[ts_ID]+self._outlane_IDs_by_ts[ts_ID]
        traci.close()

        
       
        # observation과 action의 크기를 지정

        # 1. aciton + 각 lane의 차량 대수 + 각 lane의 차량들의 총 waiting time
        # self._observation_space_len = 1 + len(self._lane_IDs) + len(self._lane_IDs) 

        # 2. action + agent별 각 lane의 차량 대수 + agent별 각 lane의 차량들의 총 waiting time
        self._observation_space_len = 1 + len(self._lane_IDs_by_ts['J9']) + len(self._lane_IDs_by_ts['J9']) 
        self._observation_space = gym.spaces.Box(low=np.zeros(self._observation_space_len), high=np.array(['inf']*self._observation_space_len))
        self._action_space = gym.spaces.Discrete(6) 
        self._actions_len = {'J3':5, 'J5':3, 'J9':4, 'J14':3, 'J18':4, 'J22':3}
       
    def reset(self):
        # 에피소드가 시작하기 직전 초기화
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        
        """
        # 초기 observation은 0행렬
        state = dict()
        self._reward_sum = dict()

        for ts_ID in self._ts_IDs:
            state[ts_ID] = np.zeros(len(self._lane_IDs_by_ts[ts_ID])*2+1)      
            self._reward_sum[ts_ID] = 0
            self._traffic_signal[ts_ID] = traffic_signal()
        self._reward_sum['sum'] = 0

        # route file의 확장자 8글자 (.rou.xml)를 제거하고 숫자를 추가한 뒤 확장자를 다시 붙인다.
        run_route_file = self._route_file[:-8] + str(self._route_file_num) + '.rou.xml'

        # route file을 50번까지 사용했다면 다시 1부터 다시 시작
        self._route_file_num = (self._route_file_num)%NUM_ROUTE_FILE+1


       # sumo (재)실행
        #sumoCmd = [self._sumoBinary, "-c",  self._config_file,  '--quit-on-end', '--random', "--start"]
        sumoCmd = [self._sumoBinary, "-n",  self._net_file, '-r', run_route_file, '--quit-on-end', '--random', "--start"]
        traci.start(sumoCmd)     

        self._done = False

        # 에피소드 1회씩 증가
        self._sim_episode += 1
            
        return state

    def step(self, actions):

        # DDPG 종류의 알고리즘인 경우에만 적용        
        # DDPG 종류의 알고리즘의 action은 어떤 state에서 각각  action을 선택할 확률들로 이루어진 tuple이다.
        # 예를들어 action이 3개라면 DDPG에서는 [action 0을 선택할 확률, action 1를 선택할 확률, action 2를 선택할 확률]
        # https://ropiens.tistory.com/96
        # 가장 큰 확률의 action을 선택
        if self._actions_are_logits:
            for ts_ID in self._ts_IDs:
                actions[ts_ID]  = np.random.choice(self._actions_len[ts_ID], p=actions[ts_ID])
            # actions = {
            #     k: np.random.choice(3, p=v)                
            #     for k, v in actions.items()
            # }

        # action : 현재는 3번

        # 1. action은 신호의 phase 번호와 같다
        # for ts_ID in self._ts_IDs:       
        #     traci.trafficlight.setPhase(ts_ID, actions[ts_ID])

        # 2. 청색불만 점등한다. 청색불은 phase 번호가 짝수이므로 action*2 번째 신호를 점등.
        # 신호등 점등 시간이 최소 점등 시간을 넘지 못 할 때
        # for ts_ID in self._ts_IDs:       
        #     traci.trafficlight.setPhase(ts_ID, actions[ts_ID]*2)

        

        if traci.simulation.getTime()==0 or traci.simulation.getTime()==10800 : self._program_ID = 0
        elif traci.simulation.getTime()==7200 : self._program_ID = 1
        
        # 3. 청색불일 때만 action O, 황색불일 때는 action X
        for ts_ID in self._ts_IDs:                      
            if ts_ID in actions.keys():    
                # 시뮬레이션이 처음 돌 때                
                if self._traffic_signal[ts_ID].current_phase_id is None:                                
                    self._traffic_signal[ts_ID].current_phase_id = actions[ts_ID]*2 # aciton*2번 청색불이 켜짐
                # 청색불 => 동일한 청색불 : 유지
                elif self._traffic_signal[ts_ID].current_phase_id == actions[ts_ID]*2:
                    pass
                # 청색 불 => 다른 청색불
                elif self._traffic_signal[ts_ID].current_phase_id%2 == 0:                
                    self._traffic_signal[ts_ID].current_phase_id = (self._traffic_signal[ts_ID].current_phase_id+1)%(self._actions_len[ts_ID]*2)  # 다음 황색불로 변경             
                    self._traffic_signal[ts_ID].yellow_run_time = 1 # 황색불 진행 시간 증가
                    self._traffic_signal[ts_ID].next_phase_id = actions[ts_ID]*2 # 다음 phase id = action*2(청색불)
            # #  황색불인경우
            # elif self._traffic_signal[ts_ID].current_phase_id%2 == 1 :                
            else:                
                # 최소시간이 지나지 않았다면 
                if self._traffic_signal[ts_ID].yellow_run_time < self._phases_info[ts_ID][self._program_ID][self._traffic_signal[ts_ID].current_phase_id]:
                    self._traffic_signal[ts_ID].yellow_run_time += 1 # 황색불 진행시간만 증가 
                # 최소시간이 지났다면                 
                else:
                    self._traffic_signal[ts_ID].current_phase_id = self._traffic_signal[ts_ID].next_phase_id # 다음 청색불로 변경
                    self._traffic_signal[ts_ID].yellow_run_time = 0# 황색불 진행 시간 초기화
                    self._traffic_signal[ts_ID].next_phase_id = None # 다음 phase id None           

            traci.trafficlight.setPhase(ts_ID, self._traffic_signal[ts_ID].current_phase_id)

        traci.simulationStep()
            
        states = self._compute_local_observation(actions) # observation 계산        
        rewards = self._compute_reward() # reward 계산

        dones = dict()
        infos = dict()
        for ts_ID in self._ts_IDs:     
            if self._traffic_signal[ts_ID].current_phase_id%2 == 0:       
                dones[ts_ID] = False
                infos[ts_ID] = {}            
                self._reward_sum[ts_ID] += rewards[ts_ID]
                self._reward_sum['sum'] += rewards[ts_ID]

        # if traci.simulation.getTime()%100 == 0:
        #     print(states)
        #     print(rewards)

        if traci.simulation.getTime() <=self._sim_max_time: # 현재 시뮬레이션 시간이 시뮬레이션 최대 시간보다 작다면 = 아직 에피소드가 끝나지 않았다면 
           # 계속 진행           
            dones['__all__'] = False
        else: # 그렇지 않다면
            dones['__all__'] = True  # 에피소드 종료
            print(self._reward_sum)
            traci.close() # sumo 종료          

        return states, rewards, dones, infos
        
    
    # reward 계산 메소드

    # 1. -(차량들의 총 대기시간)
    # def _compute_reward(self):
    #     sum_waiting_time = 0 
    #     reward = dict()

    #     for veh_ID in traci.vehicle.getIDList():
    #         # 차량들의 총 대기시간
    #         sum_waiting_time += traci.vehicle.getWaitingTime(veh_ID)

    #     for ts_ID in self._ts_IDs:
    #         reward[ts_ID] = -sum_waiting_time
    #     return reward

    # 2. 지나간 차량대수 - 멈춘 차량대수
    def _compute_reward(self):        
        rewards = dict()        

        for ts_ID in self._ts_IDs:
            halting_vehicle = 0 # 진입 lane에서 멈춘 차량 
            passed_vehicle = 0 # 진출 lane으로 지나간 차량
            # 청색불인 경우만 reward을 구한다
            if self._traffic_signal[ts_ID].current_phase_id%2 == 0 :
                for outlane_ID in self._outlane_IDs_by_ts[ts_ID]:
                    # 진출 lane으로 지나간 차량 = 진출 lane에서 마지막 차량 대수 - 진출 lane에서 정지 차량 대수
                    passed_vehicle += traci.lane.getLastStepVehicleNumber(outlane_ID)
                    passed_vehicle -= traci.lane.getLastStepHaltingNumber(outlane_ID)
                
                for inlane_ID in self._inlane_IDs_by_ts[ts_ID]:
                    halting_vehicle += traci.lane.getLastStepHaltingNumber(inlane_ID)

                rewards[ts_ID] = passed_vehicle-halting_vehicle
        
        return rewards
       
        
    # observation 계산 메소드
    # 1. 현재 action + 전체 lane별 차량 대기 시간의 합 + 전체 lane별 정치 차량 대수
    def _compute_global_observation(self, action):        
        state = dict()        
        waiting_vehicle_time_state = []
        halting_vehicle_number_state = []

        for lane_ID in self._lane_IDs:           
            waiting_vehicle_time = 0
            for veh_ID in traci.lane.getLastStepVehicleIDs(lane_ID):
                waiting_vehicle_time += traci.vehicle.getWaitingTime(veh_ID)
            waiting_vehicle_time_state.append(waiting_vehicle_time) # c
            halting_vehicle_number_state.append(traci.lane.getLastStepVehicleNumber(lane_ID)) # 각 lane별 정지 차량 대수

        for ts_ID in self._ts_IDs:
            state[ts_ID] = list()
            state[ts_ID].append(action[ts_ID])
            state[ts_ID] = state[ts_ID]+waiting_vehicle_time_state+halting_vehicle_number_state
        # observation = action + 각 lane별 차량 대기 시간의 합 +  각 lane별 정지 차량 대수
        
        #state = state+waiting_vehicle_time_state+halting_vehicle_number_state 
        return state

    # 2. 현재 phase id + 각 agent에 해당되는 lane의 차량 대기 시간의 합 + 각 agent에 해당되는 lane별 정치 차량 대수
    def _compute_local_observation(self, action):        
        state = dict()        

        for ts_ID in self._ts_IDs:                       
            # 청색불인 경우만 observation을 구한다
            if self._traffic_signal[ts_ID].current_phase_id%2 == 0:
                state[ts_ID] = list()
                waiting_vehicle_time_state = []
                halting_vehicle_number_state = []

                # # action
                # state[ts_ID].append(action[ts_ID])

                # current phase id
                state[ts_ID].append(self._traffic_signal[ts_ID].current_phase_id)

                for lane_ID in self._lane_IDs_by_ts[ts_ID]:           
                    waiting_vehicle_time = 0
                    # 각 lane에 있는 차량들의 대기시간의 합
                    for veh_ID in traci.lane.getLastStepVehicleIDs(lane_ID):                    
                        waiting_vehicle_time += traci.vehicle.getWaitingTime(veh_ID)                    
                    waiting_vehicle_time_state.append(waiting_vehicle_time)

                    # 각 lane별 정지 차량 대수
                    halting_vehicle_number_state.append(traci.lane.getLastStepVehicleNumber(lane_ID))

                # observation = action + 각 lane별 차량 대기 시간의 합 +  각 lane별 정지 차량 대수            
                state[ts_ID] = state[ts_ID]+waiting_vehicle_time_state+halting_vehicle_number_state
        return state
    
    