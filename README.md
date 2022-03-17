# 🚥 강화학습을 이용한 도로 통행 최적화 신호 모델
- **실시간으로 차선 별 정보를 얻을 수 있는 ‘도로공간’ 환경에서 공간정보를 기반으로 신호를 조절하여 차량 통행의 최적화를 달성하는 것**
  - **최적화의 기준**은 차량의 정지 횟수와 대기 시간을 감소하여 최대한 진출차선으로 빠져나가게 하는 것
  - ‘**도로공간**’은 V2V, V2I가 구현되어서 실시간으로 도로의 정차 차량 대수, 정차 시간, 신호 정보 등의 공간정보를 얻을 수 있는 환경

## 🛣︎ 학습 환경
**🛠︎ 주요 API**
- RLlib
- SUMO(Simulation of Urban MObility)

**🛠︎ 도로 네트워크**

| case03 | case06 |
| :------------: | :-------------: |
|교차로가 2개인 도로교통 환경|교차로가 6개인 이천시 도로교통 환경|
|![image](https://user-images.githubusercontent.com/58590260/137259592-73087132-a10d-4701-927d-6c3a9eabe89c.png)|![image](https://user-images.githubusercontent.com/58590260/137259384-c9220f41-e80b-44f4-adc6-984875ef6786.png)|

**🛠︎ 모델** :  DQN

**🛠︎ 에이전트** : 신호등
| case03 | case06 |
| :------------: | :-------------: |
|![image](https://user-images.githubusercontent.com/58590260/137367516-463c14a2-0b3b-410c-9944-2742c8308d73.png)|![image](https://user-images.githubusercontent.com/58590260/137366199-4e9913d8-d964-4683-94d6-c69c670f9f21.png)|

**🛠︎ 상태** : 현재 청색불 ID + [각 lane별 차량 대기 시간의 합] + [각 lane별 정지 차량 대수]

**🛠︎ 행동**
| random traffic signal model | restricted traffic signal mode |
| :------------: | :-------------: |
| 각 신호등의 청신호의 수 |  0 : 현재 신호 유지, 1: 다음 청신호로 변경 |
  
**🛠︎ 보상**
- 지나간 차량 대수 - 정지 차량 대수

**🛠︎ 조건**
| |  한 에피소드당 시뮬레이션 진행 시간 | 훈련 에피소드 횟수 |
| :------------: | :------------: | :-------------: |
| case03 | 3600초 | 15000회|
| case06 | 18000초 | 144회 |

| | random traffic signal model | restricted traffic signal mode |
| :-----------------:| :------------: | :-------------: |
| 시뮬레이션 | ![random_저화질](https://user-images.githubusercontent.com/58590260/137359804-020928c3-2423-4cd4-9f84-06e627c683cb.gif) |  ![restricted_저화질](https://user-images.githubusercontent.com/58590260/137358485-3e230d2f-3c25-4e41-8dfc-8060bf9d1df3.gif) |
| 신호 순서 | 무작위로 청색불 신호 선택하여 진행 | 순서대로 신호 진행 |
| 점등 시간 & 주기 | 청색불의 최소, 최대 점등 시간이 없음 | 각 신호등에는 주기가 있음 |

