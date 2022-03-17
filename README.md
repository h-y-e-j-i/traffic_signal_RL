# 🚥 강화학습을 이용한 도로 통행 최적화 신호 모델 🚥
- **실시간으로 차선 별 정보를 얻을 수 있는 ‘도로공간’ 환경에서 공간정보를 기반으로 신호를 조절하여 차량 통행의 최적화를 달성하는 것**
  - **최적화의 기준**은 차량의 정지 횟수와 대기 시간을 감소하여 최대한 진출차선으로 빠져나가게 하는 것
  - ‘**도로공간**’은 V2V, V2I가 구현되어서 실시간으로 도로의 정차 차량 대수, 정차 시간, 신호 정보 등의 공간정보를 얻을 수 있는 환경

## 🖥️ 사용 언어 & API
- Python
- RLlib
- SUMO(Simulation of Urban MObility)


## 🛣︎ 학습 환경
**🛠︎ 모델** :  DQN

**🛠︎ 도로 네트워크**

| case03 | case06 |
| :------------: | :-------------: |
|교차로가 2개인 도로교통 환경|교차로가 6개인 이천시 도로교통 환경|
|![image](https://user-images.githubusercontent.com/58590260/137259592-73087132-a10d-4701-927d-6c3a9eabe89c.png)|![image](https://user-images.githubusercontent.com/58590260/137259384-c9220f41-e80b-44f4-adc6-984875ef6786.png)|


**🛠︎ 에이전트** : 신호등
| case03 | case06 |
| :------------: | :-------------: |
|![image](https://user-images.githubusercontent.com/58590260/137367516-463c14a2-0b3b-410c-9944-2742c8308d73.png)|![image](https://user-images.githubusercontent.com/58590260/137366199-4e9913d8-d964-4683-94d6-c69c670f9f21.png)|

**🛠︎ 상태** : 현재 청색불 ID + [각 lane별 차량 대기 시간의 합] + [각 lane별 정지 차량 대수]

**🛠︎ 행동**
| random traffic signal model | restricted traffic signal mode |
| :------------: | :-------------: |
| 각 신호등의 청신호의 수 |  0 : 현재 신호 유지, 1: 다음 청신호로 변경 |
  
**🛠︎ 보상** : 지나간 차량 대수 - 정지 차량 대수

**🛠︎ 조건**
- 도로 네트워크 별

| |  한 에피소드당 시뮬레이션 진행 시간 | 훈련 에피소드 횟수 |
| :------------: | :------------: | :-------------: |
| case03 | 3600초 | 15000회|
| case06 | 18000초 | 144회 |

- 모델별

| | random traffic signal model | restricted traffic signal mode |
| :-----------------:| :------------: | :-------------: |
| 시뮬레이션 | ![random_저화질](https://user-images.githubusercontent.com/58590260/137359804-020928c3-2423-4cd4-9f84-06e627c683cb.gif) |  ![restricted_저화질](https://user-images.githubusercontent.com/58590260/137358485-3e230d2f-3c25-4e41-8dfc-8060bf9d1df3.gif) |
| 신호 순서 | 무작위로 청색불 신호 선택하여 진행 | 순서대로 신호 진행 |
| 점등 시간 & 주기 | 청색불의 최소, 최대 점등 시간이 없음 | 각 신호등에는 주기가 있음 |

## 📃 연구 결과
### 1️⃣ case 03
![Untitled](https://user-images.githubusercontent.com/58590260/158837379-66ee9a1f-f248-4f21-b357-a411fd803799.png)
![Untitled (1)](https://user-images.githubusercontent.com/58590260/158837487-fdb523eb-25cc-4ed1-b590-2149727a39ed.png)
### 2️⃣ case 06
![Untitled (2)](https://user-images.githubusercontent.com/58590260/158838053-f8d97819-f3fc-4b2f-922b-cbc3eaf46f23.png)
![Untitled (3)](https://user-images.githubusercontent.com/58590260/158837942-cb296926-a679-4bfa-bf61-373644b9640b.png)
### 3️⃣ 제약조건이 없는 모델
![Untitled (4)](https://user-images.githubusercontent.com/58590260/158838485-b80c3f29-0e52-4623-9064-742542fd4fb6.png)
- 훈련이 잘 된 신호등은 진입 차선의 차량을 **인식**하여 바로 청색 불 신호로 바뀜
- 일부 신호등은 **진입 차선의 차량과 상관없이** 신호가 오래 유지 되고 있음. 다른 방향의 신호를 고려한다면 개선해야 함.
- 청색 불 점등 시간이 **너무 짧아** 현실성이 낮음. 따라서 **점등 시간 조건**을 추가하면 좋은 모델로 개발 될 것으로 생각함
### 4️⃣ 제약조건이 있는 모델
![image](https://user-images.githubusercontent.com/58590260/158840515-c67a3641-b508-4ce6-ac48-86cd8ad323df.png)
- 제약 조건으로 인해 무작위 신호 선택 모델의 결과와 비교했을 때 바로 직진 신호를 바꾸지 못하는 한계점이 보이지만, 현실 세계의 도로 환경을 고려한다면 제약 조건이 없는 모델보다는 **현실성 있는 신호 체계**라고 생각함
- **진입 차선의 교통량에 좀 더 빨리 반응하는 신호 모델**로 개선해야함


