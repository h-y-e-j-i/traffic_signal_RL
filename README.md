# ~~~~~~~~~ 강화학습 신호 모델 
## 배경과 목적


- DQN 어쩌구저쩌구
- 돌아가는 모양 어쩌구 저쩌구
## 학습 환경
- **도로 네트워크**
  - case03 : 교차로 ![image](https://user-images.githubusercontent.com/58590260/137259592-73087132-a10d-4701-927d-6c3a9eabe89c.png)
  - case06 : 교차로로 ![image](https://user-images.githubusercontent.com/58590260/137259384-c9220f41-e80b-44f4-adc6-984875ef6786.png)
- **알고리즘** : DQN
- **에이전트** : 신호등
- **상태**
  - 현재 청색불 ID + [각 lane별 차량 대기 시간의 합] + [각 lane별 정지 차량 대수]
- **행동**
  - random model : 각 신호등의 청신호의 수
  - restricted model : 0 or 1
    - 0 : 현재 신호 유지, 1: 다음 청신호로 변경
- **보상**
  - 지나간 차량 대수 - 정지 차량 대수
- **조건**
  - random model
    - 무작위로 청색불 신호 선택하여 진행
    - 청색불의 최소, 최대 점등 시간이 없음
  - restricted model
    - 순서대로 신호 진행
    - 청색불의 최소, 최대 점등 시간이 존재
    - 주 청어어어어ㅓ처처처처ㅓㅇ새부

