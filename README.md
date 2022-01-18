# YoYAK
### **Y**es **o**r **Y**es, **A**ttention with gap-sentence for **K**orean long sequence

안녕하세요 투빅스 제 13회 컨퍼런스 프로젝트 YoYAK의 레포지토리입니다. 

![](https://user-images.githubusercontent.com/43404665/149938801-b4ee73e1-7162-4768-9ea6-7a69753412fa.png)

#### YoYAK은 긴 문장도 입력으로 처리할 수 있는 한국어 Abstractive Summarization Model 입니다. 
#### YoYAK은 빅데이터 분석 및 인공지능 대표 연합동아리 투빅스 제 13회 컨퍼런스에서 발표된 모델입니다. 

### YoYAK 모델 구조 
- Initial Weight : [KoBART](https://github.com/SKT-AI/KoBART)🤣
- Attention Layer : Longformer Dilated Sliding Window
- Objective Function : Gap Sentence Generation(GSG) from [PEGASUS](https://github.com/google-research/pegasus)

### YoYAK 모델 특징
- 최대 4096 길이의 토큰까지 입력값으로 처리할 수 있습니다. 
- 최대 1024 길이의 토큰까지 생성합니다. 
- 요약 태스크에 맞춘 pretraining(GSG) 과정을 거쳤습니다. 

### 학습 데이터(약 330만 문서)
- 국민청원(2017 ~ 2019)
- 위키피디아
- 나무위키( ~ 2021.03.10.)
- 모두의 말뭉치 - 뉴스

### 모델 성능


| | Under 512| Under 512 | Under 512| Over 512 | Over 512 | Over 512|
| --- | --- | --- |---|---|---|---|
| Metric | ROUGE-1| ROUGE-2 |ROUGE-L|ROUGE-1|ROUGE-2| ROUGE-L|
| YoYAK | **0.3951**|**0.3035**|**0.3573**|**0.3486**|**0.2585**|**0.3100**|
| KoBART | 0.3500 | 0.2629|0.3085|0.3482|0.2583|0.3081|

-> 저희 YoYAK 모델이 512 토큰 이상/이하 여부와 관계없이 기존의 KoBART 모델을 abstractive summarization task에 finetuning 시킨 결과보다 더 나은 결과를 보이고 있습니다. 

### YoYAK 관련 자료
- YoYAK과 관련된 자세한 사항은 컨퍼런스 자료를 확인해주세요!
- [Slide](https://drive.google.com/file/d/1rsfD0anCyETIc-Fip4d3zAWCdUlYw75i/view?usp=sharing)
- [Youtube](https://www.youtube.com/watch?v=-OV746tzhEM)

### Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/gunny97"><img src="https://user-images.githubusercontent.com/43404665/149942570-5ba951a7-7de8-4962-ac09-ded780e3541f.jpg" width="175" height="200"><br /><sub><b>Gunwoo Kim</b></sub></td>
    <td align="center"><a href="https://github.com/hyyoka"><img src="https://user-images.githubusercontent.com/55127132/127152266-d38debab-199a-493a-bf2e-cdbc82d80e89.png" width="200" height="200"><br /><sub><b>Hyowon Cho</b></sub></td>
    <td align="center"><a href="https://github.com/kimjongwoo-cell"><img src="https://user-images.githubusercontent.com/43404665/149942566-5b5c6d0c-50f0-4733-a6d3-14cd290bb508.jpg" width="200" height="200"><br /><sub><b>Jongwoo Kim</b></sub></td>
    <td align="center"><a href="https://github.com/Lainshower"><img src="https://user-images.githubusercontent.com/43404665/149942580-f7972e58-f477-4220-9a12-1381b5b4935d.jpg" width="200" height="200"><br /><sub><b>Junwon Chang</b></sub></td>
  </tr>
</table>
<table>
  <tr align = "center">
    <td align="center"><a href="https://github.com/minjin-jeon"><img src="https://user-images.githubusercontent.com/43404665/149942576-39b308ed-a3fe-442c-8336-4ba6c0cd79b8.jpg" width="175" height="200"><br /><sub><b>MinJin Jeon</b></sub></td>
    <td align="center"><a href="https://github.com/shkim960520"><img src="https://user-images.githubusercontent.com/43404665/149942578-cdc715e9-d02c-46ea-952b-8c67dd24b564.jpg" width="200" height="200"><br /><sub><b>Sanghyeon Kim</b></sub></td>
    <td align="center"><a href="https://github.com/KimJaehee0725"><img src="https://user-images.githubusercontent.com/43404665/149942561-83eb061b-441d-4d73-9e30-f89ac778fb3b.jpg" width="200" height="200"><br /><sub><b>Jaehee Kim</b></sub></td>
  </tr>
</table>
