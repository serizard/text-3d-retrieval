# text-3d-retrieval
쌈뽕한 프로젝트

## to-do list

### 세민
- download_dataset.py
    - modelnet40 다운받아 data/original 폴더에 저장하는 코드 작성

- main.py process_input 함수
    - user_input 받아서 clip 모델로 text_feature 추출하여 return하는 코드 작성

- main.py retrieve_3d 함수
    - text_feature 입력으로 받아 modelnet_embed 데이터와 유사도 구한 뒤, top-k개 모델 파일 시각화하는 코드 작성

### 도형
- utils/to_pcd.py
    - data/original 내 .off 파일들 .npy (포인트클라우드) 파일로 변환하여 data/transformed 파일에 저장하는 코드 작성

- preprocess_data.py
    - openshape의 3d 인코더를 사용하여 data/transformed 파일들의 3d shape feature를 추출한 뒤, modelnet_embed 파일에 pickle 형식으로 저장하는 코드 작성 (참고: https://github.com/Colin97/OpenShape_code/blob/master/src/example.py)