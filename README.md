# PredictionEngine

## 가상 환경 생성
> 아래 예시는 Conda에서 가상 환경을 생성하는 방법이며, 
python 3.8의 환경 필요 (python3 이상)


```bash
$ conda create -n vibe_test python=3.8
$ conda activate vibe_test
```

## 기존 필요 라이브러리 설치
#### 라이브러리 목록
```
Keras
numpy
pandas
scikit-learn
sklearn
tensorflow
```
#### 라이브러리 설치
```bash
$ pip install -r requirements.txt
```

## 휠파일 인스톨
```bash
$ pip install vibe-0.2-py3-none-any.whl --force-reinstall
```

## VIBE_inference_whl 실행

```bash
- 의 (if __name__ == "__main__":)에 테스트 코드  하위 코드는 테스트 코드로 해당 코드를 실행하면 예측된 값을 출력함
- 주석 참고

```
