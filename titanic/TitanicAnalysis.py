
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#데이터 경로
train_data_path = './data/train.csv'
#데이터 불러오기
df = pd.read_csv(train_data_path)
df.head()  

#열 선택
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]
df.head()

#나이가 20 이상인 사람
adult = df[df['Age'] >= 20]
adult.head()

#새로운 열 추가
# 나이를 범주형으로 변환하여 'Child'와 'Adult'로 구분
df['AgeGroup'] = df['Age'].apply(lambda x: 'Child' if x < 18 else 'Adult')
df.head()

#결측치 평균값으로 대체
print(df.isnull().sum())
df['Age'].fillna(df['Age'].mean(), inplace=True)
#inplace=True 기존 데이터 프레임을 변경된 내용으로 덮어 쓰겠다 
print(df.isnull().sum())

# 히스토그램 
plt.figure(figsize=(12, 5))

# 나이 히스토그램
plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')

#요금 히스토그램
plt.subplot(1, 2, 2)
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title('Distribution of Fare')

plt.show()

# 성별에 따른 생존율 산점도 그리기
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.show()