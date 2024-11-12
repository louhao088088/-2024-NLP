import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 数据库
data = {
    'text': [
        'Hello', 'How is the weather today', 'Good morning', 'Is it raining outside', 'The weather is clear', 
        'How are you', 'Where to go tomorrow', 'What is the temperature', 'Hi', 'Is it hot today', 
        'Weather forecast', 'Good luck', 'What do you think', 'The weather is nice today', 'Good morning everyone', 
        'What clothes to wear today', 'Is it cold outside', 'Hello everyone', 'Thank you', 'Goodbye', 
        'I feel great today', 'I am so happy', 'This is the best day ever', 'I feel so sad', 'I am angry', 
        'Let’s meet at 5 PM', 'What time is it?', 'It’s 8 o’clock', 'I have an appointment at 3 PM', 'Breaking news: Earthquake in Japan', 
        'The stock market crashed today', 'The president gave a speech', 'Today is a holiday', 'There’s an event at the park', 'I am looking forward to the weekend'
    ],
    'category': [
        'Greeting', 'Weather', 'Greeting', 'Weather', 'Weather', 'Greeting', 'Other', 'Weather', 'Greeting', 'Weather', 
        'Weather', 'Greeting', 'Other', 'Weather', 'Greeting', 'Weather', 'Weather', 'Greeting', 'Other', 'Greeting', 
        'Emotion', 'Emotion', 'Emotion', 'Emotion', 'Emotion', 'Time', 'Time', 'Time', 'Time', 
        'News', 'News', 'News', 'Event', 'Event','Event'
    ]
}

df = pd.DataFrame(data)

# 使用 TF-IDF（词频-逆文档频率）向量化
vectorizer = TfidfVectorizer()


X = vectorizer.fit_transform(df['text'])

y = df['category']

# 分词
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = MultinomialNB()

model.fit(X_train, y_train)
#获得预测
y_pred = model.predict(X_test)



print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))



def predict_category(text):
    text_vec = vectorizer.transform([text])
    
    prediction = model.predict(text_vec)
    
    return prediction[0]


user_input = input("Enter text for classification: ")  
predicted_category = predict_category(user_input)
print(f"Input: {user_input}\nPredicted category: {predicted_category}")
