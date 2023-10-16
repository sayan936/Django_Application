from django.shortcuts import render
from joblib import load
import json



model = load('./savedModels/model.joblib')
tfidf = load('./savedModels/tfidf.joblib')

def spam(request):

    if request.method == 'POST':
        text = request.POST['textarea_1']  # Assuming 'text' is a key in your JSON data
        X_test = tfidf.transform([text]).toarray()
        y_pred = model.predict(X_test)
        print(y_pred)
        return render (request,'index.html', {'result':y_pred})

    return render(request,'index.html')

