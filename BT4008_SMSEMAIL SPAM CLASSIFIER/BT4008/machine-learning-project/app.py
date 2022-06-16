import pickle
import streamlit as st


model=pickle.load(open("spam.pkl","rb"))
cv=pickle.load(open("vectorizer.pkl","rb"))


def main():
	st.title("Email/SMS Spam Classifier Website")
	st.subheader(":Made By Mayadhar  With Python & Streamlit")
	msg=st.text_input("Enter the Text : ")
	if st.button("Predict"):
		data=[msg]
		vect=cv.transform(data).toarray()
		prediction=model.predict(vect)
		result=prediction[0]
		if result==1:
			st.error("This is Spam Message")
		else:
			st.success("This a Ham Message")

main()