# Movie-recommendation-
Pick a movie, choose how many suggestions you want, and click Recommend.
To run it:
Install the libraries: pip install streamlit scikit-learn pandas
Start the app: streamlit run movie_app.py
Pick a movie, choose how many suggestions you want, and click Recommend.
How it works:
It uses content-based filtering: each movie's genres are turned into numbers with TF-IDF.
Cosine similarity then finds the movies whose genres are closest to the one you picked.
Streamlit gives you the web page.
