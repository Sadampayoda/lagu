from flask import Flask, render_template,request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics.pairwise import cosine_similarity



app = Flask(__name__)



    

def proses(df,user_song_matrix):
    

    target_column = 'Rating'

    # Memisahkan variabel target dan fitur
    y = df[target_column]  # Gantilah 'df' dengan nama DataFrame Anda
    X = user_song_matrix.transpose()

    # Membangun model Random Forest
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)

    # Bagi data menjadi set pelatihan dan set pengujian
    train_data, test_data, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Latih model dengan data pelatihan
    rf_model.fit(train_data, y_train)

    return rf_model





# @app.route('/')
# def hello_world():
    

@app.route('/', methods=['POST','GET'])
def recommend():
    if request.method == 'POST':
        song_title = request.form['song_title']

        # Fungsi untuk memberikan rekomendasi berbasis judul lagu
        def recommends(song_title):
            # df = pd.read_csv("song.csv")
            # df.columns = ["Id User", "song_title", "Artist", "Date of Release", "Description", "Metascore", "Rating"]


            # # df['Description'] = df['Description'].fillna(method= "bfill",inplace=False)
            # # df['Metascore'] = df['Metascore'].fillna(method= "bfill",inplace=False)
            # # df['Rating'] = df['Rating'].fillna(method= "bfill",inplace=False)
            # df['Description'] = df['Description'].bfill(inplace=False)
            # df['Metascore'] = df['Metascore'].bfill(inplace=False)
            # df['Rating'] = df['Rating'].bfill(inplace=False)
            # # # Pilih kolom-kolom yang diperlukan untuk sistem rekomendasi
            # selected_columns = ["Id User", "song_title", "Rating"]
            # df_recommender = df[selected_columns]
            # df = df_recommender.head(20000) 

            # user_song_matrix = df.pivot_table(index='Id User', columns='song_title', values='Rating', fill_value=0)
            
            # # user_song_matrix = user_song_matrix.transpose()


            # song_similarity = cosine_similarity(user_song_matrix)

            # song_index = df[df['song_title'] == song_title].index[0]
            # # Mengambil skor kemiripan cosine similarity untuk lagu tersebut
            # similar_scores = list(enumerate(song_similarity[song_index]))
            # # Mengurutkan lagu berdasarkan skor kemiripan
            # similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
            # # Mengambil 10 lagu teratas yang paling mirip
            # top_recommendations = similar_scores[1:11]

            # # Mengambil judul lagu dari indeks yang direkomendasikan
            # recommended_song_titles = [df_recommender.iloc[i[0]]['song_title'] for i in top_recommendations]
            # df = pd.read_csv('rekom.csv')
            # user_song_matrix = pd.read_csv("dataset.csv")
            # Pastikan bahwa 'user_song_matrix' dan 'df' memiliki baris yang sama:
            # user_song_matrixs = user_song_matrix.transpose()
            # df = df.iloc[:user_song_matrix.shape[0], :]

            df = pd.read_csv("song.csv")
            df.columns = ["Id User", "song_title", "Artist", "Date of Release", "Description", "Metascore", "Rating"]
            if song_title not in df['song_title'].values:
                return "Nama lagu tidak valid. Silakan masukkan lagu yang valid."
            df['Description'] = df['Description'].bfill(inplace=False)
            df['Metascore'] = df['Metascore'].bfill(inplace=False)
            df['Rating'] = df['Rating'].bfill(inplace=False)

            # df['Description'] = df['Description'].fillna(method= "bfill",inplace=False)
            # df['Metascore'] = df['Metascore'].fillna(method= "bfill",inplace=False)
            # df['Rating'] = df['Rating'].fillna(method= "bfill",inplace=False)

            selected_columns = ["Id User", "song_title", "Rating"]
            df_recommender = df[selected_columns]
            df_sample = df_recommender.head(20000) 
            # print(df)
            # user_song_matrix = df_sample.pivot_table(index='Id User', columns='song_title', values='Rating', fill_value=0)
            # user_song_matrix
            user_song_matrix = pd.read_csv("user_song_matrix.csv")
            song_similarity = cosine_similarity(user_song_matrix)

            song_index = df[df['song_title'] == song_title].index[0]
            # Mengambil skor kemiripan cosine similarity untuk lagu tersebut
            similar_scores = list(enumerate(song_similarity[song_index]))
            # Mengurutkan lagu berdasarkan skor kemiripan
            similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
            # Mengambil 10 lagu teratas yang paling mirip
            top_recommendations = similar_scores
            # print(top_recommendations)

            
            recommended_songs = []


            recommended_song_titles = set()

            for i in top_recommendations:
                song_index = i[0]
                song_title = df_recommender.loc[song_index, 'song_title']

                
                if song_title not in recommended_song_titles:
                    rating = df_recommender.loc[song_index, 'Rating']

                    recommended_songs.append({
                        'song_title': song_title,
                        'prediction': rating
                    })

                    
                    recommended_song_titles.add(song_title)

                
                if len(recommended_songs) == 10:
                    break











            # # print(df['song_title'].unique())
            # # return 'p'
           
            # try:
            #     song_ratings = user_song_matrix[song_title]
            # except KeyError:
            #     return "Nama lagu tidak valid. Silakan masukkan lagu yang valid."

            # # Temukan lagu yang belum dilihat oleh pengguna
            # unseen_songs = song_ratings[song_ratings == 0].index
            # print("Columns in df:")
            # print(unseen_songs)
            # return 'oke'
            # Lakukan prediksi untuk lagu yang belum dilihat
            # rf_model = proses(df,user_song_matrixs)
            # rf_model = joblib.load('model.joblib')
            # if set(user_song_matrixs.columns) != set(df.columns[2:]):
            #     return "Kolom pada data prediksi tidak sesuai dengan model yang dilatih."
            # predictions = rf_model.predict(df.loc[:, unseen_songs].transpose())

            # Gabungkan hasil prediksi dengan lagu yang belum dilihat
            # recommendations = pd.DataFrame({'Song': unseen_songs, 'Prediction': predictions})

            # Urutkan lagu berdasarkan prediksi
            # top_recommendations = recommendations.sort_values(by='Prediction', ascending=False).head(10)

            return recommended_songs

        # Dapatkan rekomendasi untuk lagu tertentu
        top_recommendations = recommends(song_title)
        # return top_recommendations
        # return '0'
        # Tampilkan hasil di konsol
        # print(len(top_recommendations))
        print(f'Top 10 Rekomendasi untuk lagu "{song_title}":\n{top_recommendations}')
        # print(top_recommendations[0])
        
        return render_template('index.html', song_title=song_title, recommendations=top_recommendations,loading=True)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True,port=5000)