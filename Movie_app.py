import streamlit as st
import pandas as pd
import psycopg2
import hashlib  # For password hashing

# Function to establish a connection to PostgreSQL database
def connect_to_database():
    try:
        return psycopg2.connect(
            database="SQL_Project",
            user="postgres",
            password="Computer@02",
            host="localhost",
            port="5432"
        )
    except psycopg2.Error as e:
        st.error(f"Database connection error: {e}")
        return None

# Function to execute SQL queries
def execute_query(query, params=None):
    with connect_to_database() as conn:
        if conn is None:
            return []
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

# Function to execute SQL queries that modify data
def execute_update(query, params=None):
    with connect_to_database() as conn:
        if conn is None:
            return
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            conn.commit()

# Function to hash passwords using SHA-256
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to sign in users (authentication)
def sign_in(username, password):
    hashed_password = hash_password(password)
    query = '''
        SELECT "UserID"
        FROM "Movies Database"."Users"
        WHERE "UserName" = %s AND "Password" = %s;
    '''
    result = execute_query(query, (username, hashed_password))
    return result[0][0] if result else None


# Function to load top-rated movies


# Function to load top-rated movies with accurate overall ratings
def load_top_rated(min_rating_threshold=4.0):  # Set your threshold for top ratings
    top_rated_query = f'''
    SELECT M."Title",
           STRING_AGG(M."Genres", ', ') AS "Genres",
           ROUND(AVG(R."Rating"), 2) AS "Average Rating",
           M."ReleaseYear"
    FROM "Movies Database"."Movies" M
    LEFT JOIN "Movies Database"."Ratings" R
    ON M."MovieID" = R."MovieID"
    GROUP BY M."Title", M."ReleaseYear"
    HAVING COUNT(R."Rating") > 0 AND AVG(R."Rating") >= {min_rating_threshold}  -- Apply threshold
    ORDER BY AVG(R."Rating") DESC
    LIMIT 10;  -- Top 10 rated movies
    '''
    return execute_query(top_rated_query)


# Function to register a new user with duplicate username check
def register_user(user_name, password, age, gender):
    hashed_password = hash_password(password)
    try:
        # Check if the username already exists
        query_check_username = '''
            SELECT "UserID"
            FROM "Movies Database"."Users"
            WHERE "UserName" = %s;
        '''
        existing_user = execute_query(query_check_username, (user_name,))

        if existing_user:
            st.error("Username already exists. Please choose a different username.")
            return None

        # If the username does not exist, proceed with the registration
        user_id = int(pd.Timestamp.now().timestamp())  # Adjusted logic for UserID
        query_insert_user = '''
            INSERT INTO "Movies Database"."Users" ("UserID", "UserName", "Password", "Age", "Gender")
            VALUES (%s, %s, %s, %s, %s)
            RETURNING "UserID";
        '''
        execute_update(query_insert_user, (user_id, user_name, hashed_password, age, gender))

        # Set registration success state
        st.session_state.registration_success = True
        return user_id
    except psycopg2.Error as e:
        st.error(f"Error registering user: {e}")
        return None
    
# Function to load recommendations for new users
def load_recommendations():
    recommendations_query = '''
        SELECT "Movies"."Title", STRING_AGG("Movies"."Genres", ', ') AS "Genres", "Movies"."ReleaseYear"
        FROM "Movies Database"."Movies"
        GROUP BY "Movies"."Title", "Movies"."ReleaseYear"
        ORDER BY RANDOM() LIMIT 5;
    '''
    return execute_query(recommendations_query)

# Function to get the user's favorite genres based on their ratings
def get_user_favorite_genres(user_id):
    query = '''
        SELECT M."Genres", COUNT(R."Rating") AS "GenreCount"
        FROM "Movies Database"."Ratings" R
        JOIN "Movies Database"."Movies" M ON R."MovieID" = M."MovieID"
        WHERE R."UserID" = %s AND R."Rating" >= 4  -- Only consider high ratings
        GROUP BY M."Genres"
        ORDER BY "GenreCount" DESC;
    '''
    favorite_genres = execute_query(query, (user_id,))
    return [genre[0] for genre in favorite_genres]

# Function to get movie recommendations based on user's favorite genres, including multi-genre combinations
def get_genre_based_recommendations(favorite_genres):
    if not favorite_genres:
        return []

    # Create conditions for each genre and their combinations
    genre_conditions = [f"M.\"Genres\" ILIKE '%{genre}%'" for genre in favorite_genres]
    combination_conditions = [
        f"M.\"Genres\" ILIKE '%{genre1}|{genre2}%'"
        for genre1 in favorite_genres for genre2 in favorite_genres if genre1 != genre2
    ]
    
    # Combine the genre conditions and combination conditions into one query condition
    all_conditions = genre_conditions + combination_conditions
    combined_condition = ' OR '.join(all_conditions)

    query = f'''
        SELECT M."Title", M."Genres", M."ReleaseYear",
               COUNT(*) AS relevance_score
        FROM "Movies Database"."Movies" M
        WHERE {combined_condition}
        GROUP BY M."Title", M."Genres", M."ReleaseYear"
        ORDER BY relevance_score DESC, M."Title" ASC  -- Order by relevance and title for consistent results
        LIMIT 15;  -- Get the top 15 movies based on relevance
    '''
    recommendations = execute_query(query)
    return recommendations




# Function to fetch user's rated movies
def get_rated_movies(user_id):
    query = '''
        SELECT M."Title", R."Rating", M."ReleaseYear"
        FROM "Movies Database"."Ratings" R
        JOIN "Movies Database"."Movies" M ON R."MovieID" = M."MovieID"
        WHERE R."UserID" = %s;
    '''
    return execute_query(query, (user_id,))

# Function to fetch genres for the filter
def get_genres():
    query = '''
        SELECT DISTINCT "Genres" FROM "Movies Database"."Movies";
    '''
    return [genre[0] for genre in execute_query(query)]

# Function to search movies by title or genre and include previous ratings
def search_movies(title, genre, user_id=None):
    query = """
    SELECT
        M."Title",
        M."Genres",
        M."ReleaseYear",
        AVG(R."Rating") AS "Overall Rating"  -- Calculate average for all users' ratings
    FROM
        "Movies Database"."Movies" M
    LEFT JOIN
        "Movies Database"."Ratings" R ON M."MovieID" = R."MovieID"
    WHERE
        M."Title" ILIKE %s
        AND (%s IS NULL OR M."Genres" = %s)  -- Exact match on genre if provided
    GROUP BY
        M."MovieID"
    ;
    """

    title_search = f"%{title}%"
    genre_filter = genre if genre != "All" else None
    params = (title_search, genre_filter, genre_filter)

    results = execute_query(query, params)

    # Process the results
    processed_results = []
    for row in results:
        # Create a list from the row tuple for mutability
        row_list = list(row)
        # Replace NULL with 'Not Rated'
        if row_list[3] is None:  # Check if Overall Rating is NULL
            row_list[3] = 'Not Rated'
        else:
            row_list[3] = round(float(row_list[3]), 2)  # Round to 2 decimal places if it's numeric
        processed_results.append(row_list)

    return processed_results


# Function to rate a movie and update recommendations
def rate_movie(user_id, title, release_year, rating):
    try:
        user_id = int(user_id)
        release_year = int(release_year)
        rating = int(rating) if pd.notna(rating) else None

        if rating is None:
            st.error("Rating cannot be empty.")
            return False

        query = '''
            INSERT INTO "Movies Database"."Ratings" ("UserID", "MovieID", "Rating")
            SELECT %s, M."MovieID", %s
            FROM "Movies Database"."Movies" M
            WHERE M."Title" = %s AND M."ReleaseYear" = %s
            ON CONFLICT ("UserID", "MovieID") DO UPDATE
            SET "Rating" = EXCLUDED."Rating";
        '''
        execute_update(query, (user_id, rating, title, release_year))

        # Update recommendations immediately after rating
        update_recommendations(user_id)

        return True
    except Exception as e:
        st.error(f"Error rating movie: {e}")
        return False

# Function to update recommendations for the user and store them in session state
def update_recommendations(user_id):
    user_favorite_genres = get_user_favorite_genres(user_id)
    personalized_recommendations = get_genre_based_recommendations(user_favorite_genres)
    st.session_state.personalized_recommendations = personalized_recommendations
# Function to get top-rated movies from user's favorite genres
def get_top_rated_genre_specific_movies(favorite_genres):
    if not favorite_genres:
        return []

    genre_conditions = ' OR '.join([f"M.\"Genres\" ILIKE '%{genre}%'" for genre in favorite_genres])

    query = f'''
        SELECT M."Title", M."Genres", M."ReleaseYear", ROUND(AVG(R."Rating"), 2) AS "Average Rating"
        FROM "Movies Database"."Movies" M
        LEFT JOIN "Movies Database"."Ratings" R ON M."MovieID" = R."MovieID"
        WHERE {genre_conditions}
        GROUP BY M."Title", M."Genres", M."ReleaseYear"
        HAVING AVG(R."Rating") >= 4  -- Top-rated threshold
        ORDER BY "Average Rating" DESC
        LIMIT 5;  -- Top 5 top-rated movies from preferred genres
    '''
    top_rated_movies = execute_query(query)
    return top_rated_movies
# Function to display the recommendation page based on the user's profile
def recommendation_page(user_id):
    st.header("Movie Recommendations")

    # Check if the user has rated any movies
    rated_movies = get_rated_movies(user_id)
    
    if rated_movies:
        # Generate personalized recommendations based on user's ratings and favorite genres
        user_favorite_genres = get_user_favorite_genres(user_id)
        personalized_recommendations = get_genre_based_recommendations(user_favorite_genres)

        # Get top-rated movies specific to the user's favorite genres
        top_rated_genre_specific = get_top_rated_genre_specific_movies(user_favorite_genres)

        # Combine both personalized and top-rated genre-based recommendations
        combined_recommendations = personalized_recommendations + top_rated_genre_specific

        # Ensure recommendations are unique and limit to 10 movies only
        unique_recommendations = pd.DataFrame(
            combined_recommendations, 
            columns=["Title", "Genres", "Release Year", "Extra Column"]  # Adjust the number of columns as needed
        ).drop(columns=["Extra Column"]).drop_duplicates().head(10)  # Limit to top 10 unique movies

        st.subheader("Personalized Recommendations Based on Your Preferences:")
        st.table(unique_recommendations)
    else:
        # If the user hasn't rated any movies, provide random recommendations
        st.subheader("Random Movie Recommendations:")
        random_recommendations = load_recommendations()
        st.table(pd.DataFrame(random_recommendations, columns=["Title", "Genres", "Release Year"]))

# User Profile Page: Shows rated movies and allows for searching and rating movies
def user_profile(user_id):
    st.subheader("Your Profile")
    rated_movies = get_rated_movies(user_id)

    if rated_movies:
        st.write("Your Rated Movies:")
        st.table(pd.DataFrame(rated_movies, columns=["Title", "Rating", "Release Year"]))

    search_title = st.text_input("Search for a movie by title:")
    genres_list = ["All"] + get_genres()
    selected_genre = st.selectbox("Filter by genre:", genres_list)

    search_results = search_movies(title=search_title, genre=selected_genre, user_id=user_id)

    if search_results:
        search_results_df = pd.DataFrame(search_results, columns=["Title", "Genres", "ReleaseYear", "Overall Rating"])

        # Convert ReleaseYear to int to avoid formatting issues
        search_results_df["ReleaseYear"] = search_results_df["ReleaseYear"].astype(int)

        # Display the DataFrame
        st.write("Search Results:")
        st.dataframe(search_results_df)

        # Selecting the movie by title and release year
        movie_options = [
            f"{row['Title']} ({row['ReleaseYear']})"
            for index, row in search_results_df.iterrows()
        ]

        selected_movie = st.selectbox("Select a movie to rate:", movie_options)

        # Extracting title and release year from the selected option
        if selected_movie:  # Ensure a movie is selected before proceeding
            try:
                # Extract the movie title and release year more robustly
                selected_title = selected_movie.rsplit(" (", 1)[0]  # Extract the title before the last opening parenthesis
                selected_release_year = selected_movie.rsplit("(", 1)[1].replace(')', '')  # Extract the release year after the last opening parenthesis

                # Convert the release year to an integer to avoid errors during comparison
                selected_release_year = int(selected_release_year)

                overall_rating = search_results_df[
                    (search_results_df["Title"] == selected_title) &
                    (search_results_df["ReleaseYear"] == selected_release_year)
                ]["Overall Rating"].values[0]

                st.write(f"Release Year: {selected_release_year}")
                st.write(f"Overall Rating: {overall_rating}")

                rating = st.number_input("Rate the movie (1-5):", min_value=1, max_value=5)

                if st.button("Submit Rating"):
                    if rating:
                        if rate_movie(user_id, selected_title, selected_release_year, rating):
                            st.success("Rating submitted successfully!")
                            st.session_state['top_rated_movies'] = load_top_rated()  # Update the top-rated list
                        else:
                            st.error("Error submitting your rating.")
                    else:
                        st.warning("Please select a rating.")
            except ValueError:
                st.error("Error: Unable to parse the release year. Please check the movie title format.")
    else:
        st.write("No search results available. Try searching for another movie.")



# Function to display homepage content
def homepage():
    st.header("Welcome to Cine Suggesto!")

    # Load recommendations and display them
    recommendations = load_recommendations()
    if recommendations:
        st.subheader("Recommended Movies:")
        st.table(pd.DataFrame(recommendations, columns=["Title", "Genres", "Release Year"]))
    else:
        st.write("No recommendations available at this time.")

    # Additional content can be added here, such as news, announcements, or featured movies
    st.write("Explore our vast collection of movies and find your next favorite!")

def load_data():
    try:
        movies_df = pd.read_csv('sample_movies.csv')
        ratings_df = pd.read_csv('sample_ratings.csv')
        return movies_df, ratings_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to compute the cosine similarity matrix for recommendations
@st.cache_data
def compute_similarity(movies_df):
    if not movies_df.empty:
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
        genre_matrix = vectorizer.fit_transform(movies_df['Genres'])
        similarity_matrix = cosine_similarity(genre_matrix)
        similarity_df = pd.DataFrame(similarity_matrix, index=movies_df['Title'], columns=movies_df['Title'])
        return similarity_df
    else:
        st.error("Movies data is empty. Cannot compute similarity matrix.")
        return pd.DataFrame()

# Function to display top-rated movies
def top_rated_movies_page():
    st.header("Top Rated Movies")
    top_rated_query = '''
        SELECT "Movies"."Title", ROUND(AVG("Ratings"."Rating"), 2) AS "Average Rating", "Movies"."ReleaseYear"
        FROM "Movies Database"."Movies" AS "Movies"
        JOIN "Movies Database"."Ratings" AS "Ratings" ON "Movies"."MovieID" = "Ratings"."MovieID"
        GROUP BY "Movies"."Title", "Movies"."ReleaseYear"
        ORDER BY "Average Rating" DESC
        LIMIT 10;
    '''
    top_rated_movies = execute_query(top_rated_query)
    if top_rated_movies:
        st.table(pd.DataFrame(top_rated_movies, columns=["Title", "Average Rating", "Release Year"]))
    else:
        st.write("No top-rated movies found.")



# Main Streamlit application logic
def main():
    st.set_page_config(
        page_title="Cine Suggesto : Movie Database App",
        page_icon=":movie_camera:",
        layout="wide"
    )

    st.title("Cine Suggesto : Movie Database App")

    # Set sidebar navigation
    st.sidebar.image("/Users/ganeshnalam/Desktop/DBMS/Cine Suggesto (1).png")
    st.sidebar.header("Navigation")

    # Check if the user is logged in using session state
    if "user_id" in st.session_state:
        user_id = st.session_state["user_id"]
        page = st.sidebar.selectbox("Select a page", ["Home", "Top Rated Movies", "User Profile", "Recommendations"], index=0)

        if st.sidebar.button("Sign Out"):
            # Clear the session state for user_id when the user signs out
            del st.session_state["user_id"]
            st.success("You have been signed out.")
            st.rerun()  # Use st.rerun() to refresh the app after logging out

    else:
        page = st.sidebar.selectbox("Select a page", ["Home", "Sign In", "Register"], index=0)

    if page == "Home":
        homepage()  # Call the homepage function to show movie recommendations or any other content

    elif page == "Top Rated Movies":
        st.header("Top Rated Movies")
        top_rated = load_top_rated()  # Load top-rated movies dynamically
        if top_rated:
            st.table(pd.DataFrame(top_rated, columns=["Title", "Genres", "Average Rating", "Release Year"]))
        else:
            st.write("No top-rated movies available.")

    elif page == "User Profile" and "user_id" in st.session_state:
        user_profile(user_id)  # Display the user profile page for movie rating and user interaction

    elif page == "Recommendations" and "user_id" in st.session_state:
        recommendation_page(user_id)  # Display personalized recommendations for the logged-in user

    elif page == "Sign In":
        st.header("Sign In")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")

        if st.button("Sign In"):
            user_id = sign_in(username, password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.success("You have signed in successfully!")
                st.rerun()  # Use st.rerun() to refresh the app after signing in
            else:
                st.error("Invalid username or password.")

    elif page == "Register":
        st.header("Register")
        user_name = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        age = st.number_input("Age:", min_value=1, max_value=120)
        gender = st.selectbox("Gender:", ["Male", "Female", "Other"])

        if st.button("Register"):
            user_id = register_user(user_name, password, age, gender)
            if user_id:
                st.success("Registration successful! You can now sign in.")
            else:
                st.error("Registration failed.")

if __name__ == "__main__":
    main()
