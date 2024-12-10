# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
######################################################################
import util
from pydantic import BaseModel, Field
from porter_stemmer import PorterStemmer

import numpy as np
import re
import random


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Blockbuster Bot'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        self.user_ratings = np.zeros(len(self.titles))
        self.recs = []
        self.datapoints = 1
        self.rec_index = 0
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hey! I hear you are looking for a movie! I will make some recommendations for you based on your past preferences. Tell me about a movie that you have seen."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "I hope you enjoy your movie! See you later!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    
    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies.""" +\
        """Begin by asking the user about their recent movie experiences.
           If the conversation veers away from movies, remember to steer it back without providing a response to the off-topic subject.
           As the user provides their opinions on movies, identify whether they express a like or dislike.
           Acknowledge their sentiment by repeating the movie title and ask for their thoughts on another movie. 
           Once you've discussed five movies with the user, it's time to use this information to ask them if they would like you to suggest a movie they might enjoy."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
        else:
            ##
            if re.search(r'\byes\b', line, flags=re.IGNORECASE):
                if self.rec_index >= len(self.recs):
                    return "Sorry, I don't have any more recommendations for you! Tell me more about some movies you've seen so I can make new recommendations!"

                self.rec_index += 1
                rec_movie_index = self.recs[self.rec_index]
                # whats with this whole self.titles situation
                additional_recommendations = [f"You might also like \"{self.titles[rec_movie_index][0]}\". Care for another recommendation?",
                                              f"Perhaps \"{self.titles[rec_movie_index][0]}\" would also be to your liking. Should I recommend another one?",
                                              f"I think you might appreciate \"{self.titles[rec_movie_index][0]}\" as well. Interested in another movie recommendation?"]
                return additional_recommendations[random.randrange(3)]

            if re.search(r'\bno\b', line, flags=re.IGNORECASE):
                self.recs = []
                self.rec_index = 0
                return "Ok! If you tell me more about some movies you've seen so I can make new recommendations!"

            # also whats goin on with this why not just geq 5
            if self.datapoints % 5 and self.datapoints != 0:
                self.recs = self.recommend(self.user_ratings, self.ratings, k=5)
                rec_movie_index = self.recs[self.rec_index]
                recommendation_responses = [f"According to my calculations, you should enjoy \"{self.titles[rec_movie_index][0]}\". Want another recommendation?",
                        f"Based on my analysis,\"{self.titles[rec_movie_index][0]}\" should be a good fit for you. How about another recommendation?",
                        f"From what I've calculated, \"{self.titles[rec_movie_index][0]}\" seems like a movie you would enjoy. Interested in another recommendation?"]
                return recommendation_responses[random.randrange(3)]

            titles = self.extract_titles(line)
            if len(titles) == 0:
                where_title = [f"I'm sorry, I didn't recognize a title in what you just said. Can you please tell me about a movie you've seen recently?",
                               f"Apologies, I couldn't identify any movie title in your response. Could you mention a recent film you've watched?",
                               f"It seems I missed a movie title in your message. Would you mind telling me about a film you watched recently?"]
                return where_title[random.randrange(3)]

            movie_indices = []
            for title in titles:
                curr_indices = self.find_movies_by_title(title)
                movie_indices.append(curr_indices)

            response = ""
            ##
            for i, movie_idx in enumerate(movie_indices):
                if len(movie_idx) == 0:
                    what_bruh = [f"I've never heard of \"{titles[i]}\"! Can you tell me about a different movie?",
                                 f"\"{titles[i]}\" doesn't ring a bell for me. Can you share a different film title?",
                                 f"I'm not familiar with \"{titles[i]}\". Could you mention another movie?"]
                    return what_bruh[random.randrange(3)]
                else:
                    #######
                    sentiment = self.extract_sentiment(line)
                    selected_movie_index = movie_idx[0]
                    if sentiment == 1:
                        positive = {f"Awesome! You liked \"{titles[i]}\"! Please tell me about another movie!",
                                    f"That's wonderful to hear you liked \"{titles[i]}\"! Can you tell me about another film you've seen?",
                                    f"Great! You enjoyed \"{titles[i]}\"! Could you share another movie you liked?"}
                        response += positive[random.randrange(3)]
                    if sentiment == -1:
                        negative = {f"Ah I see, you didn't like \"{titles[i]}\". Please tell me about another movie!",
                                    f"Got it, \"{titles[i]}\" wasn't your cup of tea. How about mentioning another movie?",
                                    f"So, \"{titles[i]}\" didn't appeal to you. Can you share another film title?"}
                        response += negative[random.randrange(3)]
                    if sentiment == 0:
                        neutral = {f"I'm sorry, I'm not sure how you feel about \"{titles[i]}\". Can you elaborate please?",
                                    f"Apologies, I'm unclear about your thoughts on \"{titles[i]}\". Could you elaborate a bit more?",
                                    f"I'm not certain how you feel about \"{titles[i]}\". Can you provide more details, please?"}
                        response += neutral[random.randrange(3)]
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response
    

    #########################

    if self.llm_enabled:
    response = "I processed {} in LLM Programming mode!".format(line)
else:
    yes_pattern = r'\byes\b|\byeah\b|\bya\b|\bye\b|\bype\b'
    no_pattern = r'\bno\b|\bnah\b|\bnope\b'

    if re.search(yes_pattern, line, re.IGNORECASE):
        if self.rec_index >= len(self.recs):
            return "Sorry, I've exhausted my recommendations! Please tell me more about movies."

        self.rec_index += 1
        rec_movie_index = self.recs[self.rec_index]
        additional_recommendations = [
            "You might also enjoy \"{}\". Would you like another recommendation?".format(self.titles[rec_movie_index][0]),
            "Another great choice would be \"{}\". Shall we continue?".format(self.titles[rec_movie_index][0]),
            "I would also recommend \"{}\" as well. Interested in another movie recommendation?".format(self.titles[rec_movie_index][0])
        ]
        return additional_recommendations[random.randint(0, len(additional_recommendations) - 1)]

    elif re.search(no_pattern, line, re.IGNORECASE):
        self.recs = []
        self.rec_index = 0
        return "OK! If you tell me more about some movies you've seen so I can make new recommendations!"

    elif self.datapoints % 5 == 0 and self.datapoints != 0:
        self.recs = self.recommend(self.user_ratings, self.ratings, k=5)
        rec_movie_index = self.recs[self.rec_index]
        response = "According to my calculations, you should enjoy \"{}\". Want another recommendation?".format(self.titles[rec_movie_index][0])
        return response

titles = self.extract_titles(line)
if len(titles) == 0:
    no_title = [
        "Hmm, I'm sorry, I didn't recognize a title in what you just said. Can you please tell me about a movie you've seen recently?",
        "Apologies, I couldn't identify any movie title in your response. Could you mention a recent film you've watched?",
        "I'm sorry, I missed a movie title in your message. Would you mind telling me about a film you watched recently?"
    ]
    return no_title[random.randint(0, len(no_title) - 1)]

movie_indices = []
for title in titles:
    curr_indices = self.find_movies_by_title(title)
    movie_indices.append(curr_indices)

response = ""
for i, movie_idx in enumerate(movie_indices):
    if len(movie_idx) == 0:
        unseen = [
            "I've never heard of \"{}\", sorry! Tell me about another movie.".format(titles[i]),
            "I don't see \"{}\" in my database, sorry. Please tell me about another movie.".format(titles[i]),
            "I'm not familiar with \"{}\". Could you mention another movie title?".format(titles[i])
        ]
        response += unseen[random.randint(0, len(unseen) - 1)]
    else:
        sentiment = self.extract_sentiment(line)
        selected_movie_index = movie_idx[0]
        
        if sentiment == 1:
            positive = [
                "Alright, so you enjoyed \"{}\"! Tell me what you thought of another movie. ".format(titles[i]),
                "Sweet, it seems you liked \"{}\"! Let me know what you thought of another movie.".format(titles[i]),
                "I'm glad to hear you liked \"{}\". Please tell me about another movie you liked.".format(titles[i])
            ]
            response += positive[random.randint(0, len(positive) - 1)]
            self.user_ratings[selected_movie_index] = 1

        elif sentiment == -1:
            negative = [
                "Ah, so you didn't like \"{}\". Sorry to hear that, let's talk about another movie.".format(titles[i]),
                "It appears you did not enjoy \"{}\", sorry about that. Give me another movie and we'll try again.".format(titles[i]),
                "Sorry you didn't like \"{}\". Let me know what you think about another movie!".format(titles[i])
            ]
            response += negative[random.randint(0, len(negative) - 1)]
            self.user_ratings[selected_movie_index] = -1

        elif sentiment == 0:
            neutral = [
                "I'm sorry, I'm not sure how you feel about \"{}\". Can you elaborate please?".format(titles[i]),
                "Based on your input, I can't tell if you liked \"{}\". Let me         
                
        if sentiment == 1:
            positive = [
                "Alright, so you enjoyed \"{}\"! Tell me what you thought of another movie. ".format(titles[i]),
                "Sweet, it seems you liked \"{}\"! Let me know what you thought of another movie.".format(titles[i]),
                "I'm glad to hear you liked \"{}\". Please tell me about another movie you liked.".format(titles[i])
            ]
            response += positive[random.randint(0, len(positive) - 1)]
            self.user_ratings[selected_movie_index] = 1
            self.datapoints += 1

        elif sentiment == -1:
            negative = [
                "Ah, so you didn't like \"{}\". Sorry to hear that, let's talk about another movie.".format(titles[i]),
                "It appears you did not enjoy \"{}\", sorry about that. Give me another movie and we'll try again.".format(titles[i]),
                "Sorry you didn't like \"{}\". Let me know what you think about another movie!".format(titles[i])
            ]
            response += negative[random.randint(0, len(negative) - 1)]
            self.user_ratings[selected_movie_index] = -1
            self.datapoints += 1

        elif sentiment == 0:
            neutral = [
                "I'm sorry, I'm not sure how you feel about \"{}\". Can you elaborate please?".format(titles[i]),
                "Based on your input, I can't tell if you liked \"{}\". Let me know more about it.".format(titles[i]),
                "I can't understand if you liked \"{}\" or not. Would you mind telling me more?".format(titles[i]),
                "I'm having trouble gauging your opinion on \"{}\". Can you elaborate on your thoughts?".format(titles[i]),
                "It seems I'm unsure about your feelings towards \"{}\". What did you think of it?".format(titles[i])
            ]
            response += neutral[random.randint(0, len(neutral) - 1)]
            self.datapoints += 1

return response



    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text
    
    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        
        #create empty list of movie titles
        titles = []
        
        #start index when it finds the first double quotation marks (indicating a movie title)
        start = preprocessed_input.find('"')
        
        #while there are more double quotation marks in the text
        while start != -1:
            
            #find second double quotation mark (indicating the end to a movie title)
            end = preprocessed_input.find('"', start + 1)
            
            #if there is a complete pair of double quotation marks
            if end != -1:
            
                #extract movie title (from starting index to end)
                title = preprocessed_input[start + 1:end]
                
                #add the movie title to the list
                titles.append(title)
                
                #find next set of double quotation marks
                start = preprocessed_input.find('"', end + 1)
            else:
                return titles
            
        return titles
        
        #CHANGE THIS MORE
        #need to figure out if statement
       
    # needs to add case where the year is in the title and article is in the title
    def find_movies_by_title(self, title):
        matches = []
        search_title = title.lower()

        # Extract year from the search title, if present
        year_pattern = r"\((\d{4})\)"
        year_match = re.search(year_pattern, search_title)
        search_year = year_match.group(1) if year_match else None
        search_title = search_title[:year_match.start()].strip() if year_match else search_title

        # Handle leading articles in the search title
        article_pattern = r"^(a|an|the)\s+"
        article_match = re.search(article_pattern, search_title)
        search_article = article_match.group().strip() if article_match else None
        search_title = re.sub(article_pattern, "", search_title) if article_match else search_title

        for index, (movie_title, _) in enumerate(self.titles):
            movie_title_lower = movie_title.lower()

            # Extract year from movie title, if present
            movie_year_match = re.search(year_pattern, movie_title_lower)
            movie_year = movie_year_match.group(1) if movie_year_match else None
            movie_title_stripped = movie_title_lower[:movie_year_match.start()].strip() if movie_year_match else movie_title_lower

            # Handle leading articles in the movie title
            movie_article_match = re.search(r", (a|an|the)$", movie_title_stripped)
            movie_article = movie_article_match.group(1) if movie_article_match else None
            movie_title_stripped = movie_title_stripped[:-len(movie_article_match.group())].strip() if movie_article_match else movie_title_stripped

            # Compare titles and years to find matches
            if search_title == movie_title_stripped:
                if search_year == movie_year:
                    matches.append(index)
                elif search_year is None:
                    matches.append(index)
                elif search_article is not None and movie_article is not None:
                    if f"{search_article} {search_title}" == f"{movie_title_stripped}, {movie_article}":
                        if search_year == movie_year:
                            matches.append(index)
                        elif search_year is None:
                            matches.append(index)
        return matches



        

        """<<<<<<< HEAD
        #can one of yall check this for me - not sure how the word databases should look
        =======
        # could use porter stemmer to extract word stems here
        >>>>>>> 7b25575aa732606c5d19b19de7aedae73e6fd9a6"""
    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
<<<<<<< HEAD
        #initialize sentiment score
        """Extract a sentiment rating from a line of pre-processed text, considering negations."""

        stemmer = PorterStemmer()

        stemmed_sentiment = {stemmer.stem(key): value for key, value in self.sentiment.items()}

        # Streamlined text cleaning with clear regex operations
        cleaned_input = re.sub(r'"\s*([^"]*?)\s*"', '', preprocessed_input).strip()
        cleaned_input = re.sub(r'\s+', ' ', cleaned_input)

        tokens = cleaned_input.split()
=======
        words = preprocessed_input.lower().split()
>>>>>>> 4c376f1c7f77ef206d938be05bdb4273e366fbea
        sentiment_score = 0
        is_negation = False

<<<<<<< HEAD
        for token in tokens:
            if token.lower() in ['not', 'never', "didn't", "don't", "no"]:
                is_negation = not is_negation
                continue  # Move to the next token if current one is a negation

            stemmed_token = stemmer.stem(token.lower())
            if stemmed_token in stemmed_sentiment:
                value = -1 if stemmed_sentiment[stemmed_token] == 'neg' else 1
                if is_negation:
                    value *= -1
                    is_negation = False  # Reset negation after applying it
                sentiment_score += value

        return 1 if sentiment_score > 0 else -1 if sentiment_score < 0 else 0

=======
        for word in words:
            if word in self.sentiment:  # assuming sentiment is a dictionary with 'pos' or 'neg' as values
                word_sentiment = self.sentiment[word]
                if word_sentiment == 'pos':
                    sentiment_score += 1
                elif word_sentiment == 'neg':
                    sentiment_score -= 1

        # Final sentiment determination
        if sentiment_score > 0:
            return 1
        elif sentiment_score < 0:
            return -1
        else:
            return 0
>>>>>>> 4c376f1c7f77ef206d938be05bdb4273e366fbea

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        
       ##iterate through ratings
        #for rating in binarized_ratings:
        
            #above threshold
          #  if rating > threshold:
          #      rating = 1
            #below threshold
         #   elif (rating <= threshold) and (ratings > 0):
          #      rating = -1
            
        binarized_ratings[(ratings <= threshold) & (ratings > 0)] = -1
        binarized_ratings[(ratings > threshold)] = 1

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0
        
        #find magnitude of vectors
        u_magnitude = np.linalg.norm(u)
        v_magnitude = np.linalg.norm(v)
        
        #if the movie has not been seen
        if u_magnitude == 0 or v_magnitude == 0:
            return similarity
        
        #dot product of vectors for numerator
        dot = np.dot(u, v)
        
        #magnitude of vectors for denominator
        magnitudes = u_magnitude * v_magnitude
        
        similarity = dot / magnitudes
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        # Initialize list to store recommendations
        recommended_movies = []

        # Iterate over user's ratings
        for movie_idx in range(user_ratings.shape[0]):
            user_rating = user_ratings[movie_idx]

            # Check if the user hasn't rated the movie
            if user_rating == 0:
                user_movie = ratings_matrix[movie_idx]
                total_similarity = 0

                # Calculate total similarity with other movies rated by the user
                for other_idx in range(ratings_matrix.shape[0]):
                    other_movie = ratings_matrix[other_idx]
                    our_rating = user_ratings[other_idx]

                    # If the other movie is rated and not the same as the current one
                    if our_rating != 0 and movie_idx != other_idx:
                        total_similarity += self.similarity(user_movie, other_movie) * our_rating

                # Store total similarity and movie index
                recommended_movies.append((total_similarity, movie_idx))

        # Sort recommended movies by total similarity
        recommended_movies = sorted(recommended_movies, key=lambda tup: tup[0], reverse=True)

<<<<<<< HEAD
        # Extract k movie indices from sorted recommendations
        final_recommendations = [tup[1] for tup in recommended_movies[:k]]
        
        return final_recommendations

=======
        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################
        recommendations = []

        # todo: find just the movies of interest

        for i in range(user_ratings.shape[0]):
            user_rating = user_ratings[i]
            if user_rating == 0:
                user_movie = ratings_matrix[i]
                total = 0
                for j in range(ratings_matrix.shape[0]):
                    other_movie = ratings_matrix[j]
                    our_rating = user_ratings[j]
                    if our_rating != 0 and i != j:
                        total += self.similarity(user_movie, other_movie) * our_rating
                recommendations.append((total, i))

        recommendations = sorted(recommendations, key=lambda tup: tup[0], reverse=True)
        final_recs = []
        for i in range(k):
            tup = recommendations[i]
            final_recs.append(tup[1])

        return final_recs




        #for i in range(user_ratings.shape[0]):
        #   user_rating = user_ratings[i]
         #   if user_rating == 0:
         #       user_movie = ratings_matrix[i]
          #      total = 0
          #      for j in range(ratings_matrix.shape[0]):
          #          other_movie = ratings_matrix[j]
          #          our_rating = user_ratings[j]
          ##          if our_rating != 0 and i != j:
          #              total += self.similarity(user_movie, other_movie) * our_rating
          #      recommendations.append((total, i))

        #recommendations = sorted(recommendations, key=lambda tup: tup[0], reverse=True)
        #final_recs = []
        ##for i in range(k):
         #   tup = recommendations[i]
        #    final_recs.append(tup[1])
        #return final_recs
>>>>>>> 4c376f1c7f77ef206d938be05bdb4273e366fbea

        # what we need to do:
        # first look at the user's movie ratings
        # create 2 arrays, one of the movies the user has seen, the other of the movies they haven't used
        # after those arrays are created
        # loop thru unseen movies
        # get the index of corresponding row of the unseen movie (its index) in the ratings matrix
        # find its similarity score with every seen movie in the ratings matrix
        # will have two nested for loops






        # how to create two arrays of movies the user has seen not seen
    

                
                

        # find cosine similarity between each unseen movie and seen movies and create a sum (sum += self.similarity(unseen_movie, seen_movie) * seen_movie_rating
        # after done comparing one unseen movie to everything reset the sum
        # store sum + movie index so we can give the top k scores

        
        

        ########################################################################
        #                        END OF YOUR CODE                              #
        """<<<<<<< HEAD
        ########################################################################
        =======
        #######################################################################
        >>>>>>> 7b25575aa732606c5d19b19de7aedae73e6fd9a6"""

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
