"""
Helper functions for Metis Project#4 Fletcher
"""

import requests
import json
from bs4 import BeautifulSoup
from textblob import TextBlob
import pandas as pd
import nltk


def get_country_ids () : 

    """
    Makes a request to the Constitute Project website
    and gets list of ids to use in their url format to
    access individual constitution web pages

    Returns:  List of constitution IDs
              Dictionary wiuth keys as country id and values are tuples of year_enacted, year_updated
    """

    url = 'https://www.constituteproject.org/service/constitutions?lang=en'
    response  = requests.get(url)


    json_data = json.loads(response.text)

    country_id_list = [json['id'] for json in json_data]

    year_enacted_list = [json['year_enacted'] for json in json_data]

    year_updated_list = [json['year_updated'] for json in json_data]

    enacted_updated_dict = {key:value for key,value in zip(country_id_list,zip(year_enacted_list, year_updated_list))}

    return country_id_list, enacted_updated_dict


def create_constitutions_dict (country_id_list):
    """
    Scrape Constitute Project website for each constitution
    Use Beautiful text to get the text out of the html

    Args:
    country_id_list from get_country_ids function 

    Returns:
    Dictionary with keys of Constitutue Project Constitutions IDs
    and values of the particulary constitutions text
    """

    country_const_dict = {}

    for country_id in country_id_list:
    
        url = 'https://www.constituteproject.org/constitution/%s?lang=en' %\
                                                                (country_id)
        
        response  = requests.get(url)
        page = response.text
        soup = BeautifulSoup(page, "lxml")
        country_const_dict[country_id] = soup.get_text()

    return country_const_dict


def clean_const_dict_values (constitutional_text):
    
    """
    Takes a value from the constitutions dict (a full constitutional text) and 
    returns a list of string clauses from that constitution


    Returns the text as a string list of clauses
    """
    
    #Create list of clauses based on new linee characters from split lines, method
    #then ignore lines that are simply spaces
    
    constitutional_text_list = [segment for segment in constitutional_text.splitlines() if 
                (segment != '' and segment != '  ' and segment != ' ')] 
    
    
    #Based on examining the US Constitution, can use these subsets exclude header and endings of html text
    constitutional_text_list = constitutional_text_list[0:constitutional_text_list.index('About Constitute\xa0\xa0')]
    constitutional_text_list = constitutional_text_list[constitutional_text_list.index('Try a new topic or search term.') + 2 : ]
    
    #Only want to keep clauses that have 5 words or more, this eliminates heades and such like Amendment XVI
    constitutional_text_list = [segment for segment in constitutional_text_list if len(segment.split(" ")) >= 5]
    
    return constitutional_text_list


def make_const_clauses_dataframe (const_clean_dict, enacted_updated_dict):
    
    """
    Take a constitutional dictionary of keys with name of country and values
    list of string clauses of that constitution and return a dataframe with each row
    being an individual clause.  Additionally, add the TextBlob Polarity and Subjectivity
    of each Clause to the datafame.  Also from enacted_updated_dict from the get_country_ids function
    takes the tuple values to also include the year_enacted and the year amended, if any, in the dataframe
    """

    constitution_list = []
    clause_list = []
    stemmed_clause_list = []
    polarity_list = []
    subjectivity_list = []
    year_enacted_list = []
    year_amended_list = []
    stemmer = nltk.stem.porter.PorterStemmer()

    for key in const_clean_dict.keys():
        for clause in const_clean_dict[key]:
            constitution_list.append(key)
            clause_list.append(clause)
            sentiment = TextBlob(clause).sentiment
            polarity_list.append(sentiment[0])
            subjectivity_list.append(sentiment[1])
            
            stemmed_list = [stemmer.stem(word) for word in TextBlob(clause).words]
            stemmed_clause_list.append(" ".join(stemmed_list))

            year_enacted_list.append(enacted_updated_dict[key][0])
            year_amended_list.append(enacted_updated_dict[key][1])



    df_cons = pd.DataFrame({"Country":  constitution_list,
                            "Clause":  clause_list, 
                            "Stemmed_Clause": stemmed_clause_list,
                             "Polarity": polarity_list, 
                             "Subjectivity": subjectivity_list,
                             "Year_Enacted": year_enacted_list,
                              "Year_Amended": year_amended_list})


    return df_cons


def get_lda_top_components (lda_object, feature_names, top_words = 10):
    """
    Prints out the components from LDA analysis with specified number of words
    for each topuc
    """ 
    

    for index, topic in enumerate (lda_object.components_):
        print ("Topic %d:" % (index +1))
        #Need to access from end as most important is last
        
        #argsort returns the indices which would sort an array, so getting the last
        #indices in argsort gives the highest valued components, then can get the feature
        #names from the feature list which 
        print (" ".join([feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]]))
        
    return None



def similar_clause (df_cons, doc_topic_matrix, clause_index):
    """
    Pass the data frame and a document topic matrix, find
    the clause from another constitution that has the best
    cosine similarity based on the document_topic vector
    """
    
    #get cosine similarity for clause and all other documents
    #python list, technically 2D numpy array
    #make a copy of list which country clauses will be removed from
    
    cosine_list = cosine_similarity(doc_topic_matrix[clause_index].reshape(1,-1), 
                                    nmf_topics)
    
    cosine_list = list(cosine_list[0,])
    
    copy_cosine_list = cosine_list
    
    
    #Get list of indices of country_constitution index
    #Want to get most similar clause excluding this Country
    country_index_list = df_cons.index[df_cons['Country'] == df_cons.loc[clause_index,
                                                                         "Country"]].tolist()
    
    #Remove those values from the cosines
    del copy_cosine_list[country_index_list[0]:country_index_list[-1] + 1]
    
    max_similarity = max(copy_cosine_list)
    
    similar_index = cosine_list.index(max_similarity)
    
    print (df_cons.loc[clause_index,"Country"])
    print (df_cons.loc[clause_index,"Clause"])
    print (doc_topic_matrix[clause_index])
    print ("\n")
    print ("Most Similar Clause from Another Country")
    print (df_cons.loc[similar_index,"Country"])
    print (df_cons.loc[similar_index,"Clause"])
    print (doc_topic_matrix[similar_index])
    
    return df_cons.loc[similar_index]