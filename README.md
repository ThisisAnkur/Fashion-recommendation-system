# Fashion-recommendation-system

Nowadays, people used to buy products online more than from stores. Previously, people used to buy products based on the reviews given by relatives or friends but now as the options increased and we can buy anything digitally we need to assure people that the product is good and they will like it. To give confidence in buying the products, recommender systems were built.

![image](https://user-images.githubuser

Now, we’ll look towards different types of filtering used by recommendation engines.

# Content-based filtering
This filtering is based on the description or some data provided for that product. The system finds the similarity between products based on its context or description. The user’s previous history is taken into account to find similar products the user may like.
For example, if a user likes movies such as ‘Mission Impossible’ then we can recommend him the movies of ‘Tom Cruise’ or movies with the genre ‘Action’.


In this filtering, two types of data are used. First, the likes of the user, the user’s interest, user’s personal information such as age or, sometimes the user’s history too. This data is represented by the user vector. Second, information related to the product’s known as an item vector. The item vector contains the features of all items based on which similarity between them can be calculated.

The recommendations are calculated using cosine similarity. If ‘A’ is the user vector and ‘B’ is an item vector then cosine similarity is given by


content.com/94610135/218416365-9822999f-940e-49ca-8184-b305030da769.png)

# Collaborative filtering
The recommendations are done based on the user’s behavior. History of the user plays an important role. For example, if the user ‘A’ likes ‘Coldplay’, ‘The Linkin Park’ and ‘Britney Spears’ while the user ‘B’ likes ‘Coldplay’, ‘The Linkin Park’ and ‘Taylor Swift’ then they have similar interests. So, there is a huge probability that the user ‘A’ would like ‘Taylor Swift’ and the user ‘B’ would like ‘Britney Spears’. This is the way collaborative filtering is done.

[Uploading 1_6_jpOMKFAUylzBoR10Mopw.webp…]()

Both recommendation algorithms have their advantages and disadvantages. To make more accurate recommendations nowadays the hybrid recommendation algorithm is used; that is products are recommended using both content-based and collaborative filtering together. The hybrid recommendation algorithm is more efficient and more useful.
