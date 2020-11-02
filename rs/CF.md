# Collaborative Filtering Systems

Collaborative filtering approach builds a model from a user’s past behaviors (items previously purchased or selected 
and/or numerical ratings given to those items) as well as similar decisions made by other users. It uses the past behavior relations between
users and items to predict items (or ratings for items) that the user may have an interest in.

CF can also be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering.

![](assets/rs/1.png)

## Memory-Based Collaborative Filtering

Approaches can be divided into two main sections: user-item filtering and item-item filtering. 

A user-item filtering takes a particular user, find users that are similar to that user based on similarity of ratings, and recommend items that those similar users liked.

In contrast, item-item filtering will take an item, find users who liked that item, and find other items that those users or similar users also liked. 
It takes items and outputs other items as recommendations.

> Item-Item Collaborative Filtering
* “Users who liked this item also liked …”

> User-Item Collaborative Filtering
* “Users who are similar to you also liked …”

Item based approach is usually preferred over user-based approach. 

User-based approach is often harder to scale because of the dynamic nature of users, whereas items usually don’t change much, 
and item based approach often can be computed offline and served without constantly re-training.

User or items similarities are calculated by using distance metrics/process like : Cosine similarity or Pearson correlation coefficients, 
which are only based on arithmetic operations. Another techniques where we don’t use parametric machine learning approach are classified as Memory based techniques. 
Therefore, non parametric ML approaches like KNN should also come under Memory based approach. 

### User Based Systems

### Item-Item Collaborative Filtering

Item-to-item collaborative filtering system is based on how users consider **products as similars**. 

**Item-to-item similarity** is based on how people treat these things in terms of likes and dislikes. We can say that 2 
products are similar if users treat them "equally" in terms of responses (likes or dislikes / sales).


#### What do we try to answer with Item-Item System?
These are the types of questions we want to address with an item-item system:
* Do people who like one of them also like the other ?
* Do people who purchase one of them tend to purchase the other ?

Item-to-item is a recommender system that was designed to overcome the major limitations of user-based approach: 
**sparcity and computational performance**.

> Sparcity
* In big companies, we usually face a limited set of products with a larger and increasing set of clients.
* The number of products that any given customer would have rated/bought was often too small
* This might create a large sparse matrix with a lot of empty spaces  (user x item grid)
* To address this problem, several approaches can be tried like: item-item CF, "filterbots" and dimensionality reduction, for example

> Computational Performance
* Computational performance for traditional user-user CF is not good
    * With millions of users, computing all pairs correlations is expensive
    * It involves m ** n and m x n dimensions
* Other problem is that these relationships often need to be computed frequently
    * User profiles change quickly and models need to give almost real-time solutions

Item-item CF approach tries to address these problems, providing an alternative to user-based systems.

**Item-item characteristics**

> Stable
* item-item similarity is fairly stable
* average product behaviors tend to be more stable then the average user
* Products responses (likes, ratings, sales) are obtained through several interactions of different users, so a single rating or sale 
won't affect that much the product average results

> Good Performance 
- In general, item-item methods present a good performance compared to user-based systems

**Note**
* In general, item-item is a pretty good algorithm with better performance and stability than user-user. 
However, In an application where we have a relatively small number of users and many products (fixed base of customer 50k and
millions of products to sell), it's likely to fail to outperform the user-user approach. 

#### Item-Item Recommendation Approach
 
* Compute similarity between pairs of products
    * Correlation between rating vectors
        * Useful if we have scales of ratings (1 to 5, for example)
        * On items that are co-rated (same user rated 2 items)
            * We'll be looking at many users who rated both items to see if they trend to treat them similarly or differently.
    * Cosine of item rating vectors
        * Angle between points in this multi-dimensional space
        * can be used with multi-level ratings or unary ratings/events
        * Rates can be normalized before computing cosine
    * Some conditional probability
        * For unary data
* Predict user-item rating 
    * weighted sum of rated "item neighbors"
    * Linear regression to estimate rating

The output can be also treated as a top-n format:
* Simplify model by limiting items to small "neighboors" of k-most similar items
* For a given item, let's output the k-most similar items

#### Pros & Cons item-item

> Pros
* Works quite well
    * good prediction accuracy
    * good performance on top-n predictions
* Efficient Implementation
    * number of users >> number of items
* Flexible 

> Assumptions and Limitations
* Item-item relationships need to be stable
    * Related to user preferences stability
* Depending on business and data, we can find seasonality problems
    * Christmas trees, calendars (beginning of year), etc...
    * Example: a product correlates well with products for 1 month or 2 and then nobody wants this product anymore
* Popularity Bias
    * recommender is prone to recommender popular items
    * Sometimes there's no real surprised in recommendations that you might expect from a user-user system
* item cold-start problem
    * recommender fails to recommend new or less-known items because items have either none or very little interactions
    * New products are not used to train models, so recommendations might be not updated as they should be 

#### Item-Item Recommendation Implementation

The basic structure of the item-item algorithm is to:
* Pre-compute item similarities over all pairs of items
    * Here we can use a neighborhood selection strategy
* Look for items similar to those the user likes (or has purchased, or has in their basket)
    * Create an item score aggregation function

To calculate similarity between two items, we looks into the set of items the target user has rated and computes how 
similar they are to the target item i and then selects k most similar items. 

Similarity between two items is calculated by taking the ratings of the users who have rated both the items and 
thereafter using the cosine similarity function.

![](assets/rs/2.png)

Once we have the similarity between the items, the prediction is then computed by taking a weighted average of the target 
user’s ratings on these similar items. The formula to calculate rating is very similar to the user based collaborative 
filtering except the weights are between items instead of between users.

![](assets/rs/3.png)

* The score is driven by item
* For each item to score:
    * Find similar items the user has rated/purchased
    * Compute weighted average of user's ratings 
    * Average normalized ratings, denormalize when done
    * Can consider other methods like linear regression

#### Item-Item to Unary Data

Implicity feedback is all binary data:
* clicks
* feedbacks
* purchases

> Data 
* Data is represented by a binary matrix: 1 purchase 0 no purchase

> Normalization
* Normalize vectors to unit vectors
    * Users who like many items provide less information about any particular item
    * Put less weight on users that buy everything
    * If we have large counts, we can use a log function to decrease this impact

> Distance
* Cosine similarity still work 

> Aggregating Scores
* For binary data, we can just sum neighbors similarities
* Fixed neighborhood size means this isn't unbounded
* We can bound this by top N similar items

#### Example: Item-Item Binary Data

Data is a subset of the last.fm dataset. 
It’s a sparse matrix containing 285 artists and 1226 users and contains what users have listened to what artists.

First we need to start computing the item-item relationship matrix. 
* Understand how items are related to each other based on historical purschasing data
* Objective here is to obtain a matrix with item-pair weights 
    * close to 1 --> strong relation
    * close to 0 --> weak relation
* We will be using the cosine similarity metric to detect such relationships

![](assets/rs/4.png)

````python
import pandas as pd
import numpy as np
import matplotlib as pyplot
%matplotlib inline

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

data = pd.read_csv("lastfm1.csv")
data.head()
````

![](assets/rs/5.png)

Now, let's exclude the user id so we can work with a pure item-item raw matrix. Then, we'll normalize and create the
similarity matrix.

````python
#------------------------
# ITEM-ITEM CALCULATIONS
#------------------------

# As a first step we normalize the user vectors to unit vectors.

# magnitude = sqrt(x2 + y2 + z2 + ...)
magnitude = np.sqrt(np.square(data_items).sum(axis=1))

# unitvector = (x / magnitude, y / magnitude, z / magnitude, ...)
data_items = data_items.divide(magnitude, axis='index')

def calculate_similarity(data_items):
    """Calculate the column-wise cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
    return sim

# Build the similarity matrix
data_matrix = calculate_similarity(data_items)
data_matrix.head()
````
![](assets/rs/6.png)


> Item-item summary
* Normalize user vectors to unit vectors.
* Construct a new item by item matrix.
* Compute the cosine similarity between all items in the matrix.
* Obtain item level predictions (scores)
    * It can be interesting to select K top neighboors
    * Define a neighborhood of items.
    * Calculate the score for all items for a specific user.
    * Sort by the n highest scores (most recommended)




 
 


 
