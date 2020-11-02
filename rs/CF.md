# Collaborative Filtering Systems

Collaborative filtering approach builds a model from a user’s past behaviors (items previously purchased or selected 
and/or numerical ratings given to those items) as well as similar decisions made by other users. It uses the past behavior relations between
users and items to predict items (or ratings for items) that the user may have an interest in.

In general, they can either be user-based or item-based. Item based approach is usually preferred over user-based approach. 
User-based approach is often harder to scale because of the dynamic nature of users, whereas items usually don’t change much, 
and item based approach often can be computed offline and served without constantly re-training.

CF can also be divided into Memory-Based Collaborative Filtering and Model-Based Collaborative filtering.

![](assets/rs/1.png)

# User Based Systems

# Item Based Systems

## Item-Item Collaborative Filtering

Item-to-item collaborative filtering system is based on how users consider **products as similars**. 

**Item-to-item similarity** is based on how people treat these things in terms of likes and dislikes. We can say that 2 
products are similar if users treat them "equally" in terms of responses (likes or dislikes / sales).


### What do we try to answer with Item-Item System?
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

### Item-Item Recommendation Approach
 
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

### Pros & Cons item-item

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

### Item-Item Recommendation Implementation

The basic structure of the item-item algorithm is to:
* Pre-compute item similarities over all pairs of items
* Look for items similar to those the user likes (or has purchased, or has in their basket)

 
