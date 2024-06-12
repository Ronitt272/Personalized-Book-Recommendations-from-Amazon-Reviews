# Personalized-Book-Recommendations-from-Amazon-Reviews

This project was developed as part of a team for the COMS-4995: Applied Machine Learning course at Columbia University.

# Dataset

# Introduction

In today's digital shopping landscape, platforms like Amazon offer vast book selections, overwhelming readers seeking personalized recommendations. Our project aims to address this challenge by leveraging the Amazon Books Reviews dataset to develop a recommendation system. By analyzing user feedback and book attributes, we seek to provide tailored book suggestions, enhancing readers' literary journey.  “Amazon Books Reviews” combines Amazon Review and Google Books API data, comprising millions of user reviews on over 200,000 books. It includes user-specific details and book attributes, forming a rich resource for analysis.

# Exploratory Data Analysis

These are some of the important insights that we derived from the exploratory data analysis.

<img>

Figure 1: Fiction has the greatest count of masterpieces                
Figure 2: Most of the texts are positive and only a few of the texts are neutral or negative.

# NLP - Sentiment Analysis

As part of the project, we have performed Sentiment Analysis using the columns ‘review/text’ (the textual data corresponding to reviews), and ‘Sentiment’ (label indicating sentiment of review as ‘Negative’, ‘Neutral’, and Positive). The various techniques employed are as follows:

1. lstm
