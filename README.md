# Email Intent Classifier
## A NLP program that predicts the intent of a customer query email.

### An incoming email can have one of the folllowing 6 intents
- Sales/Promotion
- Product Specifications
- Product Availability
- Product Comparison
- Shipping
- Omnichannel

The program trains on a training dataset and then predicts the intents for the testing dataset.
CV accuracy obtained on the training dataset by splitting in ratio 80 : 20 using the CountVectorizer algorithm is obtained as 91.875 %.
