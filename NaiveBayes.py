# implementation of naive bayes -> multinomial 

# naive bayes - text classification algo 
# feature X = [] (indicator function based values - 1 or 0) OR X = I{words appearing in mail}

# joint likelihood = Pi over "probability" of x_i, y_i, phi_y and phi_i_over_y from i to m
# where phi_y = Sigma over I(y_i=1) / m from i to m
# where phi_i_over_y = {Sigma over I{x_i = 1, y_i=1} / {Sigma over I{y_i=1}}}


# we know, 
# p(y=1/x) = p(x/y=1).p(y=1) / (total cases with y = 0 and y = 1)


# using laplace smooothing = count + 1 / N + 10000

# multivariate bernoulli -> whether the word is present in the class list or not.
# uses same formulae and conditional probability

# multinomial event model ->  determines the index of where the word is present.
# phi_k_over_y = Sigma of I(y_i=0) . Sigma ofI(x_i = k) / sigma of I(y_i = 0).n_i

# focusing over multinomaial classification

import pandas as py
import numpy as np
import math

SPAM_TRAIN = [
    "Congratulations you have won a lottery claim your prize now",
    "Act now limited time offer get free cash immediately",
    "You are selected for a free gift card click here to claim",
    "Urgent your account has been suspended verify now",
    "Earn money fast with zero risk investment opportunity",
    "Buy now special promotion unbeatable price satisfaction guaranteed",
    "Click here to claim your jackpot winner prize money",
    "Free membership no cost no hidden fees sign up now",
    "Wire transfer required to release your unclaimed funds",
    "Exclusive deal free gift card expires today last chance",
    "Miracle cure weight loss viagra cheap pharmaceuticals online",
    "Casino poker online betting double your income fast cash",
    "Update password immediately security alert confirm identity bank details",
    "Once in a lifetime opportunity get rich quick financial freedom",
    "Order now risk-free trial lowest price lowest rate guaranteed",
]

HAM_TRAIN = [
    "Hey are you coming to the meeting tomorrow at noon",
    "Please review the project report and share your feedback",
    "The weather is really nice today lets go for a walk",
    "Can you pick up groceries on your way back home",
    "Happy birthday hope you have a wonderful day with family",
    "Meeting rescheduled to 3 pm please update your calendar",
    "Thanks for helping me with the assignment really appreciate it",
    "Lets grab lunch together at the new restaurant downtown",
    "The movie was great we should watch it again sometime",
    "Can you send me the notes from today's lecture please",
    "I will be late tonight dont wait for me for dinner",
    "Good morning have a productive day at work today",
    "Did you finish the homework it was really difficult",
    "Lets plan a trip this weekend to the mountains",
    "The concert was amazing I had so much fun last night",
]

class NaiveBayes():
    
    def __init__(self, alfa=0.1):
        self.alfa=alfa
        self.phi_y = None
        self.phi_i_over_y = None
        self.spam_likelihood = {}
        self.ham_likelihood = {}
        
    def class_prior(self, SPAM_TRAIN : list, HAM_TRAIN : list):            
        
        n_spam, n_ham = len(SPAM_TRAIN), len(HAM_TRAIN)
        return n_spam / (n_spam + n_ham)
    
    def likelihood(self, SPAM_TRAIN : list, HAM_TRAIN : list):
        
        spam_words = []
        ham_words = []
        # spam_words = [spam_words.extend(i.lower().split()) for i in SPAM_TRAIN]
        # ham_words = [ham_words.extend(i.lower().split()) for i in HAM_TRAIN]
        
        for i in SPAM_TRAIN:
            spam_words.extend(i.lower().split())
        for i in HAM_TRAIN:
            ham_words.extend(i.lower().split())
        
        vocabulary = set(spam_words + ham_words)
        n = len(vocabulary)
        
        n_spam = len(spam_words)
        n_ham = len(ham_words)
        
        # spam_count, ham_count = (spam_words.count(word) for word in vocabulary, ham_words.count(word) for word in vocabulary)
        
        for word in vocabulary:
            spam_count = spam_words.count(word)
            ham_count = ham_words.count(word)
            
            # with laplace smoothing
            self.spam_likelihood[word] = (spam_count + self.alfa) / (n_spam + self.alfa  * n)
            self.ham_likelihood[word] = (ham_count + self.alfa) / (n_ham + self.alfa  * n)
        
        return self.spam_likelihood, self.ham_likelihood
    
    def fit(self, SPAM_TRAIN : list, HAM_TRAIN : list):
        
        self.phi_y = self.class_prior(SPAM_TRAIN, HAM_TRAIN)
        self.spam_likelihood, self.ham_likelihood = self.likelihood(SPAM_TRAIN, HAM_TRAIN)
    
    def predict(self, text):  
        
        text = text.lower().split()
        spam_score = math.log(self.phi_y)
        ham_score = math.log(1 - self.phi_y)
        
        # spam_word = self.spam_likelihood.get()  
        for i in text:
            spam_word = self.spam_likelihood.get(i)
            ham_word = self.ham_likelihood.get(i)
            
            if spam_word is None or ham_word is None:
                continue
            spam_score += math.log(spam_word)
            ham_score += math.log(ham_word)
            
        return "spam" if spam_score > ham_score else "ham"
        
if __name__ == "__main__":
    model = NaiveBayes()
    model.fit(SPAM_TRAIN, HAM_TRAIN)

    correct = 0
    total = 0
    for sentence in SPAM_TRAIN:
        total = total + 1
        if model.predict(sentence) == "spam":
            correct = correct + 1
    for sentence in HAM_TRAIN:
        total = total + 1
        if model.predict(sentence) == "ham":
            correct = correct + 1

    accuracy = correct / total
    print("Training accuracy: " + str(accuracy))

    while True:
        text = input("\nEnter a message (or 'quit' to exit): ")
        if text.lower() == "quit":
            break
        result = model.predict(text)
        print(result)