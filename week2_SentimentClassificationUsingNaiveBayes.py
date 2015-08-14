import re
import os
import math
import elice_utils

def main():
    # Implement main function for Emotion Classifier
    training1_sentences = read_text_data('./txt_sentoken/pos/')
    training2_sentences = read_text_data('./txt_sentoken/neg/')
    testing_sentence = input()

    alpha = 0.1
    prob1 = 0.5
    prob2 = 0.5

    prob_pair = naive_bayes(training1_sentences, training2_sentences, testing_sentence, alpha, prob1, prob2)

    plot_title = testing_sentence
    if len(plot_title) > 50: plot_title = plot_title[:50] + "..."
    print(elice_utils.visualize_boxplot(plot_title,
                                        list(prob_pair),
                                        ['Positive', 'Negative']))

def naive_bayes(training1_sentence, training2_sentence, testing_sentence, alpha, prob1, prob2):
    # Exercise

    # naive bayes를 통해 어떤 testing_sentence가 training1_sentence에서 나올 확률과,
    # training2_model에서 나올 확률의 상대값을 비교한다.
    
    training1_model = create_BOW(training1_sentence)
    training2_model = create_BOW(training2_sentence)
    testing_model = create_BOW(testing_sentence)
    
    training1_prob = calculate_doc_prob(training1_model, testing_model, alpha)
    training2_prob = calculate_doc_prob(training2_model, testing_model, alpha)
    
    classify1 = training1_prob + math.log(prob1)
    classify2 = training2_prob + math.log(prob2)
    
    return normalize_log_prob(classify1, classify2)

def read_text_data(directory):
    # We already implemented this function for you
    files = os.listdir(directory)
    files = [f for f in files if f.endswith('.txt')]

    all_text = ''
    for f in files:
        all_text += ' '.join(open(directory + f).readlines()) + ' '
        
    return all_text

def normalize_log_prob(prob1, prob2):
    
    # 두 확률의 총 합을 1로 만들어 정규화하는 과정.
    
    maxprob = max(prob1, prob2)

    prob1 -= maxprob
    prob2 -= maxprob
    prob1 = math.exp(prob1)
    prob2 = math.exp(prob2)

    normalize_constant = 1.0 / float(prob1 + prob2)
    prob1 *= normalize_constant
    prob2 *= normalize_constant

    return (prob1, prob2)

def calculate_doc_prob(training_model, testing_model, alpha):
    # Implement likelihood function here...
    
    # 첫번째 문장인 training_model에서 쓰인 단어 사용 횟수를 기반으로
    # 두 번째 testing_model의 문장이 만들어지는 확률을 계산한다.
    # training_model에 등장하지 않은 단어라고 해서 testing_model에 사용되지 말라는 법이 없으므로,
    # laplace smoothing을 거쳐 training_model에 없는 단어의 등장 확률을 0으로 만들지 않는 것이 중요하다.
    # 각 확률의 log값을 반환한다.
    
    #print(training_model)
    #print(testing_model)
    
    N = sum(training_model[1])
    d = len(training_model[0])
    logprob = 0.0
        
    for word in testing_model[0] :
        prob = 0.0
        if word in training_model[0] :
            prob += math.log(training_model[1][training_model[0][word]] + alpha)
            
        else :
            prob += math.log(alpha)
            
        prob -= math.log(N + d*alpha)
        logprob += prob * testing_model[1][testing_model[0][word]]

    return logprob

def create_BOW(sentence):
    # 어떤 문장이 각 단어가 몇 번 사용되었는지 조사한다.
    bow_dict = {} # 문장에 등장하는 단어에 idx값을 부여한 dictionary.
    bow = [] # bow_dict의 idx에 해당하는 단어가 등장한 횟수를 조사한 list.

    sentence = sentence.lower()
    sentence = replace_non_alphabetic_chars_to_space(sentence)
    words = sentence.split(' ')
    for token in words:
        if len(token) < 1: continue
        if token not in bow_dict:
            new_idx = len(bow)
            bow.append(0)
            bow_dict[token] = new_idx
        bow[bow_dict[token]] += 1

    return bow_dict, bow

def replace_non_alphabetic_chars_to_space(sentence):
    return re.sub(r'[^a-z]+', ' ', sentence)

if __name__ == "__main__":
    main()