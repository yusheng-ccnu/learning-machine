from nltk import word_tokenize,pos_tag
import jieba



lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}
def lookup_words(input_text):
	words = input_text.split(' ')
	result = []
	new_text = ""
	for word in words:
		if word.lower() in lookup_dict:
			word = lookup_dict[word.lower()]
		result.append(word)
		new_text = " ".join(result)
	return new_text
print(lookup_words("RT this is a retweeted tweet by Shivam Bansal"))

text="我是一个中国人,我的名字叫毛泽东"

seg_list = jieba.cut(text,cut_all=False)
seg_str = " ".join(seg_list)
print(seg_str)

eg_str = 'i am chinese man , my name is maozedong'
words = word_tokenize(seg_str)
print(words)

tags = pos_tag(words)
print(tags)
