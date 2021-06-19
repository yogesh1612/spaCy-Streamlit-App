import streamlit as st 
# NLP Pkgs
import spacy_streamlit
import spacy, time
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print('Downloading language model for the spaCy POS tagger\n'
        "(don't worry, this will only happen once)", file=stderr)
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en')
#nlp = spacy.load('en_core_web_sm')
#import os
#from PIL import Image
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pandas as pd
from nltk import Tree
import base64  # for csv downloads

def sent_tokenize1(raw_text):
    #nlp = en_core_web_sm.load()
    sents_list = sent_tokenize(raw_text)
    #sents_list = list(doc.sents); #sents_list
    index0 = [i+1 for i in range(len(sents_list))]; #index0
    sents_pd = pd.DataFrame({'sl_num':index0, 'sentence':sents_list})    
    return(sents_pd)

def token_attrib(sent0): # routine to display sentence as DF postags
	doc = nlp(sent0)

	text=[]
	lemma=[]
	postag=[]
	depcy=[]
	entity=[]

	for token in doc:
		text.append(token.text)
		lemma.append(token.lemma_)
		postag.append(token.pos_)
		depcy.append(token.dep_)
		entity.append(token.ent_type_)

	test_df = pd.DataFrame({'text':text, 'lemma':lemma, 'postag':postag, 'depcy':depcy, 'entity': entity})
	return(test_df)

def sent_attribs(sents_pd):
    tok_df0 = pd.DataFrame(columns = ["doc_index", "sent_index", "text", "lemma", "postag", "depcy", "entity"])
    for i0 in range(sents_pd.shape[0]):
        tok_df = token_attrib(str(sents_pd.sentence.iloc[i0])); #tok_df
        sent_index1 = [sents_pd.sl_num.iloc[i0]]*tok_df.shape[0]
        doc_index1 = [sents_pd.doc_index.iloc[i0]]*tok_df.shape[0]
        tok_df.insert(0, "sent_index", sent_index1)
        tok_df.insert(0, "doc_index", doc_index1)
        tok_df0 = pd.concat([tok_df0, tok_df])
    return tok_df0

def corpus2df(corpus0):
    df0 = pd.DataFrame(columns = ["doc_index", "sent_index", "text", "lemma", "postag", "depcy", "entity"])
    for i0 in range(corpus0.shape[0]):        
        df1 = sent_tokenize1(corpus0.iloc[i0,0])
        doc_index0 = [i0+1]*len(df1)
        df1.insert(0, "doc_index", doc_index0)
        df2 = sent_attribs(df1)
        df0 = pd.concat([df0, df2])
    return df0

@st.cache(persist=True)
def processed_file(tok_df0):
	data1 = pd.DataFrame(tok_df0)
	return data1

# def func to display depcy tree. Note recursive struc!!
def to_nltk_tree(node):
	if node.n_lefts + node.n_rights > 0:
		return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
	else:
		return node.orth_
    
## define a func to extract & display chunking ops
def chunkAttrib(sent0):

	doc = nlp(sent0)
	chunk1 = [(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text) for chunk in doc.noun_chunks if len(word_tokenize(chunk.text)) > 1]
    
	out_df1 = pd.DataFrame(chunk1, columns = ['chText', 'chRootText', 'chRootDep', 'chRootHead'])
	return(out_df1)

# test-drive above func
#sent0 = "Donald Trump is a controversial American President."
#chunk_df = chunkAttrib(sent0)  # 0.01 secs
#print(chunk_df)

def ner_bilou_tbl(docx):  # docx is nlp(text) obj
	ent_token = [X for X in docx if X.ent_iob_ != 'O']
	ent_iob = [X.ent_iob_ for X in docx if X.ent_iob_ != 'O']
	ent_type = [X.ent_type_ for X in docx if X.ent_iob_ != 'O']
	ent_chunk = pd.DataFrame({'ent_token':ent_token, 'ent_iob':ent_iob, 'ent_type':ent_type})
	return ent_chunk

    
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download above tbl as csv</a>'
    return href

def senti_df(raw_text):
	analyzer = SentimentIntensityAnalyzer()
	sents_list = sent_tokenize(raw_text)
	senti_df1 = pd.DataFrame(columns=['sentence', 'neg', 'neu', 'pos', 'compound'])	
	for i0 in range(len(sents_list)):
		sent0 = sents_list[i0]
		vs = analyzer.polarity_scores(sent0)	
		a0 = pd.DataFrame([vs])
		a0.insert(0, "sentence", sent0)
		senti_df1 = pd.concat([senti_df1, a0])
	return senti_df1


## main app wrapper func

def main():
    
	"""A Simple NLP app with Spacy-Streamlit"""
    
	st.title("Spacy-Streamlit NLP App")
    
	menu = ["Annotate for file", "NER by Sentence", "Phrase chunks by Sentence", "Sentiment by Sentence", "Parse_tree by Sentence"]
	choice = st.sidebar.selectbox("Menu", menu)

	if choice == "Annotate for file":        
    
		postag1 = st.checkbox("Want Postag Filtering?") 
		ner1 = st.checkbox("want NER?")    

		uploaded_file = st.file_uploader("Choose a CSV file")

		if uploaded_file is not None:		            
			data = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
			st.markdown('### Data Sample')
			st.write(data)
			id_col = st.sidebar.selectbox('Select text column', options=data.columns)
          
			data0 = data[[id_col]]
			t1 = time.time()
			tok_df0 = corpus2df(data0)
			t2 = time.time()
			td = round(t2 - t1, 2)
			st.write("Proc took ", td, " secs.")
			st.write(tok_df0)
			st.write("Table size is: ", tok_df0.shape[0]," x ", tok_df0.shape[1])
			tok_df0 = processed_file(tok_df0)
			st.markdown(get_table_download_link(tok_df0), unsafe_allow_html=True)            


	            
		else:
			st.markdown("""
			<br>
			<br>
			<br>
			<br>
			<h1 style="color:#26608e;"> Upload your CSV file to begin clustering </h1>
			<h3 style="color:#f68b28;"> MLBM </h3>
			""", unsafe_allow_html=True) 
                   

		if (postag1 == True) & (uploaded_file is not None):
			tok_df0 = processed_file(tok_df0)
			postags = pd.unique(tok_df0.postag)
			postag0 = st.selectbox("Postags to filter on:", postags)
			tok_df1 = tok_df0.loc[(tok_df0.postag == postag0),]
			st.markdown('### Postags Dataframe')                 
			st.dataframe(tok_df1)
			st.markdown(get_table_download_link(tok_df1), unsafe_allow_html=True)            

		if (ner1 == True) & (uploaded_file is not None):
			tok_df0 = processed_file(tok_df0)                
			a0 = (tok_df0.entity != "")
			tok_df2 = tok_df0.loc[a0,:]
			uniq_ner = pd.unique(tok_df2.entity)          
			st.markdown('### Entities Dataframe')                 
			st.write("entities found were: ", uniq_ner)  
			st.dataframe(tok_df2)
			st.markdown(get_table_download_link(tok_df2), unsafe_allow_html=True)
    
	if choice == "NER by Sentence":        
		st.subheader("Named Entity Recognition")
		raw_text = st.text_area("Your Text","Enter Text Here")
		docx = nlp(raw_text)
		spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)
		ner_bilou = ner_bilou_tbl(docx)
		st.dataframe(ner_bilou)
		st.markdown(get_table_download_link(ner_bilou), unsafe_allow_html=True)
        
        

	if choice == "Phrase chunks by Sentence":
		st.subheader("Noun and Verb Phrases by Sentence n Text")
		raw_text = st.text_area("Your Text","Enter Text Here")
		raw_sents_df = sent_tokenize1(raw_text)
		chunk_df = pd.DataFrame(columns = ["sent_index", "chText", "chRootText", "chRootDep", "chRootHead"])
		for i0 in range(raw_sents_df.shape[0]):            
			chunk_df0 = chunkAttrib(raw_sents_df.sentence.iloc[i0])
			sent_index0 = [i0]*chunk_df0.shape[0]
			chunk_df0.insert(0, "sent_index", sent_index0)
			chunk_df = pd.concat([chunk_df, chunk_df0])

		st.markdown('### Phrase Chunking Dataframe')            
		st.dataframe(chunk_df)
		st.markdown(get_table_download_link(chunk_df), unsafe_allow_html=True)

	if choice == "Sentiment by Sentence":
		st.subheader("Sentiment by Sentence")
		raw_text = st.text_area("Your Text","Enter Text Here")
		senti_df1 = senti_df(raw_text)
		st.markdown("### Vader Sentiment scores by Sentence")
		st.dataframe(senti_df1)
		st.markdown(get_table_download_link(senti_df1), unsafe_allow_html=True)

        
	if choice == "Parse_tree by Sentence":
		st.subheader("Sentence Parse_Tree")
		raw_text = st.text_area("Your Text","Enter Text Here")
		docx = nlp(raw_text)
		a1 = [to_nltk_tree(sent.root).pretty_print() for sent in docx.sents]
		st.write(a1)
        
                   
if __name__ == '__main__':
	main()