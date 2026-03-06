#! /usr/bin/python3

import sys, os
import re
from xml.dom.minidom import parse
import spacy

import paths
from dictionaries import Dictionaries

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stopwords_eng = stopwords.words('english')


CHEM_PREFIXES = {'nor','des','dex','iso','neo','oxo','oxy',
                 'pro','sul','tri','ben','eth','meth','prop'}

CHEM_SUFFIXES = {'ine','ide','ate','ase','ium','one','ene',
                 'ole','ane','cin','zol','pam','lol','pril',
                 'mab','tin','vir','xan','fen','zan'}

DRUGN_SUFFIXES = {'atin', 'idin', 'osin', 'asin', 'itol'}

GROUP_SUFFIXES = {'ones', 'oids', 'ines', 'ants', 'tics', 'ives', 
                  'ants', 'ents', 'ases', 'oles', 'anes'}


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML
def get_label(tks, tke, spans) :
    for (spanS,spanE,spanT) in spans :
        if tks==spanS and tke<=spanE+1 : return "B-"+spanT
        elif tks>spanS and tke<=spanE+1 : return "I-"+spanT
    return "O"

# t: token
# tokenFeatures: feature list
# i: current token position
# dist: distance to current token. Is Prev if negative, Next if positive, current if 0

def features_by_pos(tokens, tokenFeatures, i, dist, dicts):
   # Calculate the target index
   idx = i + dist

   if idx < 0 or idx >= len(tokens):
     return 

   # Get the token text
   tk_text = tokens[idx].text

   if dist > 0:
     suffix = f"Next{dist}"
   elif dist < 0:
     suffix = f"Prev{abs(dist)}"
   else:
     suffix = "" # Current token gets no suffix

   tokenFeatures.append(f"form{suffix}={tk_text}")
   #tokenFeatures.append(f"formlower{suffix}={tk_text.lower()}")
   tokenFeatures.append(f"suf2{suffix}={tk_text[-2:]}")
   tokenFeatures.append(f"suf3{suffix}={tk_text[-3:]}")
   tokenFeatures.append(f"suf4{suffix}={tk_text[-4:]}")
   tokenFeatures.append(f"suf5{suffix}={tk_text[-5:]}")
   #tokenFeatures.append(f"suf6{suffix}={tk_text[-6:]}")

   tokenFeatures.append(f"pref2{suffix}={tk_text[:2]}")
   tokenFeatures.append(f"pref3{suffix}={tk_text[:3]}")
   #tokenFeatures.append(f"pref4{suffix}={tk_text[:4]}")

   tokenFeatures.append(f"POS{suffix}={tokens[idx].pos_}")
   tokenFeatures.append(f"shape{suffix}={tokens[idx].shape}")
   #tokenFeatures.append(f"lemma{suffix}={tokens[idx].lemma_}")

   if tk_text.isupper(): tokenFeatures.append(f"isUpper{suffix}")
   if tk_text.istitle(): tokenFeatures.append(f"isTitle{suffix}")

   tokenFeatures.append(f"dep{suffix}={tokens[idx].dep_}")
   tokenFeatures.append(f"headForm{suffix}={tokens[idx].head.text.lower()}")

   #if tk_text in stopwords_eng: tokenFeatures.append(f"isStopWord{suffix}")

   if '-' in tk_text: tokenFeatures.append(f"hasDash{suffix}")

   found, val = dicts.find(tk_text.lower(), 'external')
   if found:
     for c in val: tokenFeatures.append(f"external{suffix}={c}")
     
   found, val = dicts.find(tk_text.lower(), 'externalpart')
   if found:
     for c in val: tokenFeatures.append(f"externalpart{suffix}={c}")


## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence
def extract_sentence_features(tokens, dicts):
   sentenceFeatures = {}
    
   for i, tk in enumerate(tokens):
      tokenFeatures = []

      # Loop through a window of -2 to +2
      for j in range(-2, 3):
         # Pass tokens, feature list, index, distance, and dicts
         features_by_pos(tokens, tokenFeatures, i, j, dicts)

      # Add Beginning/End of Sentence markers
      if i == 0:
         tokenFeatures.append("BoS")
      elif i == len(tokens) - 1: # Fixed this to dynamically check the last token
         tokenFeatures.append("EoS")

      if i > 0 and tokens[i-1].text == '(':
         tokenFeatures.append("afterOpenParen")
      if i < len(tokens)-1 and tokens[i+1].text == ')':
         tokenFeatures.append("beforeCloseParen")

      if tk.text[:3].lower() in CHEM_PREFIXES:
         tokenFeatures.append(f"chemPrefix")
      if tk.text[-3:].lower() in CHEM_SUFFIXES:
         tokenFeatures.append(f"chemSuffix")

      if tk.text[-4:].lower() in DRUGN_SUFFIXES:
         tokenFeatures.append("drugNSuffix")

      if tk.text[-4:].lower() in GROUP_SUFFIXES:
         tokenFeatures.append("groupSuffix")


      if i > 0 and tk.text.istitle() and tokens[i-1].text.istitle():
         tokenFeatures.append("consecutiveTitleCase")
      if i < len(tokens)-1 and tk.text.istitle() and tokens[i+1].text.istitle():
         tokenFeatures.append("nextIsTitleCase")

      if tk.text.isdigit(): tokenFeatures.append(f"isDigit")
      if re.search('[0-9]', tk.text): tokenFeatures.append(f"hasDigit")

      sentenceFeatures[i] = tokenFeatures
        
   return sentenceFeatures

## --------- Feature extractor ----------- 
## -- Extract features for each token in each
## -- sentence in each file of given dir

def extract_features(datafile, outfile) :

    # load dictionaries
    dicts = Dictionaries(os.path.join(paths.RESOURCES,"dictionaries.json"))

    # open output file
    outf = open(outfile, "w")
    
    # create analyzer. We don't need the parser now, it will be faster if disabled
    nlp = spacy.load("en_core_web_trf")
    
    # parse XML file, obtaining a DOM tree
    tree = parse(datafile)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      print(f"extracting sentence {sid}        \r", end="")
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity") # get gold standard entities
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))

      # convert the sentence to a list of tokens
      tokens = nlp(stext)
      # extract sentence features
      features = extract_sentence_features(tokens, dicts)

      # print features in format expected by CRF/SVM/MEM trainers
      for i,tk in enumerate(tokens) :
         # see if the token is part of an entity
         tks,tke = tk.idx, tk.idx+len(tk.text)
         # get gold standard tag for this token
         tag = get_label(tks, tke, spans)
         # print feature vector for this token
         print (sid, tk.text, tks, tke-1, tag, "\t".join(features[i]), sep='\t', file=outf)

      # blank line to separate sentences
      print(file=outf)

    # close output file
    outf.close()

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir outfile
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- corresponding feature vectors to outfile
## --

if __name__ == "__main__" :
    # directory with files to process
    datafile = sys.argv[1]
    # file where to store results
    featfile = sys.argv[2]
    
    extract_features(datafile, featfile)

