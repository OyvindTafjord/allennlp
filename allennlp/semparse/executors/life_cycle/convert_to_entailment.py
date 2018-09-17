"""
Script to convert the retrieved HITS into an entailment dataset
USAGE:
 python scripts/convert_to_entailment.py hits_file output_file

JSONL format of files
 1. hits_file:
 {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?"
      "choice": {"text": "dry palms", "label": "A"},
       "support": {
         "text": "Use hand sanitizers according to directions, which usually involves rubbing for
                  at least ten seconds, then allowing hands to air dry."
         ...
        }
    },
     "answerKey":"A"
  }

 2. output_file:
   {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?"
      "choice": {"text": "dry palms", "label": "A"},
      "support": {
         "text": "Use hand sanitizers according to directions, which usually involves rubbing for
                  at least ten seconds, then allowing hands to air dry."
         ...
        }
    },
     "answerKey":"A",
     "premise": "Use hand sanitizers according to directions, which usually involves rubbing for
                  at least ten seconds, then allowing hands to air dry.",
     "hypothesis": "George wants to warm his hands quickly by rubbing them. Dry palms skin
                    surface will produce the most heat."
  }
"""

import json
import re
import sys


# String used to indicate a blank
BLANK_STR = "___"



# Get a Fill-In-The-Blank (FITB) statement from the question text. E.g. "George wants to warm his
# hands quickly by rubbing them. Which skin surface will produce the most heat?" ->
# "George wants to warm his hands quickly by rubbing them. ___ skin surface will produce the most
# heat?
def get_fitb_from_question(question_text: str) -> str:
    fitb = replace_wh_word_with_blank(question_text)
    if not re.match(".*_+.*", fitb):
        print("Can't create hypothesis from: '{}'. Appending {} !".format(question_text, BLANK_STR))
        # Strip space, period and question mark at the end of the question and add a blank
        fitb = re.sub("[\.\? ]*$", "", question_text.strip()) + BLANK_STR
    return fitb


# Create a hypothesis statement from the the input fill-in-the-blank statement and answer choice.
def create_hypothesis(fitb: str, choice: str) -> str:
    if ". " + BLANK_STR in fitb or fitb.startswith(BLANK_STR):
        choice = choice.capitalize()
    else:
        choice = choice.lower()
    # Remove period from the answer choice, if the question doesn't end with the blank
    if not fitb.endswith(BLANK_STR):
        choice = choice.rstrip(".")
    # Some questions already have blanks indicated with 2+ underscores
    hypothesis = re.sub("__+", choice, fitb)
    return hypothesis


# Identify the wh-word in the question and replace with a blank
def replace_wh_word_with_blank(question_str: str):
    wh_word_offset_matches = []
    wh_words = ["which", "what", "where", "when", "how", "who", "why"]
    for wh in wh_words:
        # Some Turk-authored SciQ questions end with wh-word
        # E.g. The passing of traits from parents to offspring is done through what?
        m = re.search(wh + "\?[^\.]*[\. ]*$", question_str.lower())
        if m:
            wh_word_offset_matches = [(wh, m.start())]
            break
        else:
            # Otherwise, find the wh-word in the last sentence
            m = re.search(wh + "[ ,][^\.]*[\. ]*$", question_str.lower())
            if m:
                wh_word_offset_matches.append((wh, m.start()))

    # If a wh-word is found
    if len(wh_word_offset_matches):
        # Pick the first wh-word as the word to be replaced with BLANK
        # E.g. Which is most likely needed when describing the change in position of an object?
        wh_word_offset_matches.sort(key=lambda x: x[1])
        wh_word_found = wh_word_offset_matches[0][0]
        wh_word_start_offset = wh_word_offset_matches[0][1]
        # Replace the last question mark with period.
        question_str = re.sub("\?$", ".", question_str.strip())
        # Introduce the blank in place of the wh-word
        fitb_question = (question_str[:wh_word_start_offset] + BLANK_STR +
                         question_str[wh_word_start_offset + len(wh_word_found):])
        # Drop "of the following" as it doesn't make sense in the absence of a multiple-choice
        # question. E.g. "Which of the following force ..." -> "___ force ..."
        return fitb_question.replace(BLANK_STR + " of the following", BLANK_STR)
    elif re.match(".*[^\.\?] *$", question_str):
        # If no wh-word is found and the question ends without a period/question, introduce a
        # blank at the end. e.g. The gravitational force exerted by an object depends on its
        return question_str + " " + BLANK_STR
    else:
        # If all else fails, assume "this ?" indicates the blank. Used in Turk-authored questions
        # e.g. Virtually every task performed by living organisms requires this?
        if question_str.startswith("can"):
            return re.sub("can", " ___ ", question_str,1)
        elif question_str.startswith("do "):
            return re.sub("do", " ___ ", question_str,1)
        elif question_str.startswith("does "):
            return re.sub("does", " ___ ", question_str,1)
        return re.sub(" this[ \?]", " ___ ", question_str)




# create convert to hypothesis for Stage Indicator questions
def create_hypothesis_stage_indicator(question, op, stageOriginal, stageNew):
    op = re.sub('when|once','',op.lower(),1)
    out = "In the "+ stageNew +" stage," + op
    print(out)
    return out

# create convert to hypothesis for Stage Indicator questions
def create_hypothesis_lookup(question, op):

    op = op.lower().strip()
    q = question.lower().strip()
    p_when = ["when is", 'when are', 'when will', 'when can', 'when does', 'when do']
    p_during = ["during which life stage will","during which life stage can", "during which life stage do",
                "during which stage does", "during which stage do", "during which stage will",
                "during which stage",
                "at which life stage will", "at which life stage can", "at which life stage do",
                "at which stage does", "at which stage do", "at which stage will",

                "at what life stage will", "at what life stage can", "at what life stage do",
                "at what stage does", "at what stage do", "at what stage will",

                "in which stage does", "in which stage will", "in which stage do", "on which stage","on what stage"
                "at which stage","at what stage", "in which stage","in what stage","since which stage",
                "during which", "during when", "at what",'at which', 'since when', 'which stage','what stage']
    out = ''

    for pat in p_when:
        if pat in q:
            out = q.replace(pat,'').replace('?','').strip()
            if 'when' in op:
                out = out + ' ' +op
            elif 'once' in op:
                out = out + ' ' +op
            elif 'during' in op:
                out = out + ' ' +op
            else:
                out = out +' when it is a '+ op
            return out
    q = ' '+q
    for pat in p_during:
        if ' '+pat in q:
            if 'when' in op:
                out = q.replace(pat,op).replace('?','').strip()
            elif 'once' in op:
                out = q.replace(pat, op).replace('?','').strip()
            elif 'during' in op:
                out = q.replace(pat, op).replace('?','').strip()
            else:
                out = q.replace(pat, 'in ' + op.replace('stage','').strip()+' stage, ').replace('?','').strip()
            return out

    q = question.lower().strip()
    if (q.startswith("do ") or q.startswith("does ")) and " or " in q:
        words = q.split(" ")
        pos = 0
        q = ""
        for i in range(0, len(words)):
            if words[i]=='or':
                pos = i
                break
        for i in range(0, len(words)):
            if i==pos or i == pos-1 or i==pos+1:
                continue
            q = q+ words[i] +" "
        return create_hypothesis(get_fitb_from_question(q.strip()), op)



    return  create_hypothesis(get_fitb_from_question(question), op)

def create_hypothesis_comparision(q, op, stage1, stage2):
    phrases = ["grow","survival is no longer determined","fully developed", "spread", "gaining nutrition from", "leave the protection of", "get food", "changes occur",
               "dangerous","stops happening", "immune","make","infect","eat"," fed ","feed","need",
               "require","ability"," live","develop","construct","food","breath","form","do","have"]
    replacements = ["grow","survival is determined by","have fully developed", "spread", "gain nutrition from", "need the protection of", "get food from", "",
               "faces danger from","", "immune to","make","gets infected by ","eat"," fed by ","feed","need",
               "require","have the ability to","live in ","develop","construct","eat","breath with","form","can","have"]

    h1 = ""
    h2 = ""
    # comparative
    if (" more " in q and " than " in q) or (" appear " in q and " compared to " in q) \
            or (" more " in q and " compared to " in q):
        return h1,h2
    q = q.replace(". What is one of those things?","")

    # contrast between two stages
    if " enter" in q or "becomes" in q or " into " in q \
            or "becoming" in q \
            or ("develop" in q and ("from" in q or "before" in q or "past" in q or "after" in q or "into" in q)) \
            or ("before" in q and ("maturing" in q or "becoming" in q or "entering" or "changing")) \
            or "changing from" in q or "turn into" in q or "transition to" in q or ("changes from" in q) \
            or ("move" in q or "from" in q) or (("before" in q or "when" in q or "once" in q) and "become" in q) \
            or ("from" in q and "to" in q) or "emerge" in q or "hatch" in q\
            or ("unlike before," in q ):

        neg = False
        q = q + ' '
        if "stop" in q or "loses" in q or "loose" in q or " no " in q or " not " in q or "n't" in q \
                or "disappears" in q or "leave the protection of" in q or "before" in q:
            neg = True

        phrase = None
        i = 0
        for s in phrases:
            if s in q:
                phrase = replacements[i]
                break
            i = i + 1
        if phrase is None:
            phrase = "have"

        if phrase == "can" and neg is False:
            h1 = stage1 + " can " + op
            h2 = stage2 + " can not " + op
        elif phrase != "can" and neg is False:
            h1 = stage1 + " " + phrase + " " + op
            h2 = stage2 + " do not " + phrase + " " + op
        elif phrase == "can" and neg is True:
            h1 = stage1 + " can not " + op
            h2 = stage2 + " can " + op
        elif phrase != "can" and neg is True:
            h1 = stage1 + " do not " + phrase + " " + op
            h2 = stage2 + " " + phrase + " " + op
    elif ("Unlike " in q) or "as opposed to" in q or ((" that " in q or " but " in q or " and " in q ) and ("not" in q or "n't" in q or " no "
                                                      in q or " lack" in q or "disappear" in q
                                                                              or " stop" in q or "different" in q)):
        q = q.lower().replace("?","").strip()
        q = q.replace("what does","").replace("what sensory organ is","").replace("what kind of","").replace("what is a type of food that","") \
            .replace("what is one thing that", "")\
            .replace("what is the name of the marks that", "").replace("what is something that","").replace("what is one thing","").\
            replace("name one thing that","").replace("name something that","")\
            .replace("what physical feature do","").replace("what feature do","").replace("what features do","").\
            replace("what is something","").replace("what is it","").replace("what is this","").\
            replace("what can","").replace("what do","")\
            .replace("what are they","").replace("what are","").replace("what is that","").replace("what is","")\
            .replace("which of these","").replace("which body part","").replace("which ability","")\
            .replace("which thing","")
        if " that " in q:
            h1,h2 = q.split(" that ")
        elif " but " in q:
            h1, h2 = q.split(" but ")
        elif " and " in q:
            h1, h2 = q.split(" and ")
        elif "unlike" in q:
            h1, h2 = q.split(",")
        elif "opposed to" in q:
            h1,h2 = q.split(" as opposed to")
        elif " which " in q:
            h1, h2 = q.split(" which ")

        neg = False
        if h1!=''and "unlike" in h1 and (" not" not in h2 or " no " not in h2 or "n't" not in h2):
            neg = True

        phrase = None
        i=0
        for s in phrases:
            if s in q:
                phrase = replacements[i]
                break
            i= i+1
        if phrase is None:
            phrase = ""

        if h1!="":
            h1 = h1 +" "
            h1 = h1.replace("what","").replace("type","").replace(" of ","").replace(" do "," ").\
                replace("which","").replace("unlike","").replace("able","").replace(" to ","")\
                .replace("this","").replace(",","").replace("something","").replace("these","").strip()

        h2 = h2.replace("what", "").replace(" of ", "").replace(" do ", "")\
            .replace("which", "").replace(".","").replace("?","").replace(",","").strip()

        if neg and h1!="":
            h1 = h1 +" not "+op
            if phrase not in h2:
                h2 = h2 + " " + phrase + " " + op
            else:
                h2 = h2 + " " + op
        elif h1!="":
            h1 = h1 + " " + " " + op
            if phrase not in h2:
                h2 = h2 + " " + phrase + " " + op
            else:
                h2 = h2 + " " + op

    if h1.strip()=="" or h2.strip()=="":
        h1 = ""
        h2 = ""
    return h1,h2

if __name__ == "__main__":

    print(create_hypothesis_comparision("How do baby chick get food before maturing into adult penguins?", "X","adult","baby chick"))