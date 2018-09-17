import en_core_web_sm

import json
import requests
class Entailment:

    en_nlp = en_core_web_sm.load()
    e_map = {}
    # with open("cache_entailment_decomp.json","r") as f:
    #     e_map = json.load(f)


    def entailment_sentence(self, premise, hypothesis):
        key = premise+"@"+hypothesis
        result = 0
        if key in self.e_map:
            result =  self.e_map[key]
        else:
            result = self.entailment_sentence_ai2(premise,hypothesis)
            self.e_map[key] = result
            #print("key did not found in entailment")
            #self.save()
        return result

    def entailment_sentence_ai2(self,premise, hypothesis):
        data = {'t': premise, 'h': hypothesis}
        url = 'http://aristo-ks.allenai.org/entails'

        # GET with params in URL
        results = requests.get(url, params=data)

        f = results.text.split("<confidence>")[1]
        conf = float(f[0:f.find('<')])  # results.json()["confidence"]
        return conf

    def enatailment(self, text, hypothesis):
        max_score = 0
        support = ""
        doc = self.en_nlp(text)  # create a Doc from raw text
        sentences = list(doc.sents)
        stage_ctx = None
        print("LEN sentences = " + str(len(sentences)))
        for sen in sentences:
            premise = sen.text.replace('\n', ' ').replace('\r', '')
            if "::stage" in premise:
                premise = premise.replace("::stage ","")
                stage_ctx = premise[0:premise.find(":")]
                premise = premise[premise.find("::")+3:]
                if premise.strip()=="":
                    continue

            score = self.entailment_sentence(premise,hypothesis)

            #print(premise,score)
            if score> max_score:
                max_score = score
                support = premise

        #print(hypothesis)
        #print(support)

        return max_score #,support

    # def save(self):
    #     f = open("cache_entailment_peter_2.json", "w")
    #     json.dump(self.e_map, f)
    #     f.close()