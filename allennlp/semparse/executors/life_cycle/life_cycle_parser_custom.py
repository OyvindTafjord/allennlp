import sys
sys.path.insert(0,'/Users/tafjord/gitroot/allennlp')
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.semparse.executors.life_cycle.life_cycle_parser_maps import LifeCycleParserMaps

class LifeCycleParser:

    def __init__(self, archive_file=None):
        if archive_file is not None:
            archive = load_archive(archive_file)
            self.predictor = Predictor.from_archive(archive)
        self.parser_maps = LifeCycleParserMaps()


    def parse(self, q, o1, o2, url=None, organism=None, prediction = None):

        key = q + " (A) " + o1 + " (B) " + o2

        inp = {"question": key}
        if prediction is None:
            output = self.predictor.predict_json(inp)
        else:
            output = prediction
        lf = output['logical_form'].replace('(', '').replace(')', '').strip().split(' ')
        qtype = lf[0]

        ret = 'qType('+lf[0]+').\n'+lf[0]+'(life_cycle,'
        q1 = q
        state = None
        for i in range(1, len(lf)):

            p_term = lf[i].replace("o:","").replace("s:","")
            term,q1,state = self.parser_maps.getArgumentAtPosition(qtype,i,state,q1)
            if term is None:
                term = p_term
            if i==1 and organism is not None:
                term = organism
            if qtype=='qStageAt'and i==2:
                ret = ret+ term+','
            else:
                if term=='post_spawn':
                    ret = ret + '\"' + 'post-span' + '\",'
                elif term==organism:
                    ret = ret + '' + term.replace('_', ' ') + ','
                else:
                    ret = ret + '\"' + term.replace('_', ' ') + '\",'
        ret = ret[0:len(ret)-1]+').'
        if qtype == 'qCountStages':
            optionA = 'option(a,' + o1.strip().lower() + ').'
            optionB = 'option(b,' + o2.strip().lower() + ').'
        elif qtype == 'qIsAStageOf':
            op1 = o1.replace(' and ',',').split(',')
            optionA=''
            for s in op1:
                optionA = optionA + ' option(a,\"' + s.strip().lower() + '\").'

            op2 = o2.replace(' and ', ',').split(',')
            optionB = ''
            for s in op2:
                optionB = optionB + ' option(b,\"' + s.strip().lower() + '\").'
        elif qtype == 'qCorrectlyOrdered':
            op1 = o1.replace(' and ', ',').split(',')
            optionA = 'option(a,'
            end ='seq(null))'
            for s in op1:
                optionA = optionA + 'seq(\"' + s.strip().lower() + '\",'
                end = end +')'
            optionA = optionA + end +'.'
            op2 = o2.replace(' and ', ',').split(',')
            optionB = 'option(b,'
            end = 'seq(null))'
            for s in op2:
                optionB = optionB + 'seq(\"' + s.strip().lower() + '\",'
                end = end + ')'
            optionB = optionB + end +'.'
        else:
            optionA = 'option(a,\"' + o1.strip().lower() + '\").'
            optionB = 'option(b,\"' + o2.strip().lower() + '\").'

        if q.startswith('"') == False:
            q = "\"" + q + "\""

        logicalForm = ret + '\n' + optionA + '\n' + optionB + "\nquestion("+q+")."
        # print("url "+ str(url))
        if url is not None:
            logicalForm = logicalForm + "\nuseOnly("+url+")."

        # print(logicalForm)
        return logicalForm
