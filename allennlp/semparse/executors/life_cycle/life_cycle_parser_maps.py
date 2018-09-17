import os
import re
import json

class LifeCycleParserMaps:

    def __init__(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir, 'parser_maps/organisms.json'), 'r') as f:
            self.organisms = json.load(f)
        self.organism_list = sorted(self.organisms.keys(),key=len,reverse=True)

        with open(os.path.join(dir, 'parser_maps/stages.json'), 'r') as f:
            self.stages_map = json.load(f)
        self.stage_list = set(list(self.stages_map.keys()) + list(self.stages_map.values()))
        self.stage_list = sorted(self.stage_list, key=len, reverse=True)

        with open(os.path.join(dir, 'parser_maps/numbers.json'), 'r') as f:
            self.num_map = json.load(f)


    def get_organism_name(self, q):
        q = q.lower()
        for o in self.organism_list:
            if o in q:

                return self.organisms[o].strip(),q.replace(o,"")
            elif self.organisms[o] in q:
                return self.organisms[o].strip(),q.replace(o,"")
        return None,q

    def get_stages(self, q):
        q = q.lower()
        ret = []
        for s in self.stage_list:
            if s in q:
                if s in self.stages_map:
                    ret.append(self.stages_map.get(s).strip())
                else:
                    ret.append(s.strip())
                q = re.sub(s,"",q,1)

        return ret,q

    def getStageNumber(self, q):
        q1 = q.split(".")
        if len(q1)==2 and len(q1[1])>5:
            q = q1[1]

        q = q.replace("?","").replace(".","")+" "
        out = []
        for key in self.num_map:
            if key+" " in q:
                out.append(self.num_map[key].strip())

        if len(out)==1:
            return out[0],q
        elif len(out)>1:
            if out[-1]=='1':
                return out[-2],q

        return None,q

    def getArgumentAtPosition(self, qtype, pos, state, q):
        if pos==1:
            o,q1 = self.get_organism_name(q)
            return o,q1,None
        if pos==2 and qtype=='qStageAt':
            o, q1 = self.getStageNumber(q)
            return o, q1, None
        if pos==2 and (qtype=='qNextStage'or qtype =='qStageBefore' or qtype=='qStageIndicator'):
            o1, q1 = self.get_stages(q)
            o = 'adult'
            if len(o1)==1:
                o = o1[0]
            elif len(o1)>1:
                for s in o1:
                    if s!='adult':
                        o = s
                        break
            return o, q1, None

        if pos==2 and qtype=='qStageDifference':
            o1, q1 = self.get_stages(q)
            next = None
            o = None
            if len(o1)==1:
                return o1[0], q1, 'adult'
            elif len(o1)==2:
                return o1[0], q1, o1[1]
            elif len(o1)>2:
                next = 'adult'
                for s in o1:
                    if s!='adult' and o is None:
                        o = s
                    elif s!='adult':
                        next = s
                if s is None:
                    next  = None
            return o, q1, next

        if pos == 3 and qtype == 'qStageDifference':
            return state,q,None

        #print("missed", qtype,pos)