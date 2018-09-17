import os
import re
from allennlp.semparse.executors.life_cycle.life_cycle_parser_custom import LifeCycleParser

class LifeCycleExecutor:
    def __init__(self, archive_file=None):
        self.parser = LifeCycleParser(archive_file=archive_file) # BaselineParser() # LifeCycleParser() # GoldParser() #

        self.filepath = os.path.dirname(os.path.realpath(__file__))
        path_to_kb = os.path.join(self.filepath, 'kb.asp')
        path_to_theory = os.path.join(self.filepath, 'theory_cache.asp')
        self._path_to_query = os.path.join(self.filepath, 'query.asp')
        path_to_seq = os.path.join(self.filepath, 'seq_ds.asp')

        self.cmd = "clingo  --verbose=0 --warn no-atom-undefined '"+ path_to_kb + "'  '" + path_to_theory + "' '" + self._path_to_query + "' '"+path_to_seq+"'"

        self.conf_pat_a =re.compile('confidence\(a,(.*?)\)')
        self.conf_pat_b = re.compile('confidence\(b,(.*?)\)')
        self.ans_pat = re.compile('ans\((.*?)\)')

    def execute(self, question, prediction, url=None, organism=None):
        split_q = re.split(r' *\([A-F]\) *', question)
        if len(split_q) != 3:
            split_q = [question, None, None]
        (question_core, op1, op2) = split_q
        res = self.query(question, op1, op2, url, organism, prediction)
        return res

    def query(self, question, op1=None, op2=None, src=None,organism=None, prediction=None):
        """

        :param question: a string representig the question
        :param op1: a string representng  choice 1
        :param op2: a string representng  choice 2
        :param src: optional, if you want to restrict the solver to only one document, specify the name here
        :return: a json object
        """
        logical_form = self.parser.parse(question, op1, op2, src,organism, prediction)

        #logical_form = logical_form + " qType(" + qtype + ")."

        if logical_form is None:
            out = {"best_option":None}
            return out

        ans, ca, cb = self.solve(logical_form)

        out = {}

        out["logical_form"] = logical_form.replace("\n", "")
        out["answer_index"] = -1

        if op1 is not None:

            try:
                confidences = [float(x.strip('"')) for x in [ca, cb]]
            except:
                confidences = [0,0]
            out["confidences"] = confidences
            if ans=='a':
                out["answer"] = op1
                out["answer_index"] = 0
            elif ans=='b':
                out["answer"] = op2
                out["answer_index"] = 1
            else:
                out["answer"] = "N/A"
        else:
            out["answer"] = ans

        return out

    def solve(self, logical_form):
        """

        :param logical_form: the ASP representation of the question and the options
        :return:
            "ans" denoting which of the option is correct
            "confidence_a" : confidence in option a
            "confidence_b": confidence in option b
        """

        with open(self._path_to_query, "w") as query_file:
            print(logical_form, file=query_file)

        ans = None
        confidence_a = None
        confidence_b = None

        current_dir = os.getcwd()
        try:
            os.chdir(self.filepath)

            output = os.popen(self.cmd).read()
            # print(output)
            confidence_a = self.conf_pat_a.findall(output)
            if len(confidence_a)==1:
                confidence_a = confidence_a[0]
            confidence_b = self.conf_pat_b.findall(output)
            if len(confidence_b)==1:
                confidence_b = confidence_b[0]
            ans = self.ans_pat.findall(output)
            if len(ans)==1:
                ans = ans[0]
        except:
            print("exception for: "+ logical_form)

        os.chdir(current_dir)

        return ans, confidence_a, confidence_b

