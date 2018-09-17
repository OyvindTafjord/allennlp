%vocabulary
% organism(X) :- X is an organism
% stageAt(src, life_cycle, organism, position, name )
% nextStage(src, life_cycle, organism, name1, name2 )
% numberOfStages(src, lifecycle, organism, number)
% description(src, lifecycle, organism, text)



source(SRC):- description(SRC, life_cycle, O, P).
usable(SRC):- source(SRC),  {useOnly(S):source(S)} 0, forOrganism(O), description(SRC, life_cycle, O, _)  .
usable(SRC):- useOnly(SRC), 1 {useOnly(S):source(S)}.

qType(seq):- 1 {qType(qIsAStageOf);qType(qIsNotAStageOf); qType(qStageAt); qType(qNextStage);
        qType(qCountStages); qType(qCorrectlyOrdered); qType(qStageBetween); qType(qStageBefore) }.
forOrganism(O):- qIsAStageOf(_,O).
forOrganism(O):- qIsNotAStageOf(_,O).
forOrganism(O):- qCountStages(_,O).
forOrganism(O):- qCorrectlyOrdered(_,O).
forOrganism(O):- qNextStage(_,O,_).
forOrganism(O):- qStageBefore(_,O,_).
forOrganism(O):- qStageAt(_,O,_).
forOrganism(O):- qStageBetween(_,O,_,_).
forOrganism(O):- qLookup(_,O).
forOrganism(O):- qStageIndicator(_,O,_).

confidence(X,1):- qType(seq), ans(X).
confidence(X,0):- qType(seq), not ans(X), optionNo(X).

optionNo(X):- option(X,B).


isAStageOf(SRC, life_cycle, O, S):-stageAt(SRC, life_cycle, O, POS, S), usable(SRC).
ans(X):- qIsAStageOf(life_cycle,O), usable(SRC), optionNo(X), C1 = #count {S:isAStageOf(SRC, life_cycle, O, S),option(X,S)}, C2= #count {S:option(X,S)}, C1==C2.

nans(X):- qIsNotAStageOf(life_cycle,O), optionNo(X), usable(SRC), optionNo(X),
    C1 = #count {S:isAStageOf(SRC, life_cycle, O, S),option(X,S)}, C2= #count {S:option(X,S)}, C1==C2.
ans(X):- not nans(X), optionNo(X),qIsNotAStageOf(life_cycle,O).


ans(X):- qStageAt(life_cycle,O,Loc), wordToIndex(SRC, O, Loc, Pos), stageAt(SRC, life_cycle, O, Pos, S ), option(X,S), usable(SRC).
wordToIndex(SRC, O, last, Pos):- qStageAt(life_cycle, O, last), Pos = #max {P:stageAt(SRC, life_cycle, O, P, S )}, usable(SRC).
wordToIndex(SRC, O, middle, Pos):- qStageAt(life_cycle, O, middle), Last = #max {P:stageAt(SRC, life_cycle, O, P, S )}, Pos = (Last+1)/2, usable(SRC).
wordToIndex(SRC, O, middle, Pos):- qStageAt(life_cycle, O, middle), Last = #max {P:stageAt(SRC, life_cycle, O, P, S )}, Pos = (Last/2+1), Last\2==0, usable(SRC).
wordToIndex(SRC, O, Loc, Loc):- Loc!=last, Loc!=middle, qStageAt(life_cycle, O, Loc), usable(SRC).

nextStage(SRC, life_cycle, O, S, S1):- usable(SRC), stageAt(SRC, life_cycle, O, P,S), stageAt(SRC, life_cycle, O, P+1,S1).
afterStage(SRC, life_cycle, O, S, S1):- usable(SRC), stageAt(SRC, life_cycle, O, P,S), stageAt(SRC, life_cycle, O, Q,S1), Q>P.

ans(X):- qNextStage(life_cycle, O, S), option(X,N),   usable(SRC),   afterStage(SRC, life_cycle, O, S, N),
#count { S1:option(Y,S1), afterStage(SRC, life_cycle, O, S, S1), afterStage(SRC, life_cycle, O, S1, N) } = 0.

ans(X):- qStageBefore(life_cycle, O, S), option(X,N),   usable(SRC),   afterStage(SRC, life_cycle, O, N,S),
#count { S1:option(Y,S1), afterStage(SRC, life_cycle, O, N, S1), afterStage(SRC, life_cycle, O, S1, S) } = 0.

ans(X):- qStageBetween(life_cycle, O, S1,S2), option(X,N),   usable(SRC),   afterStage(SRC, life_cycle, O, S1, N), afterStage(SRC, life_cycle, O, N, S2).

ans(X):- qCountStages(life_cycle, O), option(X,N), usable(SRC), N == #max { P:stageAt(SRC, life_cycle, O, P,_ )}.

ans(X):- qCorrectlyOrdered(life_cycle, O), usable(SRC), option(X,N), L = #count {I:elementAt(N,I,S),stageAt(SRC, life_cycle, O, I, S)}, length(N,L).

%%%%%%%%%%%%%%%%%%%%%%% look up %%%%%%%%%%%%%%%%%%%
qType(weighted):-qType(qLookup).
confidence(SRC, X,V):- usable(SRC), qType(qLookup), question(Q), option(X,O), H = @hypothesis(Q,O), V = @entailment(P,H), description(SRC,_,_,P).
confidence(X,V):- qType(qLookup), V = #max {V1: confidence(SRC, X,V1)}, optionNo(X).
ans(X):- qType(weighted), confidence(X,V), V == #max {V1:confidence(X1,V1)}.


%%%%%%%%%%%%%%%%%%%%% indicator %%%%%%%%%%%%%%%%%%%%
qType(weighted):-qType(qStageIndicator).
stageId(SRC,ID):- qType(qStageIndicator),stageAt(SRC, _, _, ID, _), usable(SRC).
stageIndicatorId(SRC,ID):- stageAt(SRC, _, _, ID, S), usable(SRC),qStageIndicator(_,_, S).
trueForStage(SRC, life_cycle, O, Id,X,V):- qStageIndicator(life_cycle,O, S), question(Q), option(X,N),
                                           usable(SRC), stageId(SRC,Id),
                                           V = #max {@entailment(Text, H): description(SRC, life_cycle, O, Text),
                                           stageAt(SRC, life_cycle, O, Id, S1), H = @hypothesisStageIndicator(Q,N, S, S1)}.

res(SRC, life_cycle, O, 1, X , @product("1.0",V,1,ID)):-trueForStage(SRC, life_cycle, O, 1,X,V), stageIndicatorId(SRC,ID).
res(SRC, life_cycle, O, N, X, @product(V1,V2,N,ID)):-res(SRC, life_cycle, O, N-1, X , V1), trueForStage(SRC, life_cycle, O, N,X,V2),stageIndicatorId(SRC,ID). % iterative multiplication
finalResult(SRC, life_cycle, O, X , V):-res(SRC, life_cycle, O, N, X , V),  N = #max {P:stageAt(SRC, life_cycle, O, P, S )}. % the final result

confidence(X,V):- V = #max {Val:finalResult(SRC, life_cycle, O, X , Val)},qType(qStageIndicator), optionNo(X).

%%%%%%%%%%%%%%%%%%%%%%% look up %%%%%%%%%%%%%%%%%%%
qType(weighted):-qType(qStageDifference).
stage1Id(ID):- qType(qStageDifference), stageAt(SRC, _, _, ID, ST1), usable(SRC), qStageDifference(_,_,ST1,_) .
stage2Id(ID):- qType(qStageDifference), stageAt(SRC, _, _, ID, ST2), usable(SRC), qStageDifference(_,_,_,ST2) .

stage1Name(ST):- qStageDifference(_,_,ST,_), stage1Id(ID1), stage2Id(ID2), ID1<ID2.
stage1Name(ST):- qStageDifference(_,_,_,ST), stage1Id(ID1), stage2Id(ID2), ID1>ID2.

stage2Name(ST):- qStageDifference(_,_,ST,_), stage1Id(ID1), stage2Id(ID2), ID1>ID2.
stage2Name(ST):- qStageDifference(_,_,_,ST), stage1Id(ID1), stage2Id(ID2), ID1<ID2.


confidence(SRC, X,V):- usable(SRC), qType(qStageDifference), question(Q), option(X,O),
              (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1),
              H1!="", H2!="", V1 = @entailment(P,H1), V2 = @entailment(P,H2),
              V= @multiply(V1,V2), description(SRC,_,_,P).

confidence(SRC, X,V):- usable(SRC), qType(qStageDifference), question(Q), option(X,O),
              (H1,H2) = @hypothesisDifference(Q,O, ST2,ST1), stage2Name(ST2), stage1Name(ST1),
              H1=="", H2=="", H = @hypothesis(Q,O), V = @entailment(P,H), description(SRC,_,_,P).

confidence(X,V):- qType(qStageDifference), V = #max {V1: confidence(SRC, X,V1)}, optionNo(X) .

#show get/2.
%output
#show ans/1.
#show confidence/2.
%#show qType/1.
%#show optionNo/1.
%#show isAStageOf/4.
#show usable/1.
#show wordToIndex/4.
#show trueForStage/6.
%#show res/6.
#show finalResult/5.
#show confidence/2.


#script(python)

import warnings
warnings.filterwarnings("ignore")
from cached_entailment import *
from convert_to_entailment import *
import random

def multiply(x,y):
    res = float(x.string) * float(y.string)
    return str(format(res,'.8f'))

def product(x,y, id, id_target):
    res = 0
    if id == id_target:
        res = float(x.string) * float(y.string)

    else:
        res = float(x.string) * (1-float(y.string))

    return str(format(res,'.8f'))


def entailment(text, hyp):
    ent = Entailment()
    res = ent.enatailment(text.string,hyp.string)
    return str(format(res,'.8f'))

def hypothesis(question, op):
    return str(create_hypothesis(get_fitb_from_question(question.string), op.string))

def hypothesisStageIndicator(question, op, stageOriginal, stageNew):
    return str(create_hypothesis_stage_indicator(question.string, op.string, stageOriginal.string, stageNew.string))

def hypothesisDifference(question, op, stage1, stage2):
    h1,h2 = create_hypothesis_comparision(question.string, op.string, stage1.string, stage2.string)
    return str(h1),str(h2)

#end.





