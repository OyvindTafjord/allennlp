length(seq(null),0).
length(seq(X,Y),N+1):-length(Y,N), seq(X,Y).
memberOf(seq(X,Y),X):-seq(X,Y).
memberOf(seq(X,Y),Z):-seq(X,Y), memberOf(Y,Z).
elementAt(seq(X,Y),1,X):-seq(X,Y).
elementAt(seq(X,seq(Y,Z)),N,E):-seq(X,seq(Y,Z)), elementAt(seq(Y,Z),N-1,E), length(seq(X,seq(Y,Z)),M),N<=M,N>=2.

seq(X,Y):-option(_,seq(X,Y)).
seq(Y):- seq(X,seq(Y)).
seq(Y,Z):- seq(X,seq(Y,Z)).

