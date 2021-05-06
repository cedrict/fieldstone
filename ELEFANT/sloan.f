      SUBROUTINE DIAMTR_SLOAN(N,E2,ADJ,XADJ,MASK,LS,XLS,HLEVEL,SNODE,NC)
************************************************************************
*
*     PURPOSE:
*     --------
*
*     Find nodes which define a psuedo-diameter of a graph and store
*     distances from end node
*
*     INPUT:
*     ------
*
*     N      - The total number of nodes in the graph
*     E2     - Twice the number of edges in the graph  = XADJ(N+1)-1
*     ADJ    - Adjacency list for all nodes in the graph
*            - List of length 2E where E is the number of edges in 
*              the graph and 2E = XADJ(N+1)-1
*     XADJ   - Index vector for ADJ
*            - Nodes adjacent to node I are found in ADJ(J), where
*              J = XADJ(I), XADJ(I)+1, ...,XADJ(I+1)-1
*            - Degree of node I given by XADJ(I+1)-XADJ(I)
*     MASK   - Masking vector for graph
*            - Visible nodes have MASK = 0, node invisible otherwise
*     LS     - Undefined
*     XLS    - Undefined
*     HLEVEL - Undefined
*     SNODE  - Undefined
*     NC     - Undefined
*
*     OUTPUT:
*     ------
*
*     N      - Unchanged
*     E2     - Unchanged
*     ADJ    - Unchanged
*     XADJ   - Unchanged
*     MASK   - List of distances of nodes from the end node
*     LS     - List of nodes in the component
*     XLS    - Not used
*     HLEVEL - Not used
*     SNODE  - Starting node for numbering
*     NC     - The number of nodes in this component of graph
*
*     SUBROUTINES CALLED:  ROOTLS, ISORTI
*     -------------------
*
*     NOTE:      SNODE and ENODE define a pseudo-diameter
*     -----
*
*     PROGRAMMER:             Scott Sloan
*     -----------
*
*     LAST MODIFIED:          1 March 1991      Scott Sloan
*     --------------
*
*     COPYRIGHT 1989:         Scott Sloan
*     ---------------         Department of Civil Engineering
*                             University of Newcastle
*                             NSW 2308
*
************************************************************************
      INTEGER I,J,N
      INTEGER E2,NC
      INTEGER NODE
      INTEGER DEPTH,ENODE,HSIZE,ISTOP,ISTRT,JSTOP,JSTRT,SNODE,WIDTH
      INTEGER DEGREE,EWIDTH,MINDEG,SDEPTH
      INTEGER LS(N)
      INTEGER ADJ(E2),XLS(N+1)
      INTEGER MASK(N),XADJ(N+1)
      INTEGER HLEVEL(N)
*
*     Choose first guess for starting node by min degree
*     Ignore nodes that are invisible (MASK ne 0)
*
      MINDEG=N
      DO 10 I=1,N
        IF(MASK(I).EQ.0)THEN
          DEGREE=XADJ(I+1)-XADJ(I)
          IF(DEGREE.LT.MINDEG)THEN
            SNODE=I
            MINDEG=DEGREE
          END IF
        END IF
   10 CONTINUE
*
*     Generate level structure for node with min degree
*
      CALL ROOTLS_SLOAN
     &     (N,SNODE,N+1,E2,ADJ,XADJ,MASK,LS,XLS,SDEPTH,WIDTH)
*
*     Store number of nodes in this component
*
      NC=XLS(SDEPTH+1)-1
*
*     Iterate to find start and end nodes
*     
   15 CONTINUE
*
*     Store list of nodes that are at max distance from starting node
*     Store their degrees in XLS
*
      HSIZE=0
      ISTRT=XLS(SDEPTH)
      ISTOP=XLS(SDEPTH+1)-1
      DO 20 I=ISTRT,ISTOP
        NODE=LS(I)
        HSIZE=HSIZE+1
        HLEVEL(HSIZE)=NODE
        XLS(NODE)=XADJ(NODE+1)-XADJ(NODE)
   20 CONTINUE
*
*     Sort list of nodes in ascending sequence of their degree
*     Use insertion sort algorithm
*
      IF(HSIZE.GT.1)CALL ISORTI_SLOAN(HSIZE,HLEVEL,N,XLS)
*
*     Remove nodes with duplicate degrees
*
      ISTOP=HSIZE
      HSIZE=1
      DEGREE=XLS(HLEVEL(1))
      DO 25 I=2,ISTOP
        NODE=HLEVEL(I)
        IF(XLS(NODE).NE.DEGREE)THEN
          DEGREE=XLS(NODE)
          HSIZE=HSIZE+1
          HLEVEL(HSIZE)=NODE
        ENDIF
   25 CONTINUE
*
*     Loop over nodes in shrunken level
*
      EWIDTH=NC+1
      DO 30 I=1,HSIZE
        NODE=HLEVEL(I)
*
*       Form rooted level structures for each node in shrunken level
*
        CALL ROOTLS_SLOAN
     &     (N,NODE,EWIDTH,E2,ADJ,XADJ,MASK,LS,XLS,DEPTH,WIDTH)
        IF(WIDTH.LT.EWIDTH)THEN
*
*         Level structure was not aborted during assembly
*
          IF(DEPTH.GT.SDEPTH)THEN
*
*           Level structure of greater depth found
*           Store new starting node, new max depth, and begin 
*           a new iteration
*
            SNODE=NODE
            SDEPTH=DEPTH
            GOTO 15
          ENDIF
*
*         Level structure width for this end node is smallest so far
*         store end node and new min width
*
          ENODE=NODE
          EWIDTH=WIDTH
        END IF
   30 CONTINUE
*
*     Generate level structure rooted at end node if necessary
*
      IF(NODE.NE.ENODE)THEN
        CALL ROOTLS_SLOAN
     &     (N,ENODE,NC+1,E2,ADJ,XADJ,MASK,LS,XLS,DEPTH,WIDTH)
      ENDIF
*
*     Store distances of each node from end node
*
      DO 50 I=1,DEPTH
        JSTRT=XLS(I)
        JSTOP=XLS(I+1)-1
        DO 40 J=JSTRT,JSTOP
          MASK(LS(J))=I-1
   40   CONTINUE
   50 CONTINUE
      END
      SUBROUTINE GRAPH_SLOAN(N,NE,INPN,NPN,XNPN,IADJ,ADJ,XADJ)
************************************************************************
*
*     PURPOSE:
*     --------
*
*     Form adjacency list for a graph corresponding to a finite element
*     mesh
*
*     INPUT:
*     -----
*
*     N      - Number of nodes in graph (finite element mesh)
*     NE     - Number of elements in finite element mesh
*     INPN   - Length of NPN = XNPN(NE+1)-1
*     NPN    - List of node numbers for each element
*     XNPN   - Index vector for NPN
*            - nodes for element I are found in NPN(J), where
*              J = XNPN(I), XNPN(I)+1, ..., XNPN(I+1)-1
*     IADJ   - Length of vector ADJ
*            - Set IADJ=NE*NEN*(NEN-1) for a mesh of a single type of
*              element with NEN nodes
*            - IADJ=(NEN(1)*(NEN(1)-1)+,.....,+NEN(NE)*(NEN(NE)-1))
*              for a mesh of elements with varying numbers of nodes
*     ADJ    - Undefined
*     XADJ   - Undefined
*
*     OUTPUT:
*     -------
*
*     N      - Unchanged
*     NE     - Unchanged
*     INPN   - Unchanged
*     NPN    - Unchanged
*     XNPN   - Unchanged
*     IADJ   - Unchanged
*     ADJ    - Adjacency list for all nodes in graph
*            - List of length 2E where E is the number of edges in 
*              the graph (note that 2E = XADJ(N+1)-1 )
*     XADJ   - Index vector for ADJ
*            - Nodes adjacent to node I are found in ADJ(J), where
*              J = XADJ(I), XADJ(I)+1, ..., XADJ(I+1)-1
*            - Degree of node I given by XADJ(I+1)-XADJ(I)
*
*     NOTES:
*     ------
*
*     This routine typically requires about 25 percent elbow room for
*     assembling the ADJ list (i.e. IADJ/2E is typically around 1.25).
*     In some cases, the elbow room may be larger (IADJ/2E is slightly 
*     less than 2 for the 3-noded triangle) and in other cases it may be
*     zero (IADJ/2E = 1 for bar elements)
*
*     PROGRAMMER:             Scott Sloan
*     -----------
*
*     LAST MODIFIED:          1 March 1991        Scott Sloan
*     --------------
*
*     COPYRIGHT 1989:         Scott Sloan
*     ---------------         Department of Civil Engineering
*                             University of Newcastle
*                             NSW 2308
*
************************************************************************
      INTEGER I,J,K,L,M,N
      INTEGER NE
      INTEGER IADJ,INPN,NEN1
      INTEGER JSTOP,JSTRT,LSTOP,LSTRT,MSTOP,MSTRT,NODEJ,NODEK
      INTEGER ADJ(IADJ),NPN(INPN)
      INTEGER XADJ(N+1),XNPN(NE+1)
*
*     Initialise the adjacency list and its index vector
*
      DO 5 I=1,IADJ
        ADJ(I)=0
    5 CONTINUE
      DO 10 I=1,N
        XADJ(I)=0
   10 CONTINUE
*
*     Estimate the degree of each node (always an overestimate)
*
      DO 30 I=1,NE
        JSTRT=XNPN(I)
        JSTOP=XNPN(I+1)-1
        NEN1 =JSTOP-JSTRT
        DO 20 J=JSTRT,JSTOP
          NODEJ=NPN(J)
          XADJ(NODEJ)=XADJ(NODEJ)+NEN1
   20   CONTINUE
   30 CONTINUE
*
*     Reconstruct XADJ to point to start of each set of neighbours
*
      L=1
      DO 40 I=1,N
        L=L+XADJ(I)
        XADJ(I)=L-XADJ(I)
   40 CONTINUE
      XADJ(N+1)=L
*
*     Form adjacency list (which may contain zeros)
*
      DO 90 I=1,NE
        JSTRT=XNPN(I)
        JSTOP=XNPN(I+1)-1
        DO 80 J=JSTRT,JSTOP-1
          NODEJ=NPN(J)
          LSTRT=XADJ(NODEJ)
          LSTOP=XADJ(NODEJ+1)-1
          DO 70 K=J+1,JSTOP
            NODEK=NPN(K)
            DO 50 L=LSTRT,LSTOP
              IF(ADJ(L).EQ.NODEK)GOTO 70
              IF(ADJ(L).EQ.0)GOTO 55
   50       CONTINUE
            WRITE(6,1000)
            STOP
   55       CONTINUE
            ADJ(L)=NODEK
            MSTRT=XADJ(NODEK)
            MSTOP=XADJ(NODEK+1)-1
            DO 60 M=MSTRT,MSTOP
              IF(ADJ(M).EQ.0)GOTO 65
   60       CONTINUE
            WRITE(6,1000)
            STOP
   65       CONTINUE
            ADJ(M)=NODEJ
   70     CONTINUE
   80   CONTINUE
   90 CONTINUE
*
*     Strip any zeros from adjacency list
*
      K=0
      JSTRT=1
      DO 110 I=1,N
        JSTOP=XADJ(I+1)-1
        DO 100 J=JSTRT,JSTOP
          IF(ADJ(J).EQ.0)GOTO 105
          K=K+1
          ADJ(K)=ADJ(J)
  100   CONTINUE
  105   CONTINUE
        XADJ(I+1)=K+1
        JSTRT=JSTOP+1
  110 CONTINUE
*
*     Error message
*
 1000 FORMAT(//,1X,'***ERROR IN GRAPH***',
     +       //,1X,'CANNOT ASSEMBLE NODE ADJACENCY LIST',
     +       //,1X,'CHECK NPN AND XNPN ARRAYS')
      END
      SUBROUTINE ISORTI_SLOAN(NL,LIST,NK,KEY)
************************************************************************
*
*     PURPOSE:
*     --------
*
*     Order a list of integers in ascending sequence of their keys 
*     using insertion sort
*
*     INPUT:
*     ------
*
*     NL   - Length of LIST
*     LIST - A list of integers
*     NK   - Length of KEY (NK must be ge NL)
*     KEY  - A list of integer keys
*
*     OUTPUT:
*     -------
*
*     NL   - Unchanged
*     LIST - A list of integers sorted in ascending sequence of KEY
*     NK   - Unchanged
*     KEY  - Unchanged
*
*     NOTE:    Efficient for short lists only (NL lt 20)
*     -----
*
*     PROGRAMMER:             Scott Sloan
*     -----------
*
*     LAST MODIFIED:          1 March 1991     Scott Sloan
*     --------------
*
*     COPYRIGHT 1989:         Scott Sloan
*     ---------------         Department of Civil Engineering
*                             University of Newcastle
*                             NSW 2308
*
************************************************************************
      INTEGER I,J,T
      INTEGER NL,NK
      INTEGER VALUE
      INTEGER KEY(NK)
      INTEGER LIST(NL)
*
      DO 20 I=2,NL
        T=LIST(I)
        VALUE=KEY(T)
        DO 10 J=I-1,1,-1
           IF(VALUE.GE.KEY(LIST(J)))THEN
             LIST(J+1)=T
             GOTO 20
           ENDIF
           LIST(J+1)=LIST(J)
   10   CONTINUE
        LIST(1)=T
   20 CONTINUE
      END
      SUBROUTINE LABEL_SLOAN(N,E2,ADJ,XADJ,NNN,IW,OLDPRO,NEWPRO)
************************************************************************
*
*     PURPOSE:
*     --------
*
*     Label a graph for small profile and rms wavefront
*
*     INPUT:
*     ------
*
*     N      - Total number of nodes in graph
*     E2     - Twice the number of edges in the graph = XADJ(N+1)-1
*     ADJ    - Adjacency list for all nodes in graph
*            - List of length 2E where E is the number of edges in
*              the graph and 2E = XADJ(N+1)-1
*     XADJ   - Index vector for ADJ
*            - Nodes adjacent to node I are found in ADJ(J), where
*              J = XADJ(I), XADJ(I)+1, ..., XADJ(I+1)-1
*            - Degree of node I given by XADJ(I+1)-XADJ(I)
*     NNN    - Undefined
*     IW     - Undefined
*     OLDPRO - Undefined
*     NEWPRO - Undefined
*
*     OUTPUT:
*     -------
*
*     N      - Unchanged
*     E2     - Unchanged
*     ADJ    - Unchanged
*     XADJ   - Unchanged
*     NNN    - List of new node numbers
*            - New number for node I given by NNN(I)
*            - If original node numbers give a smaller profile then
*              NNN is set so that NNN(I)=I for I=1,N
*     IW     - Not used
*     OLDPRO - Profile using original node numbering
*     NEWPRO - Profile for new node numbering
*            - If original profile is smaller than new profile, then
*              original node numbers are used and NEWPRO=OLDPRO
*
*     SUBROUTINES CALLED:  DIAMTR, NUMBER, PROFIL
*     -------------------
*
*     PROGRAMMER:             Scott Sloan
*     -----------
*
*     LAST MODIFIED:          1 March 1991        Scott Sloan
*     --------------
*
*     COPYRIGHT 1989:         Scott Sloan
*     ---------------         Department of Civil Engineering
*                             University of Newcastle
*                             NSW 2308
*
***********************************************************************
      INTEGER I,N
      INTEGER E2,I1,I2,I3,NC
      INTEGER SNODE
      INTEGER LSTNUM,NEWPRO,OLDPRO
      INTEGER IW(3*N+1)
      INTEGER ADJ(E2),NNN(N)
      INTEGER XADJ(N+1)
*
*     Set all new node numbers = 0
*     This is used to denote all visible nodes
*
      DO 10 I=1,N
        NNN(I)=0
   10 CONTINUE
*
*     Define offsets
*
      I1=1
      I2=I1+N
      I3=I2+N+1
*
*     Loop while some nodes remain unnumbered
*
      LSTNUM=0
   20 IF(LSTNUM.LT.N)THEN
*
*       Find end points of p-diameter for nodes in this component
*       Compute distances of nodes from end node
*
        CALL DIAMTR_SLOAN
     &     (N,E2,ADJ,XADJ,NNN,IW(I1),IW(I2),IW(I3),SNODE,NC)
*
*       Number nodes in this component
*
        CALL NUMBER_SLOAN
     &     (N,NC,SNODE,LSTNUM,E2,ADJ,XADJ,NNN,IW(I1),IW(I2))
        GOTO 20
      END IF
*
*     Compute profiles for old and new node numbers
*
      CALL PROFIL_SLOAN(N,NNN,E2,ADJ,XADJ,OLDPRO,NEWPRO)
*
*     Use original numbering if it gives a smaller profile
*
      IF(OLDPRO.LT.NEWPRO)THEN
        DO 30 I=1,N
          NNN(I)=I
   30   CONTINUE
        NEWPRO=OLDPRO
      END IF
      END     
      SUBROUTINE NUMBER_SLOAN(N,NC,SNODE,LSTNUM,E2,ADJ,XADJ,S,Q,P)
************************************************************************
*
*     PURPOSE:
*     --------
*
*     Number nodes in component of graph for small profile and rms
*     wavefront
*
*     INPUT:
*     ------
*
*     N      - Number of nodes in graph
*     NC     - Number of nodes in component of graph
*     SNODE  - Node at which numbering starts
*     LSTNUM - Count of nodes which have already been numbered
*     E2     - Twice tne number of edges in the graph = XADJ(N+1)-1
*     ADJ    - Adjacency list for all nodes in graph
*            - List of length 2E where E is the number of edges in
*              the graph and 2E = XADJ(N+1)-1
*     XADJ   - Index vector for ADJ
*            - Nodes adjacent to node I are found in ADJ(J), where
*              J = XADJ(I), XADJ(I)+1, ..... , XADJ(I+1)-1
*     S      - List giving the distance of each node in this
*              component from the end node
*     Q      - List of nodes which are in this component
*            - Also used to store queue of active or preactive nodes
*     P      - Undefined
*
*     OUTPUT:
*     -------
*
*     N      - Unchanged
*     NC     - Unchanged
*     SNODE  - Unchanged
*     LSTNUM - Count of numbered nodes (input value incremented by NC)
*     E2     - Unchanged
*     ADJ    - Unchanged
*     XADJ   - Unchanged
*     S      - List of new node numbers 
*            - New number for node I is S(I)
*     Q      - Not used
*     P      - Not used
*
*     NOTES:
*     ------
*
*     S also serves as a list giving the status of the nodes
*     during the numbering process:
*     S(I) gt 0 indicates node I is postactive
*     S(I) =  0 indicates node I is active 
*     S(I) = -1 indicates node I is preactive
*     S(I) = -2 indicates node I is inactive
*     P is used to hold the priorities for each node
*
*     PROGRAMMER:             Scott Sloan
*     -----------
*
*     LAST MODIFIED:          1 March 1991    Scott Sloan
*     --------------
*
*     COPYRIGHT 1989:         Scott Sloan
*     ---------------         Department of Civil Engineering
*                             University of Newcastle
*                             NSW 2308
*
************************************************************************
      INTEGER I,J,N
      INTEGER E2,NC,NN,W1,W2
      INTEGER NBR
      INTEGER NEXT,NODE,PRTY
      INTEGER JSTOP,JSTRT,ISTOP,ISTRT,NABOR,SNODE
      INTEGER ADDRES,LSTNUM,MAXPRT
      INTEGER P(N),Q(NC),S(N)
      INTEGER ADJ(E2)
      INTEGER XADJ(N+1)
*
      PARAMETER (W1=1, W2=2)
*
*     Initialise priorities and status for each node in this component
*     Initial priority = W1*DIST - W2*DEGREE     where:
*     W1     = a positive weight
*     W2     = a positive weight
*     DEGREE = initial current degree for node
*     DIST   = distance of node from end node
*
      DO 10 I=1,NC
        NODE=Q(I)
        P(NODE)=W1*S(NODE)-W2*(XADJ(NODE+1)-XADJ(NODE)+1)
        S(NODE)=-2
   10 CONTINUE
*
*     Insert starting node in queue and assign it a preactive status
*     NN is the size of queue
*
      NN=1
      Q(NN)=SNODE
      S(SNODE)=-1
*
*     Loop while queue is not empty
*
   30 IF(NN.GT.0)THEN
*
*       Scan queue for node with max priority
*
        ADDRES=1
        MAXPRT=P(Q(1))
        DO 35 I=2,NN
          PRTY=P(Q(I))
          IF(PRTY.GT.MAXPRT)THEN
            ADDRES=I
            MAXPRT=PRTY
          END IF
   35   CONTINUE
*
*       NEXT is the node to be numbered next
*
        NEXT=Q(ADDRES)
*
*       Delete node NEXT from queue
*       
        Q(ADDRES)=Q(NN)
        NN=NN-1
        ISTRT=XADJ(NEXT)
        ISTOP=XADJ(NEXT+1)-1
        IF(S(NEXT).EQ.-1)THEN
*
*         Node NEXT is preactive, examine its neighbours
*
          DO 50 I=ISTRT,ISTOP
*
*           Decrease current degree of neighbour by -1
*
            NBR=ADJ(I)
            P(NBR)=P(NBR)+W2
*
*           Add neighbour to queue if it is inactive
*           assign it a preactive status
*
            IF(S(NBR).EQ.-2)THEN
              NN=NN+1
              Q(NN)=NBR
              S(NBR)=-1
            END IF
   50     CONTINUE
        END IF
*
*       Store new node number for node NEXT
*       Status for node NEXT is now postactive
*
        LSTNUM=LSTNUM+1
        S(NEXT)=LSTNUM
*
*       Search for preactive neighbours of node NEXT
*
        DO 80 I=ISTRT,ISTOP
          NBR=ADJ(I)
          IF(S(NBR).EQ.-1)THEN
*
*           Decrease current degree of preactive neighbour by -1
*           assign neighbour an active status
*
            P(NBR)=P(NBR)+W2
            S(NBR)=0
*
*           Loop over nodes adjacent to preactive neighbour
*
            JSTRT=XADJ(NBR)
            JSTOP=XADJ(NBR+1)-1
            DO 60 J=JSTRT,JSTOP
              NABOR=ADJ(J)
*
*             Decrease current degree of adjacent node by -1
*
              P(NABOR)=P(NABOR)+W2
              IF(S(NABOR).EQ.-2)THEN
*
*               Insert inactive node in queue with a preactive status
*
                NN=NN+1
                Q(NN)=NABOR
                S(NABOR)=-1
               END IF
   60       CONTINUE
          END IF
   80   CONTINUE
        GOTO 30
      END IF
      END
      SUBROUTINE PROFIL_SLOAN(N,NNN,E2,ADJ,XADJ,OLDPRO,NEWPRO)
************************************************************************
*
*     PURPOSE:
*     --------
*
*     Compute the profiles using both original and new node numbers
*
*     INPUT:
*     ------
*
*     N      - Total number of nodes in graph
*     NNN    - List of new node numbers for graph
*            - New node number for node I is given by NNN(I)
*     E2     - Twice the number of edges in the graph = XADJ(N+1)-1
*     ADJ    - Adjacency list for all nodes in graph
*            - List of length 2E where E is the number of edges in 
*              the graph and 2E = XADJ(N+1)-1
*     XADJ   - Index vector for ADJ
*            - Nodes adjacent to node I are found in ADJ(J), where
*              J = XADJ(I), XADJ(I)+1, ..., XADJ(I+1)-1
*            - Degree of node I given by XADJ(I+1)-XADJ(I)
*     OLDPRO - Undefined
*     NEWPRO - Undefined
*
*     OUTPUT:
*     -------
*
*     N      - Unchanged
*     NNN    - Unchanged
*     E2     - Unchanged
*     ADJ    - Unchanged
*     XADJ   - Unchanged
*     OLDPRO - Profile with original node numbering
*     NEWPRO - Profile with new node numbering
*
*     NOTE:      Profiles include diagonal terms
*     -----
*
*     PROGRAMMER:             Scott Sloan
*     -----------
*
*     LAST MODIFIED:          13 August 1991     Scott Sloan
*     --------------
*
*     COPYRIGHT 1989:         Scott Sloan
*     ---------------         Department of Civil Engineering
*                             University of Newcastle
*                             NSW 2308
*
***********************************************************************
      INTEGER I,J,N
      INTEGER E2
      INTEGER JSTOP,JSTRT
      INTEGER NEWMIN,NEWPRO,OLDMIN,OLDPRO
      INTEGER ADJ(E2),NNN(N)
      INTEGER XADJ(N+1)
*
*     Set profiles and loop over each node in graph
*
      OLDPRO=0
      NEWPRO=0
      DO 20 I=1,N
        JSTRT=XADJ(I)
        JSTOP=XADJ(I+1)-1
        OLDMIN=I
        NEWMIN=NNN(I)
*
*       Find lowest numbered neighbour of node I
*       (using both old and new node numbers)
*
        DO 10 J=JSTRT,JSTOP
          OLDMIN=MIN(OLDMIN,ADJ(J))
          NEWMIN=MIN(NEWMIN,NNN(ADJ(J)))
   10   CONTINUE
*
*       Update profiles
*
        OLDPRO=OLDPRO+DIM(I,OLDMIN)
        NEWPRO=NEWPRO+DIM(NNN(I),NEWMIN)
   20 CONTINUE
*
*     Add diagonal terms to profiles
*
      OLDPRO=OLDPRO+N
      NEWPRO=NEWPRO+N
      END
      SUBROUTINE ROOTLS_SLOAN(N,ROOT,MAXWID,E2,ADJ,XADJ,MASK,LS,XLS,
     +DEPTH,WIDTH)
************************************************************************
*
*     PURPOSE:
*     --------
*
*     Generate rooted level structure using a FORTRAN 77 implementation
*     of the algorithm given by George and Liu
*
*     INPUT:
*     ------
*
*     N      - Number of nodes
*     ROOT   - Root node for level structure
*     MAXWID - Max permissible width of rooted level structure
*            - Abort assembly of level structure if width is ge MAXWID
*            - Assembly ensured by setting MAXWID = N+1
*     E2     - Twice the number of edges in the graph = XADJ(N+1)-1
*     ADJ    - Adjacency list for all nodes in graph
*            - List of length 2E where E is the number of edges in 
*              the graph and 2E = XADJ(N+1)-1
*     XADJ   - Index vector for ADJ
*            - Nodes adjacent to node I are found in ADJ(J), where
*              J = XADJ(I), XADJ(I)+1, ..., XADJ(I+1)-1
*            - Degree of node I is XADJ(I+1)-XADJ(I)
*     MASK   - Masking vector for graph
*            - Visible nodes have MASK = 0
*     LS     - Undefined
*     XLS    - Undefined
*     DEPTH  - Undefined
*     WIDTH  - Undefined
*
*     OUTPUT:
*     -------
*
*     N      - Unchanged
*     ROOT   - Unchanged
*     MAXWID - unchanged
*     E2     - Unchanged
*     ADJ    - Unchanged
*     XADJ   - Unchanged
*     MASK   - Unchanged
*     LS     - List containing a rooted level structure
*            - List of length NC
*     XLS    - Index vector for LS
*            - Nodes in level I are found in LS(J), where
*              J = XLS(I), XLS(I)+1, ..., XLS(I+1)-1
*            - List of max length NC+1
*     DEPTH  - Number of levels in rooted level structure
*     WIDTH  - Width of rooted level structure
*
*     NOTE:  If WIDTH ge MAXWID then assembly has been aborted
*     -----
*
*     PROGRAMMER:             Scott Sloan
*     -----------
*
*     LAST MODIFIED:          1 March 1991      Scott Sloan
*     --------------
*
*     COPYRIGHT 1989:         Scott Sloan
*     ---------------         Department of Civil Engineering
*                             University of Newcastle
*                             NSW 2308
*
************************************************************************
      INTEGER I,J,N
      INTEGER E2,NC
      INTEGER NBR
      INTEGER NODE,ROOT
      INTEGER DEPTH,JSTOP,JSTRT,LSTOP,LSTRT,LWDTH,WIDTH
      INTEGER MAXWID
      INTEGER LS(N)
      INTEGER ADJ(E2),XLS(N+1)
      INTEGER MASK(N),XADJ(N+1)
*
*     Initialisation
*
      MASK(ROOT)=1
      LS(1)=ROOT
      NC   =1
      WIDTH=1
      DEPTH=0
      LSTOP=0
      LWDTH=1
   10 IF(LWDTH.GT.0)THEN
*
*       LWDTH is the width of the current level
*       LSTRT points to start of current level
*       LSTOP points to end of current level
*       NC counts the nodes in component
*
        LSTRT=LSTOP+1
        LSTOP=NC
        DEPTH=DEPTH+1
        XLS(DEPTH)=LSTRT
*
*       Generate next level by finding all visible neighbours
*       of nodes in current level
*
        DO 30 I=LSTRT,LSTOP
          NODE=LS(I)
          JSTRT=XADJ(NODE)
          JSTOP=XADJ(NODE+1)-1
          DO 20 J=JSTRT,JSTOP
            NBR=ADJ(J)
            IF(MASK(NBR).EQ.0)THEN
              NC=NC+1
              LS(NC)=NBR
              MASK(NBR)=1
            END IF
   20     CONTINUE
   30   CONTINUE
*
*       Compute width of level just assembled and the width of the
*       level structure so far
*
        LWDTH=NC-LSTOP
        WIDTH=MAX(LWDTH,WIDTH)
*
*       Abort assembly if level structure is too wide
*
        IF(WIDTH.GE.MAXWID)GOTO 35
        GOTO 10
      END IF
      XLS(DEPTH+1)=LSTOP+1
*
*     Reset MASK=0 for nodes in the level structure
*
   35 CONTINUE
      DO 40 I=1,NC
        MASK(LS(I))=0
   40 CONTINUE
      END
