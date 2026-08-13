
      NfemT=NV
      dt=1
      alphaT=0.5
      diffcoeff=250
      ndofT=1

      A_mat = np.zeros((NfemT,NfemT),dtype=np.float64) # FE matrix 
      rhs_xx = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
      rhs_yy = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
      rhs_xy = np.zeros(NfemT,dtype=np.float64)         # FE rhs 
      B_mat=np.zeros((2,ndofT*mV),dtype=np.float64)     # gradient matrix B 
      N_mat = np.zeros((mV,1),dtype=np.float64)         # shape functions

      counterq=0
      for iel in range (0,nel):

          exx_prev=np.zeros(mV,dtype=np.float64)
          eyy_prev=np.zeros(mV,dtype=np.float64)
          exy_prev=np.zeros(mV,dtype=np.float64)
          b_el_xx=np.zeros(mV*ndofT,dtype=np.float64)
          b_el_yy=np.zeros(mV*ndofT,dtype=np.float64)
          b_el_xy=np.zeros(mV*ndofT,dtype=np.float64)
          a_el=np.zeros((mV*ndofT,mV*ndofT),dtype=np.float64)
          Kd=np.zeros((mV,mV),dtype=np.float64)   # elemental diffusion matrix 
          MM=np.zeros((mV,mV),dtype=np.float64)   # elemental mass matrix 

          for k in range(0,mV):
               exx_prev[k]=exxn[icon_V[k,iel]]
               eyy_prev[k]=eyyn[icon_V[k,iel]]
               exy_prev[k]=exyn[icon_V[k,iel]]
          #end for

          for iq in [0,1,2]:
              for jq in [0,1,2]:

                  # position & weight of quad. point
                  rq=qcoords[iq]
                  sq=qcoords[jq]
                  weightq=qweights[iq]*qweights[jq]

                  NNNV[0:mV]=NNV(rq,sq)
                  dNNNVdr[0:mV]=dNNVdr(rq,sq)
                  dNNNVds[0:mV]=dNNVds(rq,sq)
                  N_mat[0:mV,0]=NNV(rq,sq)

                  #only valid for rectangular elements!
                  jcbi=np.zeros((ndim,ndim),dtype=np.float64)
                  jcob=hx*hy/4
                  jcbi[0,0] = 2/hx 
                  jcbi[1,1] = 2/hy
 
                  # compute dNdx & dNdy
                  for k in range(0,mV):
                      dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                      dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                      B_mat[0,k]=dNNNVdx[k]
                      B_mat[1,k]=dNNNVdy[k]
                  #end for

                  # compute mass matrix
                  MM=N_mat.dot(N_mat.T)*weightq*jcob

                  # compute diffusion matrix
                  Kd=B_mat.T.dot(B_mat)*diffcoeff*weightq*jcob

                  a_el+=MM+alphaT*Kd*dt
                  b_el_xx+=(MM-(1-alphaT)*Kd*dt).dot(exx_prev)
                  b_el_yy+=(MM-(1-alphaT)*Kd*dt).dot(eyy_prev)
                  b_el_xy+=(MM-(1-alphaT)*Kd*dt).dot(exy_prev)

                  counterq+=1
              #end for jq
          #end for iq

          # assemble matrix A_mat and right hand side rhs
          for k1 in range(0,mV):
              m1=icon_V[k1,iel]
              for k2 in range(0,mV):
                  m2=icon_V[k2,iel]
                  A_mat[m1,m2]+=a_el[k1,k2]
              #end for
              rhs_xx[m1]+=b_el_xx[k1]
              rhs_yy[m1]+=b_el_yy[k1]
              rhs_xy[m1]+=b_el_xy[k1]
          #end for

      #end for iel
    
      print("     -> matrix (m,M) %.4e %.4e " %(np.min(A_mat),np.max(A_mat)))
      print("     -> rhs (m,M) %.4e %.4e " %(np.min(rhs),np.max(rhs)))

      Txx = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xx)
      Tyy = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_yy)
      Txy = sps.linalg.spsolve(sps.csr_matrix(A_mat),rhs_xy)

      if use_srn_diff:
         exxn[:]=Txx[:]
         eyyn[:]=Tyy[:]
         exyn[:]=Txy[:]

      srn[:]=np.sqrt(0.5*(exxn[:]*exxn[:]+eyyn[:]*eyyn[:])+exyn[:]*exyn[:])

      print("     -> exxn (m,M) %.6e %.6e " %(np.min(exxn),np.max(exxn)))
      print("     -> eyyn (m,M) %.6e %.6e " %(np.min(eyyn),np.max(eyyn)))
      print("     -> exyn (m,M) %.6e %.6e " %(np.min(exyn),np.max(exyn)))
      print("     -> srn  (m,M) %.6e %.6e " %(np.min(srn),np.max(srn)))

      print("strain rate diffusion time: %.3f s" % (clock.time()-start))

   #end if use_srn_diff
