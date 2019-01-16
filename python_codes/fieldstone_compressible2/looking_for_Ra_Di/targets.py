import numpy as np

alpha=3e-5
g=10
hcapa=1250
hcond=3

DTmin=1000  ; DTmax=4000  ; DTnpts=301
Lmin=1e6    ; Lmax=3e6    ; Lnpts=301
rhomin=3000 ; rhomax=4000 ; rhonpts=101
etamin=19   ; etamax=25   ; etanpts=120
  
Ratarget=1e4
Ditarget=0.25
 
datafile=open("RaDi","w")

for i in range(0,DTnpts):
    DT=DTmin+(DTmax-DTmin)/(DTnpts-1)*i

    for j in range(0,Lnpts):
        L=Lmin+(Lmax-Lmin)/(Lnpts-1)*j

        for k in range(0,rhonpts):
            rho=rhomin+(rhomax-rhomin)/(rhonpts-1)*k

            for l in range(0,etanpts):
                eta=etamin+(etamax-etamin)/(etanpts-1)*l
                eta=10**eta

                Ra=alpha*g*DT*L**3*rho**2*hcapa/eta/hcond
                Di=alpha*g*L/hcapa

                if abs(Ra-Ratarget)/Ratarget<0.01 and abs(Di-Ditarget)/Ditarget<0.01:
                   datafile.write("%10e %10e %10e %10e %10e %10e  \n" %(Ra,Di,DT,L,rho,eta))


