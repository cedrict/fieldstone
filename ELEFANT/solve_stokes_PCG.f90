!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine solve_stokes_PCG

!use module_parameters
!use module_mesh 
!use module_constants
!use module_swarm
!use module_materials
!use module_arrays
use module_timing

implicit none


!==================================================================================================!
!==================================================================================================!
!@@ \subsection{solve\_stokes\_PCG}
!@@ This outer iteration scheme is the Preconditioned Conjugate Gradient.
!@@ Let $\tilde{\bm P}_0 \in \mathbb{R}^n$ 
!@@ and $\tilde{\K}\cdot \tilde{\bm V}_1 = \tilde{f} - \tilde{\G} \cdot\tilde{\bm P}_0$.
!@@ Set ${\bm d}_1=-{\bm q}_1=\tilde{\G}^T \cdot {\bm V}_1 - \tilde{\bm h}$.
!@@ For $k=1,2,...$ find 
!@@ \begin{eqnarray}
!@@ {\bm \phi}_k &=& \tilde{\G} \cdot {\bm d}_k \\
!@@ {\bm t}_k    &=& \tilde{\K}^{-1} \cdot {\bm \phi}_k \label{eqinner}\\
!@@ \alpha_k &=& \frac{{\bm q}_k' \cdot {\bm q}_k}{{\bm \phi}_k' \cdot {\bm t}_k + {\bm d}_k \cdot \tilde{\C} \cdot {\bm d}_k} \\
!@@ \tilde{\bm P}_k &=& \tilde{\bm P}_{k-1} + \alpha_k {\bm d}_k \\
!@@ \tilde{\bm V}_{k+1} &=& \tilde{\bm V}_k -\alpha_k {\bm t}_k \\
!@@ {\bm q}_{k+1} &=& \tilde{\bm h} -(\tilde{\G}^T \cdot \tilde{\bm V}_{k+1} -\tilde{\C}\cdot \tilde{\bm P}_k )\\
!@@ \beta_k &=& \frac{{\bm q}_{k+1}' \cdot {\bm q}_{k+1}}{{\bm q}_k' \cdot {\bm q}_k } \\
!@@ {\bm d}_{k+1} &=& -{\bm q}_{k+1} + \beta_k {\bm d}_k
!@@ \end{eqnarray}
!@@ Refs: \textcite{elsw}, p82, Algorithm 2.2
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!









if (debug) then
write(2345,*) limit//'name'//limit
end if

!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'solve_stokes_PCG:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
