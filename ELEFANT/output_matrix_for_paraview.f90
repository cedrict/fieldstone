!==================================================================================================!
!==================================================================================================!
!                                                                                                  !
! ELEFANT                                                                        C. Thieulot       !
!                                                                                                  !
!==================================================================================================!
!==================================================================================================!

subroutine output_matrix_for_paraview

use module_parameters, only: NfemP,NfemVel,GT_storage,K_storage,iproc,solve_stokes_system
use module_sparse, only: csrGT,csrK
use module_arrays, only: GT_matrix,K_matrix
use module_timing

implicit none

integer :: i,j,nz

!==================================================================================================!
!==================================================================================================!
!@@ \subsection{output\_matrix\_for\_paraview}
!@@
!==================================================================================================!

if (iproc==0) then

call system_clock(counti,count_rate)

!==============================================================================!

if (.not.solve_stokes_system) return

select case(GT_storage)

!------------------
case('matrix_FULL')

    nz=count(abs(GT_matrix)>1e-8)

    open(unit=123,file='OUTPUT/MATRIX/matrix_GT.vtu',status='replace',form='formatted')
    write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
    write(123,*) '<UnstructuredGrid>'
    write(123,*) '<Piece NumberOfPoints="',nz,'" NumberOfCells="',nz,'">'

    write(123,*) '<Points>'
    write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
    do i=1,NfemP
    do j=1,NfemVel
       if (abs(GT_matrix(i,j))>1d-8) write(123,*) i,j,0
    end do
    end do
    write(123,*) '</DataArray>'
    write(123,*) '</Points>'

    write(123,*) '<Cells>'
    write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
    do i=1,nz
    write(123,'(i8)') i-1
    end do
    write(123,*) '</DataArray>'
    write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
    do i=1,nz
    write(123,'(i8)') i
    end do
    write(123,*) '</DataArray>'
    write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
    do i=1,nz
    write(123,'(i1)') 1
    end do
    write(123,*) '</DataArray>'
    write(123,*) '</Cells>'

    write(123,*) '</Piece>'
    write(123,*) '</UnstructuredGrid>'
    write(123,*) '</VTKFile>'
    close(123)

!------------------
case('matrix_CSR')

    open(unit=123,file='OUTPUT/MATRIX/matrix_GT.vtu',status='replace',form='formatted')
    write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
    write(123,*) '<UnstructuredGrid>'
    write(123,*) '<Piece NumberOfPoints="',csrGT%nz,'" NumberOfCells="',csrGT%nz,'">'

    write(123,*) '<Points>'
    write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
    do i=1,csrGT%nr
    do j=csrGT%ia(i),csrGT%ia(i+1)-1
    write(123,*) csrGT%ja(j),i+NfemVel,0
    end do
    end do
    write(123,*) '</DataArray>'
    write(123,*) '</Points>'

    write(123,*) '<Cells>'

    write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
    do i=1,csrGT%nz
    write(123,'(i8)') i-1
    end do
    write(123,*) '</DataArray>'

    write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
    do i=1,csrGT%nz
    write(123,'(i8)') i
    end do
    write(123,*) '</DataArray>'

    write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
    do i=1,csrGT%nz
    write(123,'(i1)') 1
    end do
    write(123,*) '</DataArray>'

    write(123,*) '</Cells>'

    write(123,*) '</Piece>'
    write(123,*) '</UnstructuredGrid>'
    write(123,*) '</VTKFile>'
    close(123)

!-----------------
case('blocks_CSR')

    write(*,'(a)') shift//'cannot export GT matrix'

!-----------
case default

   stop 'output_matrix_for_paraview: unknown GT_storage value'

end select

!==============================================================================

select case(K_storage)

!------------------
case('matrix_FULL')

    nz=count(abs(K_matrix)>1e-8)

    open(unit=123,file='OUTPUT/MATRIX/matrix_K.vtu',status='replace',form='formatted')
    write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
    write(123,*) '<UnstructuredGrid>'
    write(123,*) '<Piece NumberOfPoints="',nz,'" NumberOfCells="',nz,'">'

    write(123,*) '<Points>'
    write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
    do i=1,NfemVel
    do j=1,NfemVel
       if (abs(K_matrix(i,j))>1d-8) write(123,*) i,j,0
    end do
    end do
    write(123,*) '</DataArray>'
    write(123,*) '</Points>'

    write(123,*) '<Cells>'
    write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
    do i=1,nz
    write(123,'(i8)') i-1
    end do
    write(123,*) '</DataArray>'
    write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
    do i=1,nz
    write(123,'(i8)') i
    end do
    write(123,*) '</DataArray>'
    write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
    do i=1,nz
    write(123,'(i1)') 1
    end do
    write(123,*) '</DataArray>'
    write(123,*) '</Cells>'

    write(123,*) '</Piece>'
    write(123,*) '</UnstructuredGrid>'
    write(123,*) '</VTKFile>'
    close(123)

!------------------
case('matrix_MUMPS')
    
    write(*,'(a)') shift//'cannot export K matrix'

!------------------
case('matrix_CSR')

    open(unit=123,file='OUTPUT/MATRIX/matrix_K.vtu',status='replace',form='formatted')
    write(123,*) '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="BigEndian">'
    write(123,*) '<UnstructuredGrid>'
    write(123,*) '<Piece NumberOfPoints="',csrK%nz,'" NumberOfCells="',csrK%nz,'">'

    write(123,*) '<Points>'
    write(123,*) '<DataArray type="Float32" NumberOfComponents="3" Format="ascii">'
    do i=1,csrK%n
    do j=csrK%ia(i),csrK%ia(i+1)-1
    write(123,*) i,csrK%ja(j),0
    end do
    end do
    write(123,*) '</DataArray>'
    write(123,*) '</Points>'


    write(123,*) '<Cells>'

    write(123,*) '<DataArray type="Int32" Name="connectivity" Format="ascii">'
    do i=1,csrK%nz
    write(123,'(i8)') i-1
    end do
    write(123,*) '</DataArray>'

    write(123,*) '<DataArray type="Int32" Name="offsets" Format="ascii">'
    do i=1,csrK%nz
    write(123,'(i8)') i
    end do
    write(123,*) '</DataArray>'

    write(123,*) '<DataArray type="Int32" Name="types" Format="ascii">'
    do i=1,csrK%nz
    write(123,'(i1)') 1
    end do
    write(123,*) '</DataArray>'

    write(123,*) '</Cells>'

    write(123,*) '</Piece>'
    write(123,*) '</UnstructuredGrid>'
    write(123,*) '</VTKFile>'
    close(123)

!------------------
case('blocks_MUMPS')

    write(*,'(a)') shift//'cannot export K matrix'

!------------------
case('blocks_CSR')

!-----------
case default

   stop 'output_matrix_for_paraview: unknown K_storage value'

end select













!==============================================================================!

call system_clock(countf) ; elapsed=dble(countf-counti)/dble(count_rate)

write(*,'(a,f6.2,a)') 'output_matrix_for_paraview:',elapsed,' s'

end if ! iproc

end subroutine

!==================================================================================================!
!==================================================================================================!
