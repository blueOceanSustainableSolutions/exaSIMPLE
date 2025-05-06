
	
!***********************************************************************
!>
!!     User-defined module for possible routines common to other 
!!     user-defined modules
!!
!!     Maritime Research Institute of Netherlands (MARIN)
!!
!!     ReFRESCO
!!
!!     $Id: common_user_code.F90.xml 10382 2020-02-11 16:15:27Z JWindt@MARIN.LOCAL $
!!
!!     Maintainer: Douwe Rijpkema
!!     Level:      Infrastructure/User-defined-routines
!!
!!     (c) 2005-2018 MARIN
!!     Proprietary  data. Unauthorized use, distribution,
!!     or duplication is prohibited. All rights reserved.
!!
!<
!***********************************************************************
MODULE common_user_code

USE boundaries_controls
USE main_controls
USE equations 
USE equations_controls 
USE fielddata
USE geometry
USE logging
USE math
USE parameters
USE topology
USE tracing

! this macro will silence the compiler warning #7712 unused variables
#define refresco_not_used(x) IF(.FALSE.) x=x
    

	
USE parallel
USE pressure
          

	
IMPLICIT NONE

PRIVATE

INTEGER,        PRIVATE :: nAllCells,nIntCells,NBndCells,nIntFaces,nBndFaces,nfamilies
    

	
LOGICAL,  POINTER, PRIVATE :: useReferencePressureLocation
INTEGER,  POINTER, PRIVATE :: pref_process,pref_cell
REAL(dp), POINTER, PRIVATE :: cent_c_3(:,:)
REAL(dp), POINTER, PRIVATE :: exact_v_c_3(:,:), exact_p_c(:)
REAL(dp), POINTER, PRIVATE :: exact_grad_v_c_33(:,:,:), exact_grad_p_c_3(:,:)
REAL(dp), POINTER, PRIVATE :: exact_hessian_p_c_33(:,:,:)
          

	
PUBLIC common_user_code_initial,&
       common_user_code_init0,&
       common_user_code_init3,&
       common_user_code_init4,&
       common_user_code_init5,&
       common_user_code_adapt_reallocate,&
       common_user_code_restart,&
       common_user_code_timestep,&
       common_user_code_outit,&
       common_user_code_final,&
       common_user_code_exit, &
       common_user_code_read_controls

CONTAINS
    

	
		
!! Use this routine for coding at the earliest stage of the computation 
!======================================================================
   SUBROUTINE common_user_code_init0
!======================================================================
      

		

		
  CALL tracing_trace_begin(trace_common_user_code_init0)
      

		

		
  CALL tracing_trace_end(trace_common_user_code_init0)

END SUBROUTINE common_user_code_init0
      
	

	
		
!======================================================================
   SUBROUTINE common_user_code_read_controls
!=====================================================================

      

		

		
  CALL tracing_trace_begin(trace_common_user_code_read_controls)
      

		
                  

		
  CALL tracing_trace_end(trace_common_user_code_read_controls)

END SUBROUTINE common_user_code_read_controls
      
	
       

	
		
!> Use this to set initial values. To resolve dependencies between 
!! several *_initial modules this is called several times. Thus 
!< do not put ALLOCATE statements here. 
!======================================================================
   SUBROUTINE common_user_code_initial
!=====================================================================
      

		
INTEGER :: icell
REAL(DP) :: x,y,z         
            

		
  CALL tracing_trace_begin(trace_common_user_code_initial)
      

		
! the exact solution and its gradient
DO icell=1,nAllCells

  x = cent_c_3(icell,1)
  y = cent_c_3(icell,2)
  z = cent_c_3(icell,3)

  exact_v_c_3(icell,1) = -cos(PI*x)*sin(PI*y)
  exact_grad_v_c_33(icell,1,1) = PI*sin(x*PI)*sin(y*PI)
  exact_grad_v_c_33(icell,1,2) = -PI*cos(x*PI)*cos(y*PI)
  exact_grad_v_c_33(icell,1,3) = 0.0D0

  exact_v_c_3(icell,2) =  sin(PI*x)*cos(PI*y)
  exact_grad_v_c_33(icell,2,1) = PI*cos(x*PI)*cos(y*PI)
  exact_grad_v_c_33(icell,2,2) = -PI*sin(x*PI)*sin(y*PI)
  exact_grad_v_c_33(icell,2,3) = 0.0D0

  exact_v_c_3(icell,3) = 0.0d0
  exact_grad_v_c_33(icell,3,1) = 0.0D0
  exact_grad_v_c_33(icell,3,2) = 0.0D0
  exact_grad_v_c_33(icell,3,3) = 0.0D0

  exact_p_c(icell) = (-cos(2*y*PI)-cos(2*x*PI))/4.0E+0
  exact_grad_p_c_3(icell,1) = PI*sin(2*x*PI)/2.0E+0
  exact_grad_p_c_3(icell,2) = PI*sin(2*y*PI)/2.0E+0
  exact_grad_p_c_3(icell,3) = 0.0D0

  exact_hessian_p_c_33(icell,1,1) = PI*PI*cos(2*x*PI)
  exact_hessian_p_c_33(icell,1,2) = 0.0D0
  exact_hessian_p_c_33(icell,1,3) = 0.0D0

  exact_hessian_p_c_33(icell,2,1) = 0.0D0
  exact_hessian_p_c_33(icell,2,2) = PI*PI*cos(2*y*PI)
  exact_hessian_p_c_33(icell,2,3) = 0.0D0

  exact_hessian_p_c_33(icell,3,1) = 0.0D0
  exact_hessian_p_c_33(icell,3,2) = 0.0D0
  exact_hessian_p_c_33(icell,3,3) = 0.0D0
END DO

! ref pressure must match exact solution
IF (useReferencePressureLocation.AND.thisprocess==pref_process) &
  controls%equations%equation(equation_loc_pressure)%EQPressure%pressureReference=exact_p_c(pref_cell)
            

		
  CALL tracing_trace_end(trace_common_user_code_initial)

END SUBROUTINE common_user_code_initial
      
	

	
		
!! Use this routine to set up linked lists, to get sizes or to allocate fields 
!======================================================================
   SUBROUTINE common_user_code_init3
!======================================================================
      

		

		
  CALL tracing_trace_begin(trace_common_user_code_init3)
      

		
CALL geometry_get_pointer(cent_c_3 = cent_c_3)
CALL fielddata_register_field("ExactVelocity",      &
                              ndim = 3,             &
                              toSave=.TRUE.,        &
                              hasField =.TRUE.,     &
                              hasBoundary =.TRUE.,  &
                              hasGradients=.TRUE.,  &
                              toSaveGradients=.TRUE.)
CALL fielddata_register_field("ExactPressure",      &
                              ndim=1,               &
                              toSave=.TRUE.,        &
                              hasField =.TRUE.,     &
                              hasBoundary =.TRUE.,  &
                              hasGradients=.TRUE.,  &
                              hasHessian=.TRUE.,    &
                              toSaveHessian=.TRUE., &
                              toSaveGradients=.TRUE.)
            

		
  CALL tracing_trace_end(trace_common_user_code_init3)

END SUBROUTINE common_user_code_init3
      
	

	
		
!> Use *_init4 module for fielddata_init4,
!< equation_system_register_equation_system ... statements 
!======================================================================
   SUBROUTINE common_user_code_init4
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_common_user_code_init4)
      

		

		
  CALL tracing_trace_end(trace_common_user_code_init4)
   
END SUBROUTINE common_user_code_init4
    
	

	
		
!! Use *_init5 module for all *_get_pointer statements 
!======================================================================
   SUBROUTINE common_user_code_init5
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_common_user_code_init5)
   
    CALL topology_get_size( nAllCells = nAllCells, &
                            nIntCells = nIntCells, &
                            NBndCells = NBndCells,&
                            nIntFaces = nIntFaces,&
                            nBndFaces = nBndFaces,&
                            nfamilies = nfamilies)
      

		
CALL fielddata_get_pointer("ExactVelocity", &
                           Field_3=exact_v_c_3, &
                           Grad_33=exact_grad_v_c_33)
CALL fielddata_get_pointer("ExactPressure", &
                           Field=exact_p_c, &
                           Grad_3=exact_grad_p_c_3, &
                           Hessian_33=exact_hessian_p_c_33) 
CALL pressure_get_pointer(theUseReferencePressureLocation=useReferencePressureLocation, &
                          thePref_process=pref_process, &
                          thePref_cell=pref_cell)
            

		
  CALL tracing_trace_end(trace_common_user_code_init5)

END SUBROUTINE common_user_code_init5
      
	

	
		
!! Reallocates variables. Remark: registered variables are reallocated 
!!  automatically.
!======================================================================
   SUBROUTINE common_user_code_adapt_reallocate
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_common_user_code_adapt_reallocate)

! example
!CALL reall_resize(data_f,1,nIntFaces+nBndFaces,INITVALUE_DP)

 
      

		

		
  CALL tracing_trace_end(trace_common_user_code_adapt_reallocate)

END SUBROUTINE common_user_code_adapt_reallocate
      
	


    
!! Use *_restart routines for restart 
!======================================================================
   SUBROUTINE common_user_code_restart
!=====================================================================
      

    

    
  CALL tracing_trace_begin(trace_common_user_code_restart)
   
      

    

    
  CALL tracing_trace_end(trace_common_user_code_restart)

END SUBROUTINE common_user_code_restart
      
  


	
		
!! This routine is only executed when called from other user coding
!======================================================================
   SUBROUTINE common_user_code_timestep(timestep,simultime)
!======================================================================

INTEGER   :: timestep
REAL(dp)  :: simultime
      

		

		    
  refresco_not_used(timestep)  ! remove this when you use 'timestep'
  refresco_not_used(simultime) ! remove this when you use 'simultime'
  CALL tracing_trace_begin(trace_common_user_code_timestep)
      

		

		
  CALL tracing_trace_end(trace_common_user_code_timestep)

END SUBROUTINE common_user_code_timestep
      
	

	
		
!! This routine is only executed when called from other user coding
!======================================================================
   SUBROUTINE common_user_code_outit(out_iter)
!======================================================================

INTEGER :: out_iter
      

		

		
  refresco_not_used(out_iter)  ! remove this when you use 'out_iter'
  CALL tracing_trace_begin(trace_common_user_code_outit)
      

		

		
  CALL tracing_trace_end(trace_common_user_code_outit)

END SUBROUTINE common_user_code_outit
      
	

	
		
!> This routine is executed only once after the computation has stoped
!! due to max no of iteration, truncation error or stopfile. It is not 
!< executed when killfile is used
!======================================================================
   SUBROUTINE common_user_code_final
!======================================================================
      

		

		
  CALL tracing_trace_begin(trace_common_user_code_final)
      

		

		
  CALL tracing_trace_end(trace_common_user_code_final)

END SUBROUTINE common_user_code_final
      
	

	
		
!! Use *_exit module for DEALLOCATE statements etc. 
!======================================================================
   SUBROUTINE common_user_code_exit
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_common_user_code_exit)
      

		

		
  CALL tracing_trace_end(trace_common_user_code_exit)

END SUBROUTINE common_user_code_exit
      
	

	

	
END MODULE common_user_code
    
