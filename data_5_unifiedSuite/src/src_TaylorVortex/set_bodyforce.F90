	
!***********************************************************************
!>
!!     User-defined module for extra momentum body-forces
!!
!!     Maritime Research Institute of Netherlands (MARIN)
!!
!!     ReFRESCO
!!
!!     $Id: set_bodyforce.F90.xml 11737 2020-11-06 12:18:31Z MDeij@MARIN.LOCAL $
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
MODULE set_bodyforce

USE boundaries_controls
! only use this if you have a common_user_code module
! USE common_user_code 
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
    
USE materials_controls
          	
IMPLICIT NONE

PRIVATE

INTEGER,        PRIVATE :: nAllCells,nIntCells,NBndCells,nIntFaces,nBndFaces,nfamilies
INTEGER, PRIVATE :: icell,bgMaterialLoc
LOGICAL, PRIVATE :: hasSources   
REAL(dp), POINTER, PRIVATE :: bodyforce_c_3(:,:), cent_c_3(:,:)
          	
PUBLIC set_bodyforce_initial,&
       set_bodyforce_init3,&
       set_bodyforce_init4,&
       set_bodyforce_init5,&
       set_bodyforce_restart,&
       set_bodyforce_timestep,&
       set_bodyforce_outit,&
       set_bodyforce_final,&
       set_bodyforce_exit, &
       set_bodyforce_read_controls

CONTAINS
    		
!======================================================================
   SUBROUTINE set_bodyforce_read_controls
!=====================================================================
	
  CALL tracing_trace_begin(trace_set_bodyforce_read_controls)
     		
  CALL tracing_trace_end(trace_set_bodyforce_read_controls)

END SUBROUTINE set_bodyforce_read_controls
      		
!> Use this to set initial values. To resolve dependencies between 
!! several *_initial modules this is called several times. Thus 
!< do not put ALLOCATE statements here. 
!======================================================================
   SUBROUTINE set_bodyforce_initial
!=====================================================================
                 
REAL(DP) :: rho,mu,x,y
          	
  CALL tracing_trace_begin(trace_set_bodyforce_initial)
        
bgMaterialLoc = materials_controls_get_element_by_name(controls%general%material)
rho=controls%materials%material(bgMaterialLoc)%fluid%density
mu=controls%materials%material(bgMaterialLoc)%fluid%viscosityMolecular

DO icell = 1, nIntCells

   x = cent_c_3(icell,1)
   y = cent_c_3(icell,2)

   bodyforce_c_3(icell,1) = -rho*PI*cos(x*PI)*sin(x*PI)-2.d0*mu*PI**2*cos(x*PI)*sin(y*PI)+PI*sin(2.d0*x*PI)/2.0d0
   bodyforce_c_3(icell,2) = PI*sin(2.d0*y*PI)/2.0d0-rho*PI*cos(y*PI)*sin(y*PI)+2.d0*mu*PI**2*sin(x*PI)*cos(y*PI)

END DO
          	
  CALL tracing_trace_end(trace_set_bodyforce_initial)

END SUBROUTINE set_bodyforce_initial
      		
!! Use this routine to set up linked lists, to get sizes or to allocate fields 
!======================================================================
   SUBROUTINE set_bodyforce_init3
!======================================================================
     		
  CALL tracing_trace_begin(trace_set_bodyforce_init3)
     	
  CALL tracing_trace_end(trace_set_bodyforce_init3)

END SUBROUTINE set_bodyforce_init3
      		
!> Use *_init4 module for fielddata_init4,
!< equation_system_register_equation_system ... statements 
!======================================================================
   SUBROUTINE set_bodyforce_init4
!=====================================================================
      	
  CALL tracing_trace_begin(trace_set_bodyforce_init4)
      	
  CALL tracing_trace_end(trace_set_bodyforce_init4)
   
END SUBROUTINE set_bodyforce_init4
    	
!! Use *_init5 module for all *_get_pointer statements 
!======================================================================
   SUBROUTINE set_bodyforce_init5
!=====================================================================
      	
  CALL tracing_trace_begin(trace_set_bodyforce_init5)
   
  CALL topology_get_size( nAllCells = nAllCells, &
                          nIntCells = nIntCells, &
                          NBndCells = NBndCells,&
                          nIntFaces = nIntFaces,&
                          nBndFaces = nBndFaces,&
                          nfamilies = nfamilies)
      
IF (controls%bodyForces%UserDefined) THEN
   CALL fielddata_get_pointer("BodyForce",Field_3=bodyforce_c_3)
END IF
CALL geometry_get_pointer(cent_c_3 = cent_c_3)            
		
  CALL tracing_trace_end(trace_set_bodyforce_init5)

END SUBROUTINE set_bodyforce_init5
          
!! Use *_restart routines for restart
!======================================================================
   SUBROUTINE set_bodyforce_restart
!======================================================================
     
   CALL tracing_trace_begin(trace_set_bodyforce_restart)
     
   CALL tracing_trace_end(trace_set_bodyforce_restart)

END SUBROUTINE set_bodyforce_restart
      
  	
!! This routine is executed at each timestep
!======================================================================
   SUBROUTINE set_bodyforce_timestep(timestep,simultime)
!======================================================================

INTEGER   :: timestep
REAL(dp)  :: simultime
      		
  refresco_not_used(timestep)  ! remove this when you use 'timestep'
  refresco_not_used(simultime) ! remove this when you use 'simultime'
  CALL tracing_trace_begin(trace_set_bodyforce_timestep)
      	
  CALL tracing_trace_end(trace_set_bodyforce_timestep)

END SUBROUTINE set_bodyforce_timestep
      	
!! This routine is executed at each outer iteration
!======================================================================
   SUBROUTINE set_bodyforce_outit(out_iter)
!======================================================================

INTEGER :: out_iter
     	
  refresco_not_used(out_iter)  ! remove this when you use 'out_iter'
  CALL tracing_trace_begin(trace_set_bodyforce_outit)
      	
  CALL tracing_trace_end(trace_set_bodyforce_outit)

END SUBROUTINE set_bodyforce_outit
      	
!> This routine is executed only once after the computation has stoped
!! due to max no of iteration, truncation error or stopfile. It is not 
!< executed when killfile is used
!======================================================================
   SUBROUTINE set_bodyforce_final
!======================================================================
     	
  CALL tracing_trace_begin(trace_set_bodyforce_final)
     	
  CALL tracing_trace_end(trace_set_bodyforce_final)

END SUBROUTINE set_bodyforce_final
      	
!! Use *_exit module for DEALLOCATE statements etc. 
!======================================================================
   SUBROUTINE set_bodyforce_exit
!=====================================================================
    	
  CALL tracing_trace_begin(trace_set_bodyforce_exit)
      	
  CALL tracing_trace_end(trace_set_bodyforce_exit)

END SUBROUTINE set_bodyforce_exit
      
END MODULE set_bodyforce
    
