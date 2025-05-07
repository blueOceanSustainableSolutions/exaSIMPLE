
	
!***********************************************************************
!>
!!     User-defined module for setting up source terms of any phi equation
!!
!!     Maritime Research Institute of Netherlands (MARIN)
!!
!!     ReFRESCO
!!
!!     $Id: set_source.F90.xml 11737 2020-11-06 12:18:31Z MDeij@MARIN.LOCAL $
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
MODULE set_source

USE boundaries_controls
! Only use this if you have a common_user_code module
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

    

	

	
IMPLICIT NONE

PRIVATE

INTEGER,        PRIVATE :: nAllCells,nIntCells,NBndCells,nIntFaces,nBndFaces,nfamilies
    

	

	
PUBLIC set_source_initial,&
       set_source_init3,&
       set_source_init4,&
       set_source_init5,&
       set_source_restart,&
       set_source_timestep,&
       set_source_outit,&
       set_source_final,&
       set_source_exit, &
       set_source_read_controls

CONTAINS
    

	
		
!======================================================================
   SUBROUTINE set_source_read_controls
!=====================================================================

      

		

		
  CALL tracing_trace_begin(trace_set_source_read_controls)
      

		

		
  CALL tracing_trace_end(trace_set_source_read_controls)

END SUBROUTINE set_source_read_controls
      
	


	
		
!> Use this to set initial values. To resolve dependencies between 
!! several *_initial modules this is called several times. Thus 
!< do not put ALLOCATE statements here. 
!======================================================================
   SUBROUTINE set_source_initial
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_source_initial)
      

		

		
  CALL tracing_trace_end(trace_set_source_initial)

END SUBROUTINE set_source_initial
      
	

	
		
!! Use this routine to set up linked lists, to get sizes or to allocate fields 
!======================================================================
   SUBROUTINE set_source_init3
!======================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_source_init3)
      

		

		
  CALL tracing_trace_end(trace_set_source_init3)

END SUBROUTINE set_source_init3
      
	

	
		
!> Use *_init4 module for fielddata_init4,
!< equation_system_register_equation_system ... statements 
!======================================================================
   SUBROUTINE set_source_init4
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_source_init4)
      

		

		
  CALL tracing_trace_end(trace_set_source_init4)
   
END SUBROUTINE set_source_init4
    
	

	
		
! Use *_init5 module for all *_get_pointer statements 
!======================================================================
   SUBROUTINE set_source_init5
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_source_init5)
   
    CALL topology_get_size( nAllCells = nAllCells, &
                            nIntCells = nIntCells, &
                            NBndCells = NBndCells,&
                            nIntFaces = nIntFaces,&
                            nBndFaces = nBndFaces,&
                            nfamilies = nfamilies)
      

		

		
  CALL tracing_trace_end(trace_set_source_init5)

END SUBROUTINE set_source_init5
      
	
	
  
    
!! Use *_restart routines for restart
!======================================================================
   SUBROUTINE set_source_restart
!======================================================================
      

    

    
  CALL tracing_trace_begin(trace_set_source_restart)
      

    

    
  CALL tracing_trace_end(trace_set_source_restart)

END SUBROUTINE set_source_restart
      
  
	
		
!! This routine is executed at each timestep
!======================================================================
   SUBROUTINE set_source_timestep(timestep,simultime)
!======================================================================

INTEGER   :: timestep
REAL(dp)  :: simultime
      

		

		
  refresco_not_used(timestep)  ! remove this when you use 'timestep'
  refresco_not_used(simultime) ! remove this when you use 'simultime'
  CALL tracing_trace_begin(trace_set_source_timestep)
      

		

		
  CALL tracing_trace_end(trace_set_source_timestep)

END SUBROUTINE set_source_timestep
      
	

	
		
!! This routine is executed at each outer iteration
!======================================================================
   SUBROUTINE set_source_outit(out_iter)
!======================================================================

INTEGER :: out_iter
      

		

		
  refresco_not_used(out_iter)  ! remove this when you use 'out_iter'
  CALL tracing_trace_begin(trace_set_source_outit)
      

		

		
  CALL tracing_trace_end(trace_set_source_outit)

END SUBROUTINE set_source_outit
      
	

	
		
!> This routine is executed only once after the computation has stoped
!! due to max no of iteration, truncation error or stopfile. It is not 
!< executed when killfile is used
!======================================================================
   SUBROUTINE set_source_final
!======================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_source_final)
      

		

		
  CALL tracing_trace_end(trace_set_source_final)

END SUBROUTINE set_source_final
      
	

	
		
!! Use *_exit module for DEALLOCATE statements etc. 
!======================================================================
   SUBROUTINE set_source_exit
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_source_exit)
      

		

		
  CALL tracing_trace_end(trace_set_source_exit)

END SUBROUTINE set_source_exit
      
	

	

	
END MODULE set_source
    
