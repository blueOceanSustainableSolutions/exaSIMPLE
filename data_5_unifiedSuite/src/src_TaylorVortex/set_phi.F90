
	
!***********************************************************************
!>
!!     User-defined module for setting up initial and boundary conditions
!!
!!     Maritime Research Institute of Netherlands (MARIN)
!!
!!     ReFRESCO
!!
!!     $Id: set_phi.F90.xml 11737 2020-11-06 12:18:31Z MDeij@MARIN.LOCAL $
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
MODULE set_phi

USE boundaries_controls
USE main_controls
USE equations 
USE equations_controls 
USE fielddata
USE fielddata_tools
USE geometry
USE logging
USE math
USE parameters
USE topology
USE tracing
! Only use this if you have a common_user_code module
! USE common_user_code

! this macro will silence the compiler warning #7712 unused variables
#define refresco_not_used(x) IF(.FALSE.) x=x

    

	
          
USE counters
          
        

	
IMPLICIT NONE

PRIVATE

INTEGER,        PRIVATE :: nAllCells,nIntCells,NBndCells,nIntFaces,nBndFaces,nfamilies
    

	
          
INTEGER,  POINTER, PRIVATE :: family_f(:)
INTEGER,  POINTER, PRIVATE :: cell1_f(:), cell2_f(:)
REAL(DP), POINTER, PRIVATE :: cent_c_3(:,:), cent_f_3(:,:)
REAL(DP), POINTER, PRIVATE :: v_c_3(:,:), p_c(:)
REAL(DP), POINTER, PRIVATE :: exact_v_c_3(:,:)
          
        

	
PUBLIC set_phi_initial,&
       set_phi_init3,&
       set_phi_init4,&
       set_phi_init5,&
       set_phi_restart,&
       set_phi_timestep,&
       set_phi_outit,&
       set_phi_final,&
       set_phi_exit

CONTAINS
    

	
		
!> Use this to set initial values. To resolve dependencies between 
!! several *_initial modules this is called several times. Thus 
!< do not put ALLOCATE statements here. 
!======================================================================
   SUBROUTINE set_phi_initial
!=====================================================================
      

		
            
INTEGER  :: ifamily, xmin_loc, xmax_loc, ymin_loc, ymax_loc
INTEGER  :: iface, icell
          
          

		
  CALL tracing_trace_begin(trace_set_phi_initial)
      

		
            

! initial condition
DO icell=1,nIntCells

  v_c_3(icell,1)=0.0d0  
  v_c_3(icell,2)=0.0d0  
  v_c_3(icell,3)=0.0d0
  p_c(icell)    =0.0d0  

END DO

! boundary conditions
DO ifamily =1, nfamilies
  IF (TRIM('x_neg')==TRIM(controls%boundaries%family_fam(ifamily)%name)) THEN
    xmin_loc = ifamily
  ELSE IF (TRIM('x_pos')==TRIM(controls%boundaries%family_fam(ifamily)%name)) THEN
    xmax_loc = ifamily
  ELSE IF (TRIM('y_neg')==TRIM(controls%boundaries%family_fam(ifamily)%name)) THEN
    ymin_loc = ifamily
  ELSE IF (TRIM('y_pos')==TRIM(controls%boundaries%family_fam(ifamily)%name)) THEN
    ymax_loc = ifamily
  END IF
END DO

! velocity on boundaries
DO iface = nIntFaces+1,nIntFaces+nBndFaces 
  IF (family_f(iface)==xmin_loc .OR. &
      family_f(iface)==xmax_loc .OR. &
      family_f(iface)==ymin_loc .OR. &
      family_f(iface)==ymax_loc      ) THEN
    icell=cell2_f(iface)
    v_c_3(icell,:)=exact_v_c_3(icell,:)
  END IF
END DO
          
          

		
  CALL tracing_trace_end(trace_set_phi_initial)

END SUBROUTINE set_phi_initial
      
	

	
		
!! Use this routine to set up linked lists, to get sizes or to allocate fields 
!======================================================================
   SUBROUTINE set_phi_init3
!======================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_phi_init3)
      

		

		
  CALL tracing_trace_end(trace_set_phi_init3)

END SUBROUTINE set_phi_init3
      
	

	
		
!> Use *_init4 module for fielddata_init4,
!< equation_system_register_equation_system ... statements 
!======================================================================
   SUBROUTINE set_phi_init4
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_phi_init4)
      

		

		
  CALL tracing_trace_end(trace_set_phi_init4)
   
END SUBROUTINE set_phi_init4
    
	

	
		
! Use *_init5 module for all *_get_pointer statements 
!======================================================================
   SUBROUTINE set_phi_init5
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_phi_init5)
   
    CALL topology_get_size( nAllCells = nAllCells, &
                            nIntCells = nIntCells, &
                            NBndCells = NBndCells,&
                            nIntFaces = nIntFaces,&
                            nBndFaces = nBndFaces,&
                            nfamilies = nfamilies)
      

		
            
CALL topology_get_pointer(cell1_f=cell1_f,cell2_f=cell2_f,family_f=family_f )
CALL geometry_get_pointer(cent_c_3 = cent_c_3, cent_f_3 = cent_f_3)
CALL fielddata_get_pointer("Velocity", Field_3=v_c_3)
CALL fielddata_get_pointer("Pressure", Field=p_c)
CALL fielddata_get_pointer("ExactVelocity", Field_3=exact_v_c_3)
          
          

		
  CALL tracing_trace_end(trace_set_phi_init5)

END SUBROUTINE set_phi_init5
      
	
  
  
    
!! Use *_restart routines for restart
!======================================================================
   SUBROUTINE set_phi_restart
!======================================================================
      

    

    
  CALL tracing_trace_begin(trace_set_phi_restart)
      

    

    
  CALL tracing_trace_end(trace_set_phi_restart)

END SUBROUTINE set_phi_restart
      
  

	
		
!! This routine is executed at each timestep
!======================================================================
   SUBROUTINE set_phi_timestep(timestep,simultime)
!======================================================================

INTEGER   :: timestep
REAL(dp)  :: simultime
      

		

		
  refresco_not_used(timestep)  ! remove this when you use 'timestep'
  refresco_not_used(simultime) ! remove this when you use 'simultime'
  CALL tracing_trace_begin(trace_set_phi_timestep)
      

		

		
  CALL tracing_trace_end(trace_set_phi_timestep)

END SUBROUTINE set_phi_timestep
      
	

	
		
!! This routine is executed at each outer iteration
!======================================================================
   SUBROUTINE set_phi_outit(out_iter)
!======================================================================

INTEGER :: out_iter
      

		

		
  refresco_not_used(out_iter)  ! remove this when you use 'out_iter'
  CALL tracing_trace_begin(trace_set_phi_outit)
      

		

		
  CALL tracing_trace_end(trace_set_phi_outit)

END SUBROUTINE set_phi_outit
      
	

	
		
!> This routine is executed only once after the computation has stoped
!! due to max no of iteration, truncation error or stopfile. It is not 
!< executed when killfile is used
!======================================================================
   SUBROUTINE set_phi_final
!======================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_phi_final)
      

		

		
  CALL tracing_trace_end(trace_set_phi_final)

END SUBROUTINE set_phi_final
      
	

	
		
!! Use *_exit module for DEALLOCATE statements etc. 
!======================================================================
   SUBROUTINE set_phi_exit
!=====================================================================
      

		

		
  CALL tracing_trace_begin(trace_set_phi_exit)
      

		

		
  CALL tracing_trace_end(trace_set_phi_exit)

END SUBROUTINE set_phi_exit
      
	

	
	
	
END MODULE set_phi
    

          
            
            
          
        
